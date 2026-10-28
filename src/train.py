from __future__ import annotations

import copy
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from model import save_checkpoint


@dataclass
class AugConfig:
    hflip: float = 0.5
    rotate: float = 10.0
    scale: float = 0.10
    translate: float = 0.08
    brightness: float = 0.20
    contrast: float = 0.20
    erase_p: float = 0.25
    erase_area: tuple = (0.02, 0.12)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.10
    class_weight_power: float = 0.5
    pct_start: float = 0.15
    freeze_epochs: int = 0
    backbone_lr_mult: float = 1.0
    patience: int = 15
    amp: bool = True
    seed: int = 42
    aug: AugConfig = field(default_factory=AugConfig)


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def class_weights(y: np.ndarray, power: float = 0.5, num_classes: int = 7) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    return w * counts.sum() / (w * counts).sum()


def gpu_augment(x: torch.Tensor, a: AugConfig) -> torch.Tensor:
    B, _, H, W = x.shape
    dev = x.device
    u = lambda: torch.rand(B, device=dev) * 2 - 1

    if a.hflip > 0:
        flip = (torch.rand(B, device=dev) < a.hflip).view(B, 1, 1, 1)
        x = torch.where(flip, x.flip(-1), x)

    if a.rotate or a.scale or a.translate:
        ang = u() * math.radians(a.rotate)
        sc = 1 + u() * a.scale
        tx, ty = u() * a.translate * 2, u() * a.translate * 2
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, tx], 1),
                             torch.stack([sin, cos, ty], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    if a.brightness or a.contrast:
        b = (u() * a.brightness).view(B, 1, 1, 1)
        c = (1 + u() * a.contrast).view(B, 1, 1, 1)
        m = x.mean((2, 3), keepdim=True)
        x = ((x - m) * c + m + b).clamp(0, 1)

    if a.erase_p > 0:
        area = torch.empty(B, device=dev).uniform_(*a.erase_area) * H * W
        ratio = torch.exp(torch.empty(B, device=dev).uniform_(math.log(0.3), math.log(3.3)))
        h = (area * ratio).sqrt().clamp(max=H).view(B, 1, 1)
        w = (area / ratio).sqrt().clamp(max=W).view(B, 1, 1)
        cy = (torch.rand(B, device=dev) * H).view(B, 1, 1)
        cx = (torch.rand(B, device=dev) * W).view(B, 1, 1)
        yy = torch.arange(H, device=dev).view(1, H, 1)
        xx = torch.arange(W, device=dev).view(1, 1, W)
        box = ((yy - cy).abs() < h / 2) & ((xx - cx).abs() < w / 2)
        apply = (torch.rand(B, device=dev) < a.erase_p).view(B, 1, 1)
        mask = (box & apply).unsqueeze(1)
        x = torch.where(mask, torch.rand_like(x), x)
    return x


@torch.inference_mode()
def predict_logits(model: nn.Module, X: np.ndarray | torch.Tensor, device, batch: int = 1024) -> torch.Tensor:
    model.eval()
    Xt = X if torch.is_tensor(X) else torch.from_numpy(X)
    outs = []
    for i in range(0, len(Xt), batch):
        xb = Xt[i:i + batch].to(device, non_blocking=True).float().div_(255)
        if xb.ndim == 3:
            xb = xb.unsqueeze(1)
        outs.append(model(xb).float().cpu())
    return torch.cat(outs)


def evaluate(model: nn.Module, X: np.ndarray, device, batch: int = 1024) -> np.ndarray:
    return predict_logits(model, X, device, batch).softmax(1).numpy()


def _param_groups(params, lr: float, wd: float):
    params = [p for p in params if p.requires_grad]
    return [{"params": [p for p in params if p.ndim > 1], "lr": lr, "weight_decay": wd},
            {"params": [p for p in params if p.ndim <= 1], "lr": lr, "weight_decay": 0.0}]


def _set_backbone_trainable(model, flag: bool) -> None:
    for p in model.backbone_parameters():
        p.requires_grad_(flag)


def fit(model, data: dict, cfg: TrainConfig, device, ckpt_path, verbose: bool = True) -> dict:
    set_seed(cfg.seed)
    device = torch.device(device)
    model.to(device)

    Xtr = torch.from_numpy(data["train"][0]).unsqueeze(1).to(device)
    ytr = torch.from_numpy(data["train"][1]).to(device)
    Xva, yva = data["val"]
    yva_t = torch.from_numpy(yva)

    w = torch.tensor(class_weights(data["train"][1], cfg.class_weight_power), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=cfg.label_smoothing)
    crit_cpu = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    N, bs = len(ytr), cfg.batch_size
    steps = N // bs

    def make_optimizer(main_stage: bool):
        if not main_stage:
            return torch.optim.AdamW(_param_groups(model.head_parameters(), cfg.lr, cfg.weight_decay)), None
        if cfg.backbone_lr_mult != 1.0:
            groups = (_param_groups(model.backbone_parameters(), cfg.lr * cfg.backbone_lr_mult, cfg.weight_decay)
                      + _param_groups(model.head_parameters(), cfg.lr, cfg.weight_decay))
        else:
            groups = _param_groups(model.parameters(), cfg.lr, cfg.weight_decay)
        opt = torch.optim.AdamW(groups)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in groups], total_steps=(cfg.epochs - cfg.freeze_epochs) * steps,
            pct_start=cfg.pct_start, div_factor=25, final_div_factor=1e3)
        return opt, sched

    if cfg.freeze_epochs > 0:
        _set_backbone_trainable(model, False)
        opt, sched = make_optimizer(main_stage=False)
    else:
        opt, sched = make_optimizer(main_stage=True)

    hist = defaultdict(list)
    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    t_start = time.time()
    for epoch in range(cfg.epochs):
        if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs:
            _set_backbone_trainable(model, True)
            opt, sched = make_optimizer(main_stage=True)
            if verbose:
                print(f"--- Epoch {epoch + 1}: mở khoá backbone, bắt đầu fine-tune toàn bộ ---")

        model.train()
        t0 = time.time()
        perm = torch.randperm(N, device=device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb = gpu_augment(Xtr[idx].float().div_(255), cfg.aug)
            yb = ytr[idx]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(xb)
                loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            loss_sum += loss.detach()
            correct += (out.argmax(1) == yb).sum()

        logits = predict_logits(model, Xva, device)
        pred = logits.argmax(1).numpy()
        tr_loss, tr_acc = (loss_sum / steps).item(), (correct / (steps * bs)).item()
        va_loss = crit_cpu(logits, yva_t).item()
        va_acc = float((pred == yva).mean())
        va_f1 = float(f1_score(yva, pred, average="macro"))
        lr_now = opt.param_groups[-1]["lr"]
        for k, v in dict(train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                         val_acc=va_acc, val_f1=va_f1, lr=lr_now).items():
            hist[k].append(v)

        improved = va_f1 > best_f1
        if improved:
            best_f1, best_epoch, bad = va_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(model, ckpt_path, extra=dict(
                best_epoch=epoch + 1, val_acc=va_acc, val_macro_f1=va_f1,
                train_config={**asdict(cfg)}))
        else:
            bad += 1
        if verbose:
            print(f"Ep {epoch + 1:3d}/{cfg.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f} | lr {lr_now:.2e} | "
                  f"{time.time() - t0:5.1f}s {'*' if improved else ''}")
        if bad >= cfg.patience and epoch >= cfg.freeze_epochs:
            if verbose:
                print(f"Early stopping: {cfg.patience} epoch liền val F1 không cải thiện.")
            break

    model.load_state_dict(best_state)
    hist = dict(hist)
    hist["best_epoch"] = best_epoch
    hist["train_time_min"] = (time.time() - t_start) / 60
    if verbose:
        print(f"Xong sau {hist['train_time_min']:.1f} phút. Best epoch {best_epoch + 1}: "
              f"val F1 {best_f1:.4f}, val acc {hist['val_acc'][best_epoch]:.4f} -> {ckpt_path}")
    return hist
