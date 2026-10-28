from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

FER_MEAN, FER_STD = 0.5077, 0.2550
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BASE_SIZE = 48


class InputAdapter(nn.Module):

    def __init__(self, size: int, channels: int, mean, std):
        super().__init__()
        self.size, self.channels = int(size), int(channels)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.size != BASE_SIZE:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        if self.channels == 3:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std


def conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class EmotionCNN(nn.Module):

    def __init__(self, num_classes: int = 7, widths=(64, 128, 256, 512),
                 drops=(0.10, 0.20, 0.30, 0.40), head_dropout: float = 0.5):
        super().__init__()
        blocks, cin = [], 1
        for w, d in zip(widths, drops):
            blocks.append(nn.Sequential(
                conv_bn_relu(cin, w), conv_bn_relu(w, w),
                nn.MaxPool2d(2), nn.Dropout2d(d),
            ))
            cin = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(cin, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_backbone(arch: str, num_classes: int, pretrained: bool, head_dropout: float):
    from torchvision import models

    arch = arch.lower()
    if arch in ("resnet18", "resnet50"):
        ctor = getattr(models, arch)
        weights = (models.ResNet18_Weights.IMAGENET1K_V1 if arch == "resnet18"
                   else models.ResNet50_Weights.IMAGENET1K_V2) if pretrained else None
        net = ctor(weights=weights)
        net.fc = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(net.fc.in_features, num_classes))
        head = net.fc
    elif arch in ("mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0"):
        ctor = getattr(models, arch)
        net = ctor(weights="DEFAULT" if pretrained else None)
        last = net.classifier[-1]
        net.classifier[-1] = nn.Linear(last.in_features, num_classes)
        head = net.classifier
    else:
        raise ValueError(f"Kiến trúc chưa hỗ trợ: {arch}")
    return net, head


class EmotionModel(nn.Module):
    def __init__(self, kind: str, arch: str, input_size: int, channels: int, mean, std,
                 num_classes: int = 7, pretrained: bool = False, head_dropout: float = 0.5):
        super().__init__()
        self.meta = dict(kind=kind, arch=arch, input_size=int(input_size), channels=int(channels),
                         mean=list(map(float, mean)), std=list(map(float, std)),
                         num_classes=int(num_classes), head_dropout=float(head_dropout),
                         classes=CLASSES[:num_classes], base_size=BASE_SIZE)
        self.adapter = InputAdapter(input_size, channels, mean, std)
        if kind == "cnn":
            self.net = EmotionCNN(num_classes, head_dropout=head_dropout)
            self._head = self.net.head
        else:
            self.net, self._head = build_backbone(arch, num_classes, pretrained, head_dropout)

    def forward(self, x):
        return self.net(self.adapter(x))

    def head_parameters(self):
        return list(self._head.parameters())

    def backbone_parameters(self):
        head_ids = {id(p) for p in self.head_parameters()}
        return [p for p in self.parameters() if id(p) not in head_ids]


def build_model(kind: str = "cnn", **kw) -> EmotionModel:
    if kind == "cnn":
        cfg = dict(arch="custom_cnn", input_size=48, channels=1,
                   mean=(FER_MEAN,), std=(FER_STD,), head_dropout=0.5)
    elif kind == "transfer":
        cfg = dict(arch="resnet18", input_size=112, channels=3,
                   mean=IMAGENET_MEAN, std=IMAGENET_STD, head_dropout=0.3, pretrained=True)
    else:
        raise ValueError("kind phải là 'cnn' hoặc 'transfer'")
    cfg.update(kw)
    return EmotionModel(kind=kind, **cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: EmotionModel, path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(model.meta)
    if extra:
        meta.update(extra)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"meta": model.meta, "state_dict": state}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model(path, device="cpu") -> EmotionModel:
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    m = dict(ckpt["meta"])
    model = EmotionModel(kind=m["kind"], arch=m["arch"], input_size=m["input_size"],
                         channels=m["channels"], mean=m["mean"], std=m["std"],
                         num_classes=m["num_classes"], pretrained=False,
                         head_dropout=m.get("head_dropout", 0.5))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def export_onnx(model: EmotionModel, path, opset: int = 17) -> Path:
    path = Path(path)
    m = copy.deepcopy(model).cpu().eval()
    dummy = torch.rand(1, 1, BASE_SIZE, BASE_SIZE)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  do_constant_folding=True)
    try:
        torch.onnx.export(m, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(m, dummy, str(path), **kwargs)
    return path
