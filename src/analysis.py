from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import CLASSES
from train import gpu_augment


def _save(fig, save):
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")


def plot_class_distribution(data: dict, save=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    k, width = len(data), 0.8 / len(data)
    for i, (split, (_, y)) in enumerate(data.items()):
        c = np.bincount(y, minlength=7)
        bars = ax.bar(np.arange(7) + (i - (k - 1) / 2) * width, c, width, label=f"{split} ({len(y)})")
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(range(7), CLASSES)
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố lớp FER-2013")
    ax.legend()
    _save(fig, save)
    plt.show()
    y = data["train"][1]
    c = np.bincount(y, minlength=7)
    print("Train:  " + " | ".join(f"{n}: {v} ({v / c.sum():.1%})" for n, v in zip(CLASSES, c)))
    print(f"Tỉ lệ mất cân bằng lớn nhất / nhỏ nhất: {c.max() / c.min():.1f} lần "
          f"({CLASSES[c.argmax()]} / {CLASSES[c.argmin()]})")


def show_augmentations(X: np.ndarray, aug, device, n_img=6, n_aug=5, save=None, seed=0):
    torch.manual_seed(seed)
    x = torch.from_numpy(X[:n_img]).float().div(255).unsqueeze(1).to(device)
    rows = [x] + [gpu_augment(x, aug) for _ in range(n_aug)]
    fig, axes = plt.subplots(n_img, n_aug + 1, figsize=(1.4 * (n_aug + 1), 1.4 * n_img))
    for c, r in enumerate(rows):
        for i in range(n_img):
            ax = axes[i, c]
            ax.imshow(r[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title("Gốc" if c == 0 else f"Aug {c}", fontsize=8)
    plt.tight_layout()
    _save(fig, save)
    plt.show()


def plot_history(hist: dict, title: str, save=None):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    best = hist.get("best_epoch", int(np.argmax(hist["val_f1"]))) + 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(ep, hist["train_loss"], label="train")
    a1.plot(ep, hist["val_loss"], label="validation")
    a1.set_title(f"{title} — Loss")
    a2.plot(ep, hist["train_acc"], label="train acc")
    a2.plot(ep, hist["val_acc"], label="val acc")
    a2.plot(ep, hist["val_f1"], "--", label="val macro-F1")
    a2.set_title(f"{title} — Accuracy / F1")
    for a in (a1, a2):
        a.axvline(best, color="gray", ls=":", lw=1, label=f"best epoch {best}")
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
        a.legend()
    plt.tight_layout()
    _save(fig, save)
    plt.show()


def summarize(y: np.ndarray, probs: np.ndarray, name: str) -> dict:
    pred = probs.argmax(1)
    acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
    per_class = f1_score(y, pred, average=None, labels=range(7))
    print(f"===== {name} trên tập TEST ({len(y)} ảnh) =====")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print(classification_report(y, pred, labels=range(7), target_names=CLASSES, digits=3, zero_division=0))
    return dict(name=name, accuracy=float(acc), macro_f1=float(f1),
                per_class_f1={c: float(v) for c, v in zip(CLASSES, per_class)})


def plot_confusion(y: np.ndarray, pred: np.ndarray, title: str, save=None):
    cm = confusion_matrix(y, pred, labels=range(7))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, mat, fmt, sub in ((axes[0], cm, "d", "số lượng"), (axes[1], cmn, ".2f", "chuẩn hoá theo hàng = recall")):
        im = ax.imshow(mat, cmap="Blues")
        thr = mat.max() / 2
        for i in range(7):
            for j in range(7):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > thr else "black")
        ax.set_xticks(range(7), CLASSES, rotation=45)
        ax.set_yticks(range(7), CLASSES)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Nhãn thật")
        ax.set_title(f"{title} — {sub}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    _save(fig, save)
    plt.show()
    off = cmn.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off.ravel()), off.shape))[0][:5]
    print("Các cặp nhầm lẫn nhiều nhất (thật -> dự đoán, % của lớp thật):")
    for i, j in top:
        print(f"  {CLASSES[i]:8s} -> {CLASSES[j]:8s}: {off[i, j]:.1%} ({cm[i, j]} ảnh)")


def show_misclassified(X, y, probs, n=16, mode="confident", save=None, seed=0):
    pred, conf = probs.argmax(1), probs.max(1)
    wrong = np.where(pred != y)[0]
    if mode == "confident":
        idx = wrong[np.argsort(-conf[wrong])][:n]
    else:
        idx = np.random.default_rng(seed).choice(wrong, size=min(n, len(wrong)), replace=False)
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    for ax in np.ravel(axes):
        ax.axis("off")
    for k, (ax, i) in enumerate(zip(np.ravel(axes), idx)):
        ax.imshow(X[i], cmap="gray")
        ax.set_title(f"#{k + 1} (idx {i})\nThật: {CLASSES[y[i]]}\nĐoán: {CLASSES[pred[i]]} ({conf[i]:.2f})",
                     fontsize=8, color="crimson")
    plt.tight_layout()
    _save(fig, save)
    plt.show()
    return idx


def plot_per_class_f1(results: list[dict], save=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(results)
    for i, r in enumerate(results):
        v = [r["per_class_f1"][c] for c in CLASSES]
        bars = ax.bar(np.arange(7) + (i - (len(results) - 1) / 2) * w, v, w, label=r["name"])
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(range(7), CLASSES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("F1 theo từng lớp trên tập test")
    ax.legend()
    _save(fig, save)
    plt.show()
