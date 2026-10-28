from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
_NAME_TO_IDX = {c.lower(): i for i, c in enumerate(CLASSES)}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def imread_gray(path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None


def _load_csv(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    X = np.stack([np.asarray(p.split(), dtype=np.uint8) for p in df["pixels"]]).reshape(-1, 48, 48)
    y = df["emotion"].to_numpy(np.int64)
    usage = df["Usage"].astype(str).to_numpy() if "Usage" in df.columns else None
    return X, y, usage


def _load_folder(root: Path):
    Xs, ys = [], []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        idx = _NAME_TO_IDX.get(cls_dir.name.lower())
        if idx is None:
            print(f"[data] Bỏ qua thư mục không phải lớp cảm xúc: {cls_dir.name}")
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in _IMG_EXTS:
                continue
            img = imread_gray(f)
            if img is None:
                continue
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
            Xs.append(img)
            ys.append(idx)
    return np.stack(Xs), np.asarray(ys, dtype=np.int64)


def _split(X, y, ratio, seed):
    return train_test_split(X, y, test_size=ratio, stratify=y, random_state=seed)


def load_fer(path, val_ratio: float = 0.1, seed: int = 42, drop_blank: bool = True,
             use_cache: bool = True) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy dữ liệu tại {path}. Đặt fer2013.csv hoặc thư mục fer2013/ (train/, test/) vào data/.")
    cache = path.parent / f"{path.stem}_cache_v{int(val_ratio * 100)}_s{seed}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        out = {s: (z[f"X_{s}"], z[f"y_{s}"]) for s in ("train", "val", "test")}
        _describe(out, f"cache {cache.name}")
        return out

    if path.is_file():
        X, y, usage = _load_csv(path)
        if usage is not None and {"Training", "PublicTest", "PrivateTest"} <= set(usage):
            out = {"train": (X[usage == "Training"], y[usage == "Training"]),
                   "val": (X[usage == "PublicTest"], y[usage == "PublicTest"]),
                   "test": (X[usage == "PrivateTest"], y[usage == "PrivateTest"])}
            src = "CSV: Training / PublicTest / PrivateTest"
        else:
            Xtr, Xte, ytr, yte = _split(X, y, 0.1, seed)
            Xtr, Xva, ytr, yva = _split(Xtr, ytr, val_ratio / 0.9, seed)
            out = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
            src = "CSV không có Usage: tự chia stratified"
    else:
        Xtr, ytr = _load_folder(path / "train")
        Xtr, Xva, ytr, yva = _split(Xtr, ytr, val_ratio, seed)
        if (path / "test").exists():
            Xte, yte = _load_folder(path / "test")
        else:
            Xtr, Xte, ytr, yte = _split(Xtr, ytr, 0.1, seed)
        out = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
        src = f"thư mục ảnh, val = {val_ratio:.0%} train (stratified)"

    if drop_blank:
        Xtr, ytr = out["train"]
        keep = Xtr.reshape(len(Xtr), -1).std(1) > 2.0
        if (~keep).sum():
            print(f"[data] Loại {(~keep).sum()} ảnh trơn (không có nội dung) khỏi tập train")
        out["train"] = (Xtr[keep], ytr[keep])

    if use_cache:
        np.savez_compressed(cache, **{f"X_{s}": v[0] for s, v in out.items()},
                            **{f"y_{s}": v[1] for s, v in out.items()})
    _describe(out, src)
    return out


def _describe(out: dict, src: str) -> None:
    print(f"[data] Nguồn: {src}")
    for s, (X, y) in out.items():
        print(f"  {s:5s}: {len(y):6d} ảnh | " +
              " ".join(f"{c[:3]}={n}" for c, n in zip(CLASSES, np.bincount(y, minlength=7))))
