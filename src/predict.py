from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASE = 48


FER_EYE_DIST = 0.43
FER_EYE_Y = 0.38


def crop_face(gray: np.ndarray, box, landmarks=None, margin: float = 0.12,
              align: bool = True, out_size: int = BASE) -> np.ndarray:
    x, y, w, h = box
    if landmarks is not None:
        e1, e2 = sorted([np.asarray(landmarks[0], float), np.asarray(landmarks[1], float)], key=lambda p: p[0])
        dist = float(np.hypot(*(e2 - e1)))
        if dist > 0.15 * max(w, h):
            angle = float(np.degrees(np.arctan2(e2[1] - e1[1], e2[0] - e1[0]))) if align else 0.0
            angle = float(np.clip(angle, -30, 30))
            side = dist / FER_EYE_DIST
            mx, my = (e1 + e2) / 2
            M = cv2.getRotationMatrix2D((float(mx), float(my)), angle, 1.0)
            M[0, 2] += side / 2 - mx
            M[1, 2] += side * FER_EYE_Y - my
            return _warp_square(gray, M, side, out_size)
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1 + 2 * margin)
    M = np.float32([[1, 0, side / 2 - cx], [0, 1, side / 2 - cy]])
    return _warp_square(gray, M, side, out_size)


def _warp_square(gray, M, side, out_size):
    s = max(int(round(side)), 1)
    face = cv2.warpAffine(gray, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_AREA)


def to_face48(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape != (BASE, BASE):
        interp = cv2.INTER_AREA if min(img.shape) > BASE else cv2.INTER_LINEAR
        img = cv2.resize(img, (BASE, BASE), interpolation=interp)
    return img


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class EmotionPredictor:

    def __init__(self, weights_path, backend: str = "onnx", num_threads: int | None = None):
        p = Path(weights_path)
        if not p.is_absolute():
            p = ROOT / p
        meta_path, onnx_path = p.with_suffix(".json"), p.with_suffix(".onnx")
        if not p.exists() and not onnx_path.exists():
            raise FileNotFoundError(f"Không tìm thấy weights: {p} (hãy chạy notebooks/train.ipynb trước)")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.classes = self.meta.get("classes",
                                     ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
        self.name = self.meta.get("arch", p.stem)
        self.backend = None

        if backend == "onnx" and onnx_path.exists():
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                if num_threads:
                    so.intra_op_num_threads = num_threads
                self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
                self.input_name = self.sess.get_inputs()[0].name
                self.backend = "onnx"
            except Exception as e:
                print(f"[predict] Không mở được ONNX ({e}) -> dùng PyTorch")
        if self.backend is None:
            import torch
            from model import load_model
            if num_threads:
                torch.set_num_threads(num_threads)
            self._torch = torch
            self.model = load_model(p, "cpu")
            self.backend = "torch"

    def predict_batch(self, faces48: list[np.ndarray]) -> np.ndarray:
        if not faces48:
            return np.zeros((0, len(self.classes)), np.float32)
        x = (np.stack(faces48).astype(np.float32) / 255.0)[:, None]
        if self.backend == "onnx":
            logits = self.sess.run(None, {self.input_name: x})[0]
        else:
            with self._torch.inference_mode():
                logits = self.model(self._torch.from_numpy(x)).numpy()
        return _softmax(logits.astype(np.float32))

    def predict(self, face_img: np.ndarray):
        probs = self.predict_batch([to_face48(face_img)])[0]
        k = int(probs.argmax())
        return self.classes[k], float(probs[k]), probs


def main():
    ap = argparse.ArgumentParser(description="Dự đoán cảm xúc cho 1 ảnh khuôn mặt đã crop")
    ap.add_argument("image")
    ap.add_argument("--model", choices=["cnn", "transfer"], default="cnn")
    ap.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    args = ap.parse_args()
    from data import imread_gray
    img = imread_gray(args.image)
    if img is None:
        sys.exit(f"[LỖI] Không đọc được ảnh: {args.image}")
    pred = EmotionPredictor(ROOT / "weights" / f"{args.model}.pt", args.backend)
    label, conf, probs = pred.predict(img)
    print(f"{label} ({conf:.3f})")
    for c, p in sorted(zip(pred.classes, probs), key=lambda t: -t[1]):
        print(f"  {c:9s} {p:.3f}")


if __name__ == "__main__":
    main()
