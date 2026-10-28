from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Face:
    box: tuple
    score: float
    landmarks: np.ndarray | None = None


class FaceDetector:
    def __init__(self, kind: str = "yunet", yunet_path=None, score_threshold: float = 0.7,
                 nms_threshold: float = 0.3, detect_width: int = 640, min_face: int = 40,
                 margin_yunet: float = 0.12, margin_haar: float = 0.0):
        self.kind = kind.lower()
        self.detect_width, self.min_face = int(detect_width), int(min_face)
        self._input_size = None
        has_haar = hasattr(cv2, "CascadeClassifier")
        if self.kind == "haar" and not has_haar:
            print(f"[detector] OpenCV {cv2.__version__} không còn Haar Cascade -> dùng YuNet.")
            self.kind = "yunet"
        if self.kind == "yunet":
            if yunet_path and Path(yunet_path).exists() and hasattr(cv2, "FaceDetectorYN"):
                self.net = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320),
                                                     score_threshold, nms_threshold, 5000)
            elif has_haar:
                print(f"[detector] CẢNH BÁO: không tìm thấy YuNet ({yunet_path}) "
                      "hoặc OpenCV < 4.8 -> chuyển sang Haar Cascade.")
                self.kind = "haar"
            else:
                raise FileNotFoundError(
                    f"Không tìm thấy model YuNet: {yunet_path}. Tải face_detection_yunet_2023mar.onnx "
                    "từ github.com/opencv/opencv_zoo (models/face_detection_yunet) vào weights/.")
        if self.kind == "haar":
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.crop_margin = margin_yunet if self.kind == "yunet" else margin_haar

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        H, W = frame_bgr.shape[:2]
        faces, scale = self._detect_at(frame_bgr, self.detect_width)
        if not faces and self.kind == "yunet" and W > self.detect_width // 2:
            faces, scale = self._detect_at(frame_bgr, self.detect_width // 2)
        out = []
        for f in faces:
            x, y, w, h = (np.asarray(f.box, dtype=np.float32) / scale).tolist()
            if min(w, h) < self.min_face:
                continue
            lm = None if f.landmarks is None else f.landmarks / scale
            out.append(Face((int(x), int(y), int(w), int(h)), f.score, lm))
        return out

    def _detect_at(self, frame_bgr, width):
        H, W = frame_bgr.shape[:2]
        scale = min(1.0, width / W)
        small = frame_bgr if scale == 1.0 else cv2.resize(
            frame_bgr, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
        faces = self._detect_yunet(small) if self.kind == "yunet" else self._detect_haar(small, scale)
        return faces, scale

    def _detect_yunet(self, img) -> list[Face]:
        h, w = img.shape[:2]
        if self._input_size != (w, h):
            self.net.setInputSize((w, h))
            self._input_size = (w, h)
        _, res = self.net.detect(img)
        if res is None:
            return []
        return [Face(tuple(r[:4]), float(r[14]), r[4:14].reshape(5, 2).astype(np.float32)) for r in res]

    def _detect_haar(self, img, scale) -> list[Face]:
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        m = max(20, int(self.min_face * scale))
        rects = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(m, m))
        return [Face(tuple(map(float, r)), 1.0, None) for r in rects]
