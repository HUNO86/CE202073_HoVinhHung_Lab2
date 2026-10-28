from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import yaml

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

from detector import FaceDetector
from predict import EmotionPredictor, crop_face

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://")
COLORS = {
    "Angry": (40, 40, 230), "Disgust": (40, 150, 40), "Fear": (160, 60, 160),
    "Happy": (0, 210, 255), "Sad": (210, 120, 30), "Surprise": (0, 140, 255),
    "Neutral": (200, 200, 200),
}
LOW_CONF_COLOR = (128, 128, 128)


def load_config(path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"[app] Không thấy {path}, dùng cấu hình mặc định.")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


_IP_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3})(?::(\d{1,5}))?(/.*)?$")


def normalize_stream(s: str) -> str | None:
    m = _IP_RE.match(s)
    if not m:
        return None
    ip, port, path = m.group(1), m.group(2) or "8080", m.group(3) or "/video"
    return f"http://{ip}:{port}{path}"


def classify_source(src: str):
    s = str(src).strip().strip('"').strip("'")
    if s.isdigit():
        return "webcam", int(s)
    if s.lower().startswith(STREAM_PREFIXES):
        if s.lower().startswith(("http://", "https://")) and not urlparse(s).path.strip("/"):
            s = s.rstrip("/") + "/video"
        return "stream", s
    url = normalize_stream(s)
    if url:
        return "stream", url
    p = Path(s)
    if not p.exists() and (ROOT / s).exists():
        p = ROOT / s
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {s}")
    return ("image" if p.suffix.lower() in IMAGE_EXTS else "video"), str(p)


def imread_any(path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


class CameraStream:

    def __init__(self, src, open_timeout_ms=4000, read_timeout_ms=4000, reconnect_delay=1.0,
                 stale_timeout=2.5):
        self.src = src
        self.open_ms, self.read_ms = int(open_timeout_ms), int(read_timeout_ms)
        self.reconnect_delay, self.stale_timeout = float(reconnect_delay), float(stale_timeout)
        self.frame, self.frame_id, self.last_ok = None, 0, 0.0
        self.attempts = 0
        self.lock = threading.Lock()
        self.stopped = False
        self.cap = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self):
        self.attempts += 1
        if isinstance(self.src, int):
            api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.src, api)
        else:
            try:
                cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_ms,
                                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_ms])
            except (cv2.error, TypeError):
                cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                if self.cap is not None:
                    self.cap.release()
                self.cap = self._open()
                if not self.cap.isOpened():
                    time.sleep(self.reconnect_delay)
                    continue
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame, self.last_ok = frame, time.time()
                    self.frame_id += 1
            else:
                self.cap.release()
                self.cap = None
                time.sleep(self.reconnect_delay)

    def read(self):
        with self.lock:
            alive = self.frame is not None and (time.time() - self.last_ok) < self.stale_timeout
            return alive, (None if self.frame is None else self.frame.copy()), self.frame_id

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / float(aw * ah + bw * bh - inter + 1e-6)


class Track:
    _next_id = 0

    def __init__(self, face):
        self.id = Track._next_id
        Track._next_id += 1
        self.face, self.probs, self.missed = face, None, 0


class FaceTracker:

    def __init__(self, iou_thr=0.3, max_missed=3):
        self.iou_thr, self.max_missed = iou_thr, max_missed
        self.tracks: list[Track] = []

    def update(self, faces):
        pairs = sorted(((iou(t.face.box, f.box), ti, fi) for ti, t in enumerate(self.tracks)
                        for fi, f in enumerate(faces)), reverse=True)
        used_t, used_f = set(), set()
        for s, ti, fi in pairs:
            if s < self.iou_thr:
                break
            if ti in used_t or fi in used_f:
                continue
            self.tracks[ti].face, self.tracks[ti].missed = faces[fi], 0
            used_t.add(ti)
            used_f.add(fi)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        self.tracks += [Track(f) for fi, f in enumerate(faces) if fi not in used_f]

    def active(self):
        return [t for t in self.tracks if t.missed == 0]

    def reset(self):
        self.tracks = []


def put_label(img, text, org, color, scale=0.6, thick=2, text_color=(0, 0, 0)):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thick, cv2.LINE_AA)


def draw_banner(img, text, color=(0, 0, 200)):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h // 2 - 30), (w, h // 2 + 30), color, -1)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, text, (max(5, (w - tw) // 2), h // 2 + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)


class EmotionPipeline:
    def __init__(self, detector: FaceDetector, predictor: EmotionPredictor, conf_threshold=0.4,
                 alpha=0.5, detect_every=2, align=True):
        self.det, self.pred = detector, predictor
        self.conf_threshold, self.alpha = float(conf_threshold), float(alpha)
        self.detect_every, self.align = max(1, int(detect_every)), bool(align)
        self.tracker = FaceTracker(max_missed=max(2, self.detect_every))
        self.frame_idx = 0
        self.last_crops: list[np.ndarray] = []

    def process(self, frame, force_detect=False, smooth=True):
        if force_detect or self.frame_idx % self.detect_every == 0:
            self.tracker.update(self.det.detect(frame))
        self.frame_idx += 1
        tracks = self.tracker.active()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crops = [crop_face(gray, t.face.box, t.face.landmarks, self.det.crop_margin, self.align)
                 for t in tracks]
        self.last_crops = crops
        probs = self.pred.predict_batch(crops)

        results = []
        for t, p in zip(tracks, probs):
            t.probs = p if (t.probs is None or not smooth) else self.alpha * p + (1 - self.alpha) * t.probs
            k = int(t.probs.argmax())
            label, conf = self.pred.classes[k], float(t.probs[k])
            results.append(dict(id=t.id, box=t.face.box, label=label, conf=conf, det_score=t.face.score))
            self._draw_face(frame, t.face.box, label, conf)
        return frame, results

    def _draw_face(self, img, box, label, conf):
        x, y, w, h = box
        low = conf < self.conf_threshold
        color = LOW_CONF_COLOR if low else COLORS.get(label, (0, 255, 0))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        put_label(img, f"{label}{'?' if low else ''} {conf:.2f}", (x, y - 2), color)


def draw_hud(img, fps, model_name, backend, det_kind, n_faces, show_crops, crops):
    put_label(img, f"FPS: {fps:5.1f} | faces: {n_faces}", (8, 30), (30, 30, 30), 0.65, 2, (0, 255, 0))
    put_label(img, f"{model_name} ({backend}) | {det_kind}", (8, 58), (30, 30, 30), 0.5, 1, (255, 255, 255))
    if show_crops:
        for i, c in enumerate(crops[:4]):
            big = cv2.resize(c, (96, 96), interpolation=cv2.INTER_NEAREST)
            y0 = 70 + i * 100
            if y0 + 96 <= img.shape[0] and 104 <= img.shape[1]:
                img[y0:y0 + 96, 8:104] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def fit(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    return img if s == 1.0 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


class Display:
    def __init__(self, title, enabled=True, max_width=1280, max_height=900):
        self.title, self.enabled = title, enabled
        self.max_width, self.max_height = max_width, max_height
        if enabled:
            try:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            except cv2.error as e:
                raise RuntimeError(
                    "OpenCV trong môi trường này KHÔNG có giao diện cửa sổ (đang cài bản headless).\n"
                    "Sửa bằng 2 lệnh sau rồi chạy lại:\n"
                    "    pip uninstall -y opencv-python-headless opencv-contrib-python-headless "
                    "opencv-contrib-python opencv-python\n"
                    "    pip install opencv-python\n"
                    "(Hoặc thêm --no-display --save out.jpg để chạy không cần cửa sổ.)") from e

    def show(self, img, wait=1):
        if not self.enabled:
            return -1
        img = fit(img, self.max_width, self.max_height)
        cv2.imshow(self.title, img)
        return cv2.waitKey(wait) & 0xFF

    def closed(self):
        if not self.enabled:
            return False
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True


def save_image(img, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return path


def run_image(path, pipe, disp, args):
    frame = imread_any(path)
    if frame is None:
        print(f"[LỖI] Không đọc được ảnh (file hỏng hoặc không phải ảnh): {path}")
        return 1
    frame = fit(frame, 1280, 1280)
    t0 = time.perf_counter()
    out, res = pipe.process(frame, force_detect=True, smooth=False)
    dt = time.perf_counter() - t0
    draw_hud(out, 1.0 / max(dt, 1e-6), pipe.pred.name, pipe.pred.backend, pipe.det.kind, len(res),
             args.show_crops, pipe.last_crops)
    if not res:
        draw_banner(out, "No face detected", (60, 60, 60))
        print("[app] Không phát hiện khuôn mặt nào trong ảnh.")
    for r in res:
        print(f"  Mặt #{r['id']} box={r['box']}: {r['label']} ({r['conf']:.3f})")
    save_to = Path(args.save) if args.save else ROOT / "outputs" / f"{Path(path).stem}_{args.model}.jpg"
    print(f"[app] Đã lưu kết quả: {save_image(out, save_to)}  ({dt * 1000:.0f} ms)")
    if disp.enabled:
        print("[app] Nhấn phím bất kỳ (hoặc đóng cửa sổ) để thoát.")
        while not disp.closed():
            if disp.show(out, 50) != 255:
                break
    return 0


def run_video(src, kind, pipe, disp, args, cam_cfg):
    stream = None
    cap = None
    writer = None
    if kind == "video":
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[LỖI] Không mở được file video: {src}")
            return 1
    else:
        print(f"[app] Đang kết nối camera: {src}  (q/ESC để thoát)")
        stream = CameraStream(src, cam_cfg.get("open_timeout_ms", 4000), cam_cfg.get("read_timeout_ms", 4000),
                              cam_cfg.get("reconnect_delay", 1.0), cam_cfg.get("stale_timeout", 2.5))

    fps, n, t_total = 0.0, 0, 0.0
    last_id, last_frame, was_alive = -1, None, None
    t_prev = None
    t_wait, warned = time.time(), False
    try:
        while True:
            if kind == "video":
                ok, frame = cap.read()
                if not ok:
                    print("[app] Hết video.")
                    break
                alive = True
            else:
                alive, frame, fid = stream.read()
                if alive != was_alive:
                    if alive:
                        print("[app] Đã kết nối camera.")
                    elif was_alive:
                        print("[app] MẤT KẾT NỐI CAMERA - đang thử kết nối lại... "
                              "(kiểm tra Wi-Fi / app IP camera trên điện thoại)")
                        pipe.tracker.reset()
                    was_alive = alive
                if alive and fid == last_id:
                    key = disp.show(last_frame, 5) if last_frame is not None else cv2.waitKey(5) & 0xFF
                    if key in (ord("q"), 27) or disp.closed():
                        break
                    continue
                last_id = fid if alive else last_id

            if not alive:
                base = (frame if frame is not None else np.zeros((480, 640, 3), np.uint8))
                vis = (base * 0.4).astype(np.uint8)
                msg = ("Connecting to camera..." if frame is None
                       else "CAMERA DISCONNECTED - reconnecting...")
                draw_banner(vis, msg)
                if frame is None and time.time() - t_wait > 8:
                    if not warned:
                        print("[app] Chưa kết nối được camera sau 8 giây. Kiểm tra: URL đúng (có /video), "
                              "laptop và điện thoại cùng Wi-Fi, app IP camera đang Start server, tường lửa.")
                        warned = True
                    put_label(vis, "Check URL / Wi-Fi / phone app", (10, vis.shape[0] - 20), (0, 0, 200), 0.6, 2,
                              (255, 255, 255))
                t_prev = None
                key = disp.show(vis, 30)
                if not disp.enabled:
                    time.sleep(0.03)
                if key in (ord("q"), 27) or disp.closed():
                    break
                if key == ord("s"):
                    print(f"[app] Đã lưu ảnh: {save_image(vis, ROOT / 'outputs' / f'snap_{int(time.time())}.jpg')}")
                continue

            out, res = pipe.process(frame)
            now = time.perf_counter()
            if t_prev is not None:
                dt = now - t_prev
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / max(dt, 1e-6)
                n += 1
                t_total += dt
            t_prev = now
            draw_hud(out, fps, pipe.pred.name, pipe.pred.backend, pipe.det.kind, len(res),
                     args.show_crops, pipe.last_crops)
            last_frame = out

            if args.save:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                             (out.shape[1], out.shape[0]))
                writer.write(out)

            key = disp.show(out, 1)
            if key in (ord("q"), 27) or disp.closed():
                break
            if key == ord("s"):
                print(f"[app] Đã lưu ảnh: {save_image(out, ROOT / 'outputs' / f'snap_{int(time.time())}.jpg')}")
            if key == ord("c"):
                args.show_crops = not args.show_crops
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[app] Dừng bởi người dùng.")
    finally:
        if stream:
            stream.release()
        if cap:
            cap.release()
        if writer:
            writer.release()
    if n:
        print(f"[app] Đã xử lý {n} khung hình, FPS trung bình: {n / t_total:.1f} "
              f"(không tính thời gian chờ kết nối / mất kết nối)")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Nhận diện cảm xúc khuôn mặt realtime")
    ap.add_argument("--source", default=None,
                    help="IP:cổng điện thoại (192.168.1.31:8080), URL đầy đủ, đường dẫn ảnh/video, hoặc số webcam (0). "
                         "Bỏ trống = lấy camera.source trong config.yaml")
    ap.add_argument("--model", choices=["cnn", "transfer"], default="cnn", help="cnn = M1, transfer = M2")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", default=None, help="Ghi đè đường dẫn weights trong config")
    ap.add_argument("--backend", choices=["onnx", "torch"], default=None)
    ap.add_argument("--detector", choices=["yunet", "haar"], default=None)
    ap.add_argument("--conf", type=float, default=None, help="Ngưỡng confidence")
    ap.add_argument("--save", default=None, help="Lưu kết quả ra file (.jpg cho ảnh, .mp4 cho video)")
    ap.add_argument("--show-crops", action="store_true", help="Hiện ảnh 48×48 đưa vào mô hình")
    ap.add_argument("--no-display", action="store_true", help="Không mở cửa sổ (chạy nền / test)")
    ap.add_argument("--max-frames", type=int, default=0, help="Dừng sau N khung hình (để test)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    cam_cfg, det_cfg = cfg.get("camera", {}), cfg.get("detector", {})
    inf_cfg, disp_cfg = cfg.get("inference", {}), cfg.get("display", {})

    source = args.source or cam_cfg.get("source")
    if source is None:
        print("[LỖI] Chưa có nguồn vào: dùng --source hoặc đặt camera.source trong config.yaml")
        return 2
    try:
        kind, src = classify_source(str(source))
    except FileNotFoundError as e:
        print(f"[LỖI] {e}")
        return 2

    weights = args.weights or cfg.get("models", {}).get(args.model, f"weights/{args.model}.pt")
    try:
        predictor = EmotionPredictor(rpath(weights), args.backend or inf_cfg.get("backend", "onnx"),
                                     inf_cfg.get("num_threads"))
    except FileNotFoundError as e:
        print(f"[LỖI] {e}")
        return 2

    try:
        detector = FaceDetector(
            kind=args.detector or det_cfg.get("type", "yunet"),
            yunet_path=rpath(det_cfg.get("yunet_path", "weights/face_detection_yunet_2023mar.onnx")),
            score_threshold=det_cfg.get("score_threshold", 0.7), nms_threshold=det_cfg.get("nms_threshold", 0.3),
            detect_width=det_cfg.get("detect_width", 640), min_face=det_cfg.get("min_face", 40),
            margin_yunet=det_cfg.get("margin_yunet", 0.12), margin_haar=det_cfg.get("margin_haar", 0.0))
    except FileNotFoundError as e:
        print(f"[LỖI] {e}")
        return 2
    pipe = EmotionPipeline(detector, predictor,
                           conf_threshold=args.conf if args.conf is not None else inf_cfg.get("conf_threshold", 0.4),
                           alpha=inf_cfg.get("smoothing_alpha", 0.5),
                           detect_every=det_cfg.get("detect_every", 2), align=det_cfg.get("align", True))
    print(f"[app] Nguồn: {kind} | Mô hình: {args.model} ({predictor.name}, {predictor.backend}) | "
          f"Detector: {detector.kind}")

    try:
        disp = Display(disp_cfg.get("window", "Emotion Recognition"), not args.no_display,
                       disp_cfg.get("max_width", 1280), disp_cfg.get("max_height", 900))
    except RuntimeError as e:
        print(f"[LỖI] {e}")
        return 3
    try:
        if kind == "image":
            return run_image(src, pipe, disp, args)
        return run_video(src, kind, pipe, disp, args, cam_cfg)
    finally:
        if disp.enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
