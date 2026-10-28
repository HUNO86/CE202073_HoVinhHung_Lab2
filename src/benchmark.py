from __future__ import annotations

import copy
import time

import numpy as np
import torch


def _bench(fn, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ms = float(np.median(ts) * 1000)
    return {"latency_ms": ms, "fps": 1000.0 / ms}


def latency_torch(model, batch: int = 1, runs: int = 300, warmup: int = 50) -> dict:
    m = copy.deepcopy(model).cpu().eval()
    x = torch.rand(batch, 1, 48, 48)

    def fn():
        with torch.inference_mode():
            m(x)
    return _bench(fn, runs, warmup)


def latency_onnx(path, batch: int = 1, runs: int = 300, warmup: int = 50) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(batch, 1, 48, 48).astype(np.float32)
    return _bench(lambda: sess.run(None, {name: x}), runs, warmup)


def onnx_max_diff(model, path, n: int = 8) -> float:
    import onnxruntime as ort
    m = copy.deepcopy(model).cpu().eval()
    x = torch.rand(n, 1, 48, 48)
    with torch.inference_mode():
        ref = m(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    return float(np.abs(ref - out).max())
