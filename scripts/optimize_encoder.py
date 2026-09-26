"""
Export the KinematicEncoder to ONNX, apply dynamic INT8 weight quantization, and benchmark
PyTorch FP32 vs ONNX Runtime FP32 vs ONNX Runtime INT8 on CPU.

    python scripts/optimize_encoder.py                       # architecture benchmark (seeded init)
    python scripts/optimize_encoder.py --checkpoint enc.pt   # same, on trained weights

Without --checkpoint the weights are seeded random, so size and latency are real properties of
the architecture while output fidelity should be re-checked on the trained checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from suturing_pipeline.synthesis.kinematic_encoder import KinematicEncoder  # noqa: E402


def export(model: torch.nn.Module, path: Path, kin_dim: int) -> None:
    kin, lab = torch.randn(2, kin_dim), torch.tensor([0, 1])
    torch.onnx.export(model, (kin, lab), str(path), input_names=["kinematics", "gesture_label"],
                      output_names=["embeddings"], opset_version=17, dynamo=False,
                      dynamic_axes={"kinematics": {0: "batch"}, "gesture_label": {0: "batch"}, "embeddings": {0: "batch"}})


def quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def session(path: Path, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def timeit(fn, runs: int) -> dict:
    for _ in range(10):
        fn()
    t = []
    for _ in range(runs):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    return {"p50_ms": round(float(np.percentile(t, 50)), 3), "p95_ms": round(float(np.percentile(t, 95)), 3)}


def fidelity(ref: np.ndarray, out: np.ndarray) -> dict:
    a, b = ref.reshape(len(ref), -1), out.reshape(len(out), -1)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {"cosine_mean": round(float(cos.mean()), 6), "cosine_min": round(float(cos.min()), 6),
            "max_abs_err": round(float(np.abs(a - b).max()), 6)}


def run(kin_dim=76, seq_len=77, embed_dim=768, checkpoint=None, threads=4, runs=200, out_dir=None) -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    model = KinematicEncoder(kin_dim=kin_dim, seq_len=seq_len, embed_dim=embed_dim).eval()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    params = sum(p.numel() for p in model.parameters())
    tmp = Path(out_dir or tempfile.mkdtemp())
    fp32, int8 = tmp / "encoder_fp32.onnx", tmp / "encoder_int8.onnx"
    export(model, fp32, kin_dim)
    import onnx
    onnx.checker.check_model(str(fp32))
    quantize(fp32, int8)
    s32, s8 = session(fp32, threads), session(int8, threads)
    report = {"date": str(date.today()), "weights": "trained checkpoint" if checkpoint else "seeded random init",
              "parameters": params, "hardware": f"{platform.machine()} CPU, {threads} threads",
              "size_mb": {"fp32": round(fp32.stat().st_size / 2**20, 1), "int8": round(int8.stat().st_size / 2**20, 1)},
              "batches": {}}
    rng = np.random.default_rng(0)
    for bs in (1, 16):
        kin = rng.standard_normal((bs, kin_dim)).astype(np.float32)
        lab = rng.integers(0, 16, size=bs).astype(np.int64)
        feeds = {"kinematics": kin, "gesture_label": lab}
        with torch.no_grad():
            ref = model(torch.from_numpy(kin), torch.from_numpy(lab)).numpy()
        o32, o8 = s32.run(None, feeds)[0], s8.run(None, feeds)[0]
        with torch.no_grad():
            t_torch = timeit(lambda: model(torch.from_numpy(kin), torch.from_numpy(lab)), runs)
        report["batches"][str(bs)] = {
            "torch_fp32": t_torch, "ort_fp32": timeit(lambda: s32.run(None, feeds), runs),
            "ort_int8": timeit(lambda: s8.run(None, feeds), runs),
            "ort_fp32_vs_torch": fidelity(ref, o32), "ort_int8_vs_torch": fidelity(ref, o8)}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    rep = run(checkpoint=a.checkpoint, threads=a.threads)
    out = ROOT / "docs" / f"encoder-optimization-{rep['date']}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
