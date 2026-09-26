"""ONNX export parity and INT8 fidelity on a small encoder (skipped when onnxruntime is absent)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from optimize_encoder import run  # noqa: E402


def test_export_matches_torch_and_int8_stays_close(tmp_path):
    rep = run(kin_dim=76, seq_len=4, embed_dim=32, runs=5, threads=1, out_dir=tmp_path)
    b = rep["batches"]["16"]
    assert b["ort_fp32_vs_torch"]["max_abs_err"] < 1e-4          # graph export is exact up to float noise
    assert b["ort_int8_vs_torch"]["cosine_min"] > 0.99            # quantized weights preserve direction
    assert rep["size_mb"]["int8"] < rep["size_mb"]["fp32"]
