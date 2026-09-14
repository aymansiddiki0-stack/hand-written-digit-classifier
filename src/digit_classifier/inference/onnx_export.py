"""ONNX export and PyTorch parity validation.

Parity contract: on held-out samples, top-1 predicted-class agreement between
PyTorch and ONNX Runtime must be 100%, and the maximum absolute probability
difference must stay within PROB_TOLERANCE. Both the tolerance and the
observed maximum are recorded in the parity report artifact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

PROB_TOLERANCE = 1e-5
OPSET = 18


class OnnxExportError(Exception):
    pass


def export_to_onnx(model: nn.Module, output_path: Path) -> Path:
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=OPSET,
            dynamo=False,
        )
    except Exception as exc:
        raise OnnxExportError(f"ONNX export failed: {exc}") from exc
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise OnnxExportError("ONNX export produced no file")
    return output_path


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@torch.inference_mode()
def validate_parity(
    model: nn.Module,
    onnx_path: Path,
    inputs: torch.Tensor,
    report_path: Path | None = None,
    tolerance: float = PROB_TOLERANCE,
) -> dict:
    """Compare PyTorch and ONNX Runtime on the given normalized inputs."""
    if inputs.ndim != 4 or inputs.shape[1:] != (1, 28, 28):
        raise ValueError(f"expected [N,1,28,28] inputs, got {tuple(inputs.shape)}")
    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    torch_probs, onnx_probs = [], []
    onnx_ms = 0.0
    batch = 512
    for start in range(0, inputs.shape[0], batch):
        chunk = inputs[start : start + batch]
        torch_probs.append(torch.softmax(model(chunk), dim=1).numpy())
        t0 = time.perf_counter()
        (logits,) = session.run(["logits"], {"image": chunk.numpy()})
        onnx_ms += (time.perf_counter() - t0) * 1000.0
        onnx_probs.append(_softmax(logits))

    tp = np.concatenate(torch_probs)
    op = np.concatenate(onnx_probs)
    torch_top1 = tp.argmax(axis=1)
    onnx_top1 = op.argmax(axis=1)
    agreement = float((torch_top1 == onnx_top1).mean())
    max_prob_diff = float(np.abs(tp - op).max())

    report = {
        "n_samples": int(inputs.shape[0]),
        "top1_agreement": agreement,
        "max_prob_diff": max_prob_diff,
        "tolerance": tolerance,
        "within_tolerance": max_prob_diff <= tolerance,
        "full_agreement": agreement == 1.0,
        "onnx_total_inference_ms": round(onnx_ms, 2),
        "onnxruntime_version": ort.__version__,
        "torch_version": torch.__version__,
        "opset": OPSET,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
