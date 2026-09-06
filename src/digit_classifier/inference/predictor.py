"""Model loading and prediction.

The "current model" is selected by ``select_model``, which writes
``artifacts/models/current.json``:

    {
      "model_id": "<run id>",
      "model_kind": "compact-cnn" | "baseline-mlp",
      "checkpoint_file": "<file in the same directory>",
      "checkpoint_sha256": "<sha256 of the checkpoint file>",
      "model_config": { ... constructor kwargs ... },
      "selected_at": "<iso timestamp>",
      "test_accuracy": <float from the metrics artifact>
    }

Loading verifies the checksum before deserializing, so a corrupted or
tampered checkpoint is rejected before any bytes reach ``torch.load``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from digit_classifier.models import baseline as baseline_mod
from digit_classifier.models import cnn as cnn_mod
from digit_classifier.training.datasets import normalize_images
from digit_classifier.training.train import load_checkpoint

CURRENT_MODEL_FILE = "current.json"


class ModelLoadError(Exception):
    """Raised when the current model cannot be loaded safely."""


def _build_architecture(kind: str, model_config: dict) -> nn.Module:
    if kind == cnn_mod.MODEL_KIND:
        return cnn_mod.CompactCnn(**model_config)
    if kind == baseline_mod.MODEL_KIND:
        return baseline_mod.BaselineMlp(**model_config)
    raise ModelLoadError(f"unknown model kind: {kind!r}")


@dataclass(frozen=True)
class PredictionOutput:
    digit: int
    confidence: float
    probabilities: list[float]
    inference_ms: float


class Predictor:
    def __init__(self, model: nn.Module, model_id: str, model_kind: str) -> None:
        self._model = model.eval()
        self.model_id = model_id
        self.model_kind = model_kind

    @torch.inference_mode()
    def predict(self, image_u8: np.ndarray) -> PredictionOutput:
        """Predict a single 28x28 uint8 grayscale image."""
        if image_u8.shape != (28, 28) or image_u8.dtype != np.uint8:
            raise ValueError(f"expected uint8 (28,28), got {image_u8.dtype} {image_u8.shape}")
        start = time.perf_counter()
        x = normalize_images(image_u8[np.newaxis, :, :])
        logits = self._model(x)
        probs = torch.softmax(logits, dim=1)[0]
        if not torch.isfinite(probs).all():
            raise RuntimeError("model produced non-finite probabilities")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        digit = int(probs.argmax())
        return PredictionOutput(
            digit=digit,
            confidence=float(probs[digit]),
            probabilities=[float(p) for p in probs],
            inference_ms=round(elapsed_ms, 2),
        )


def load_current_model(models_dir: Path) -> Predictor:
    current_path = models_dir / CURRENT_MODEL_FILE
    if not current_path.is_file():
        raise ModelLoadError("no current model is selected (current.json missing)")
    meta = json.loads(current_path.read_text(encoding="utf-8"))

    checkpoint_path = models_dir / meta["checkpoint_file"]
    if not checkpoint_path.is_file():
        raise ModelLoadError(f"checkpoint file missing: {meta['checkpoint_file']}")

    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != meta["checkpoint_sha256"]:
        raise ModelLoadError(
            f"checkpoint checksum mismatch for {meta['checkpoint_file']}; artifact corrupt?"
        )

    model = _build_architecture(meta["model_kind"], dict(meta["model_config"]))
    try:
        load_checkpoint(checkpoint_path, model, expected_kind=meta["model_kind"])
    except (ValueError, FileNotFoundError) as exc:
        raise ModelLoadError(str(exc)) from exc
    return Predictor(model, model_id=meta["model_id"], model_kind=meta["model_kind"])
