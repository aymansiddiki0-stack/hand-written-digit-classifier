"""Training loop with reproducibility controls and run metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from digit_classifier.training.datasets import make_loader
from digit_classifier.training.metrics import evaluate_predictions


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def predict(model: nn.Module, tensors: tuple[torch.Tensor, torch.Tensor], batch_size: int = 256):
    model.eval()
    preds = []
    loader = make_loader(tensors, batch_size=batch_size, shuffle=False)
    for xb, _ in loader:
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds).numpy()


def evaluate_split(model: nn.Module, tensors: tuple[torch.Tensor, torch.Tensor]) -> dict:
    y_pred = predict(model, tensors)
    y_true = tensors[1].numpy()
    return evaluate_predictions(y_true, y_pred)


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    run_metadata_path: Path
    best_epoch: int
    best_val_accuracy: float
    epochs_run: int


def train_model(
    model: nn.Module,
    data: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    model_kind: str,
    run_id: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    artifacts_dir: Path,
    params_path: Path | None = None,
    dataset_checksums: dict[str, str] | None = None,
    extra_config: dict | None = None,
) -> TrainResult:
    """Train on data['train'], select the best epoch by validation accuracy,
    persist the best checkpoint with metadata and checksum."""
    seed_everything(seed)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    t_start = time.monotonic()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = make_loader(data["train"], batch_size=batch_size, shuffle=True, seed=seed)

    best_val_acc = -1.0
    best_epoch = -1
    best_state: dict | None = None
    epoch_log: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, batches = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            batches += 1
        val_metrics = evaluate_split(model, data["val"])
        epoch_log.append(
            {
                "epoch": epoch,
                "mean_train_loss": running_loss / max(batches, 1),
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)

    models_dir = artifacts_dir / "models"
    run_meta_dir = artifacts_dir / "run_metadata"
    models_dir.mkdir(parents=True, exist_ok=True)
    run_meta_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = models_dir / f"{run_id}.pt"
    torch.save(
        {
            "model_kind": model_kind,
            "run_id": run_id,
            "state_dict": model.state_dict(),
            "model_config": extra_config or {},
        },
        checkpoint_path,
    )

    metadata = {
        "run_id": run_id,
        "model_kind": model_kind,
        "seed": seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "numpy_version": np.__version__,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "epoch_log": epoch_log,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_seconds": round(time.monotonic() - t_start, 2),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "params_sha256": sha256_file(params_path) if params_path else None,
        "dataset_checksums": dataset_checksums or {},
        "reproducibility_note": (
            "Python/NumPy/PyTorch seeded; exact bitwise reproducibility may vary "
            "across hardware and PyTorch builds."
        ),
        "argv": sys.argv,
    }
    run_metadata_path = run_meta_dir / f"{run_id}.json"
    run_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainResult(
        checkpoint_path=checkpoint_path,
        run_metadata_path=run_metadata_path,
        best_epoch=best_epoch,
        best_val_accuracy=best_val_acc,
        epochs_run=epochs,
    )


def load_checkpoint(path: Path, model: nn.Module, expected_kind: str) -> nn.Module:
    """Load a checkpoint, rejecting missing files, wrong kinds, and mismatched shapes."""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"checkpoint is unreadable or malformed: {path.name}: {exc}") from exc
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"checkpoint missing state_dict: {path.name}")
    if payload.get("model_kind") != expected_kind:
        raise ValueError(
            f"checkpoint kind {payload.get('model_kind')!r} != expected {expected_kind!r}"
        )
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError as exc:
        raise ValueError(f"checkpoint does not match model architecture: {exc}") from exc
    model.eval()
    return model
