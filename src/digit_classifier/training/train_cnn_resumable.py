"""Resumable CNN training: one epoch per invocation.

A single training command completing all 10 epochs is fragile to interrupt.
This trainer persists model, optimizer, and RNG state between invocations:

    uv run python -m digit_classifier.training.train_cnn_resumable        # one epoch
    uv run python -m digit_classifier.training.train_cnn_resumable finalize

``finalize`` restores the best-validation-epoch weights, writes the standard
checkpoint + run metadata + metrics artifacts (same schemas as train.py),
and deletes the resume state.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from digit_classifier.config import find_project_root, load_config
from digit_classifier.models.cnn import MODEL_KIND, CompactCnn
from digit_classifier.training.datasets import load_mnist_tensors, make_loader, random_shift
from digit_classifier.training.train import evaluate_split, seed_everything, sha256_file

STATE_FILE = "cnn_resume_state.pt"


def _state_path(root: Path) -> Path:
    return root / "artifacts" / "run_metadata" / STATE_FILE


def _new_state(cfg, run_id: str) -> dict:
    seed_everything(cfg.training.seed)
    model = CompactCnn(dropout_conv=cfg.cnn.dropout_conv, dropout_fc=cfg.cnn.dropout_fc)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cnn.learning_rate)
    return {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "epochs_done": 0,
        "epoch_log": [],
        "best_val_accuracy": -1.0,
        "best_epoch": -1,
        "best_state_dict": None,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "wall_seconds": 0.0,
    }


def run_epoch() -> int:
    root = find_project_root()
    cfg = load_config()
    state_path = _state_path(root)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(state["torch_rng"])
    else:
        run_id = f"cnn-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        state = _new_state(cfg, run_id)

    if state["epochs_done"] >= cfg.cnn.epochs:
        print(f"all {cfg.cnn.epochs} epochs already done; run 'finalize'")
        return 0

    model = CompactCnn(dropout_conv=cfg.cnn.dropout_conv, dropout_fc=cfg.cnn.dropout_fc)
    model.load_state_dict(state["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cnn.learning_rate)
    optimizer.load_state_dict(state["optimizer_state"])
    loss_fn = nn.CrossEntropyLoss()

    data = load_mnist_tensors(root / cfg.data.interim_dir, root / cfg.data.processed_dir)
    epoch = state["epochs_done"] + 1
    # Distinct, deterministic shuffling per epoch:
    loader = make_loader(
        data["train"],
        batch_size=cfg.cnn.batch_size,
        shuffle=True,
        seed=cfg.training.seed + epoch,
    )

    # Stateless stepped LR decay: derived from the epoch number so it
    # survives resumption without scheduler state.
    lr = cfg.cnn.learning_rate * (cfg.cnn.lr_gamma ** ((epoch - 1) // cfg.cnn.lr_step_epochs))
    for group in optimizer.param_groups:
        group["lr"] = lr

    augment_gen = torch.Generator()
    augment_gen.manual_seed(cfg.training.seed * 1000 + epoch)

    t0 = time.monotonic()
    model.train()
    running, batches = 0.0, 0
    for xb, yb in loader:
        xb = random_shift(xb, cfg.cnn.augment_max_shift, augment_gen)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        running += float(loss.detach())
        batches += 1

    val = evaluate_split(model, data["val"])
    elapsed = time.monotonic() - t0
    entry = {
        "epoch": epoch,
        "mean_train_loss": running / max(batches, 1),
        "val_accuracy": val["accuracy"],
        "epoch_seconds": round(elapsed, 1),
    }
    state["epoch_log"].append(entry)
    state["epochs_done"] = epoch
    state["wall_seconds"] += elapsed
    if val["accuracy"] > state["best_val_accuracy"]:
        state["best_val_accuracy"] = val["accuracy"]
        state["best_epoch"] = epoch
        state["best_state_dict"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state["model_state"] = model.state_dict()
    state["optimizer_state"] = optimizer.state_dict()
    state["torch_rng"] = torch.get_rng_state()
    torch.save(state, state_path)

    print(
        f"epoch {epoch}/{cfg.cnn.epochs} (lr {lr:g}): loss {entry['mean_train_loss']:.4f} "
        f"val_acc {val['accuracy']:.4f} ({elapsed:.0f}s) "
        f"best so far: epoch {state['best_epoch']} @ {state['best_val_accuracy']:.4f}"
    )
    return 0


def finalize() -> int:
    root = find_project_root()
    cfg = load_config()
    state_path = _state_path(root)
    if not state_path.is_file():
        print("no resume state found")
        return 2
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state["best_state_dict"] is None:
        print("no completed epochs to finalize")
        return 2

    model = CompactCnn(dropout_conv=cfg.cnn.dropout_conv, dropout_fc=cfg.cnn.dropout_fc)
    model.load_state_dict(state["best_state_dict"])
    model.eval()

    data = load_mnist_tensors(root / cfg.data.interim_dir, root / cfg.data.processed_dir)
    val_metrics = evaluate_split(model, data["val"])
    test_metrics = evaluate_split(model, data["test"])

    run_id = state["run_id"]
    models_dir = root / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{run_id}.pt"
    extra_config = {"dropout_conv": cfg.cnn.dropout_conv, "dropout_fc": cfg.cnn.dropout_fc}
    torch.save(
        {
            "model_kind": MODEL_KIND,
            "run_id": run_id,
            "state_dict": model.state_dict(),
            "model_config": extra_config,
        },
        checkpoint_path,
    )

    processed = root / cfg.data.processed_dir
    interim = root / cfg.data.interim_dir
    metadata = {
        "run_id": run_id,
        "model_kind": MODEL_KIND,
        "seed": cfg.training.seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": "cpu",
        "epochs_requested": cfg.cnn.epochs,
        "epochs_completed": state["epochs_done"],
        "best_epoch": state["best_epoch"],
        "best_val_accuracy": state["best_val_accuracy"],
        "epoch_log": state["epoch_log"],
        "batch_size": cfg.cnn.batch_size,
        "learning_rate": cfg.cnn.learning_rate,
        "model_config": extra_config,
        "started_at": state["started_at"],
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_seconds": round(state["wall_seconds"], 2),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "params_sha256": sha256_file(root / "params.yaml"),
        "dataset_checksums": {
            "split_indices": sha256_file(processed / "mnist_split_indices.npz"),
            "mnist_manifest": sha256_file(interim / "mnist_manifest.json"),
        },
        "training_mode": "resumable-one-epoch-per-invocation",
        "reproducibility_note": (
            "Python/NumPy/PyTorch seeded; RNG state persisted between epoch "
            "invocations; exact bitwise reproducibility may vary across hardware."
        ),
    }
    (root / "artifacts" / "run_metadata" / f"{run_id}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    metrics_path = root / "artifacts" / "metrics" / f"{run_id}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_kind": MODEL_KIND,
                "best_epoch": state["best_epoch"],
                "validation": val_metrics,
                "test": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state_path.unlink()

    print(f"run_id: {run_id}")
    print(f"best epoch: {state['best_epoch']}  val acc: {val_metrics['accuracy']:.4f}")
    t_acc, t_f1 = test_metrics["accuracy"], test_metrics["macro_f1"]
    print(f"test accuracy: {t_acc:.4f}  macro F1: {t_f1:.4f}")
    low = [
        (c["digit"], round(c["recall"], 4)) for c in test_metrics["per_class"] if c["recall"] < 0.98
    ]
    print("classes with recall < 0.98:", low or "none")
    return 0


if __name__ == "__main__":
    sys.exit(finalize() if (len(sys.argv) > 1 and sys.argv[1] == "finalize") else run_epoch())
