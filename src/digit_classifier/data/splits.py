"""Deterministic train/validation splitting.

Carves the validation split from the 60k MNIST training portion with a
seeded permutation. The official 10k test set is never touched here. Split
indices and their checksum get persisted so every later stage trains on the
same recorded partition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class SplitError(Exception):
    """Raised when a split cannot be created or fails validation."""


@dataclass(frozen=True)
class SplitResult:
    indices_path: Path
    manifest_path: Path
    n_train: int
    n_val: int


def make_split_indices(
    n_samples: int, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 1:
        raise SplitError(f"cannot split {n_samples} samples")
    if not 0.0 < validation_fraction < 0.5:
        raise SplitError(f"validation_fraction out of range: {validation_fraction}")
    n_val = int(round(n_samples * validation_fraction))
    if n_val == 0 or n_val >= n_samples:
        raise SplitError(f"degenerate split: n_val={n_val} of {n_samples}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def write_split(
    processed_dir: Path,
    n_samples: int,
    validation_fraction: float,
    seed: int,
    labels: np.ndarray | None = None,
) -> SplitResult:
    """Create, validate, and persist the train/val split."""
    train_idx, val_idx = make_split_indices(n_samples, validation_fraction, seed)

    # Invariants checked on every write, not assumed.
    if np.intersect1d(train_idx, val_idx).size:
        raise SplitError("train and validation indices overlap")
    if train_idx.size + val_idx.size != n_samples:
        raise SplitError("split does not cover all samples exactly once")

    label_coverage: dict[str, list[int]] | None = None
    if labels is not None:
        if labels.shape[0] != n_samples:
            raise SplitError("labels length does not match n_samples")
        val_classes = np.unique(labels[val_idx])
        train_classes = np.unique(labels[train_idx])
        if val_classes.size < 10 or train_classes.size < 10:
            raise SplitError(
                f"split lost classes: train has {train_classes.size}, val has {val_classes.size}"
            )
        label_coverage = {
            "train_class_counts": np.bincount(labels[train_idx], minlength=10).tolist(),
            "val_class_counts": np.bincount(labels[val_idx], minlength=10).tolist(),
        }

    processed_dir.mkdir(parents=True, exist_ok=True)
    indices_path = processed_dir / "mnist_split_indices.npz"
    np.savez(indices_path, train=train_idx, val=val_idx)

    digest = hashlib.sha256(indices_path.read_bytes()).hexdigest()
    manifest = {
        "split": "mnist-train-val",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "n_samples": n_samples,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "indices_file": indices_path.name,
        "indices_sha256": digest,
        "label_coverage": label_coverage,
        "note": "official 10k test set excluded; never used for tuning",
    }
    manifest_path = processed_dir / "mnist_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return SplitResult(
        indices_path=indices_path,
        manifest_path=manifest_path,
        n_train=int(train_idx.size),
        n_val=int(val_idx.size),
    )


def load_split(processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    indices_path = processed_dir / "mnist_split_indices.npz"
    manifest_path = processed_dir / "mnist_split_manifest.json"
    if not indices_path.is_file() or not manifest_path.is_file():
        raise SplitError("split artifacts missing; run split generation first")
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(indices_path.read_bytes()).hexdigest()
    if digest != manifest["indices_sha256"]:
        raise SplitError("split indices checksum mismatch; regenerate the split")
    data = np.load(indices_path)
    return data["train"], data["val"]
