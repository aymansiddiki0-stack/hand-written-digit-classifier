"""Deterministic train/validation splitting.

Carves the validation split from the 60k MNIST training portion with a
seeded permutation. The official 10k test set is never touched here.
"""

from __future__ import annotations

import numpy as np


class SplitError(Exception):
    """Raised when a split cannot be created or fails validation."""


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
