"""Dataset construction and the normalization contract.

MNIST_MEAN/MNIST_STD are the normalization contract for this project.
Anything that touches pixel tensors (training, evaluation, ONNX parity,
camera preprocessing later) needs to import these, not re-derive them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from digit_classifier.data.splits import load_split

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def normalize_images(images_u8: np.ndarray) -> torch.Tensor:
    """uint8 [N,28,28] -> normalized float32 tensor [N,1,28,28]."""
    if images_u8.dtype != np.uint8 or images_u8.ndim != 3:
        raise ValueError(f"expected uint8 [N,28,28], got {images_u8.dtype} {images_u8.shape}")
    x = torch.from_numpy(images_u8.copy()).float().div_(255.0)
    x = (x - MNIST_MEAN) / MNIST_STD
    return x.unsqueeze(1)


def load_mnist_tensors(
    interim_dir: Path, processed_dir: Path
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load train/val/test tensors using the recorded deterministic split."""
    train_images = np.load(interim_dir / "mnist_train_images.npy")
    train_labels = np.load(interim_dir / "mnist_train_labels.npy")
    test_images = np.load(interim_dir / "mnist_test_images.npy")
    test_labels = np.load(interim_dir / "mnist_test_labels.npy")
    train_idx, val_idx = load_split(processed_dir)

    def pack(images: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return normalize_images(images), torch.from_numpy(labels.astype(np.int64))

    return {
        "train": pack(train_images[train_idx], train_labels[train_idx]),
        "val": pack(train_images[val_idx], train_labels[val_idx]),
        "test": pack(test_images, test_labels),
    }


def make_loader(
    tensors: tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed if seed is not None else 0)
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def random_shift(batch: torch.Tensor, max_shift: int, generator: torch.Generator) -> torch.Tensor:
    """Randomly translate each image by up to +/- max_shift pixels (training-only
    augmentation). Uses torch.roll: MNIST borders are background-constant after
    normalization, so wrapped edge pixels are background too."""
    if max_shift == 0:
        return batch
    n = batch.shape[0]
    shifts = torch.randint(-max_shift, max_shift + 1, (n, 2), generator=generator)
    out = batch.clone()
    for dy, dx in {(int(a), int(b)) for a, b in shifts}:
        mask = (shifts[:, 0] == dy) & (shifts[:, 1] == dx)
        if dy or dx:
            out[mask] = torch.roll(batch[mask], shifts=(dy, dx), dims=(2, 3))
    return out
