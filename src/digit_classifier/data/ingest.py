"""IDX file parsing for MNIST images and labels.

Structural validation only: IDX magic numbers, declared sample counts,
28x28 image dimensions, and label range 0-9.
"""

from __future__ import annotations

import struct

import numpy as np

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049
IMAGE_SIZE = 28
NUM_CLASSES = 10


class MnistIngestionError(Exception):
    """Raised when MNIST data is missing, corrupt, or structurally invalid."""


def parse_idx_images(payload: bytes) -> np.ndarray:
    if len(payload) < 16:
        raise MnistIngestionError("image file too short to contain an IDX header")
    magic, count, rows, cols = struct.unpack(">IIII", payload[:16])
    if magic != IMAGE_MAGIC:
        raise MnistIngestionError(f"bad image magic number: {magic} (expected {IMAGE_MAGIC})")
    if rows != IMAGE_SIZE or cols != IMAGE_SIZE:
        raise MnistIngestionError(f"unexpected image dimensions: {rows}x{cols}")
    expected_len = 16 + count * rows * cols
    if len(payload) != expected_len:
        raise MnistIngestionError(
            f"image payload length {len(payload)} != declared {expected_len} (truncated file?)"
        )
    data = np.frombuffer(payload, dtype=np.uint8, offset=16)
    return data.reshape(count, rows, cols)


def parse_idx_labels(payload: bytes) -> np.ndarray:
    if len(payload) < 8:
        raise MnistIngestionError("label file too short to contain an IDX header")
    magic, count = struct.unpack(">II", payload[:8])
    if magic != LABEL_MAGIC:
        raise MnistIngestionError(f"bad label magic number: {magic} (expected {LABEL_MAGIC})")
    if len(payload) != 8 + count:
        raise MnistIngestionError(
            f"label payload length {len(payload)} != declared {8 + count} (truncated file?)"
        )
    labels = np.frombuffer(payload, dtype=np.uint8, offset=8)
    if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
        raise MnistIngestionError(
            f"labels outside 0..{NUM_CLASSES - 1}: min={labels.min()} max={labels.max()}"
        )
    return labels
