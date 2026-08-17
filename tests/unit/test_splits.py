"""Deterministic split tests."""

from pathlib import Path

import numpy as np
import pytest

from digit_classifier.data.splits import (
    SplitError,
    load_split,
    make_split_indices,
    write_split,
)


def test_same_seed_same_split() -> None:
    a = make_split_indices(1000, 0.1, seed=42)
    b = make_split_indices(1000, 0.1, seed=42)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_different_seed_different_split() -> None:
    a = make_split_indices(1000, 0.1, seed=42)
    b = make_split_indices(1000, 0.1, seed=43)
    assert not np.array_equal(a[1], b[1])


def test_no_overlap_and_full_coverage() -> None:
    train, val = make_split_indices(997, 0.13, seed=7)
    assert np.intersect1d(train, val).size == 0
    combined = np.sort(np.concatenate([train, val]))
    assert np.array_equal(combined, np.arange(997))


def test_degenerate_inputs_rejected() -> None:
    with pytest.raises(SplitError):
        make_split_indices(1, 0.1, seed=1)
    with pytest.raises(SplitError):
        make_split_indices(100, 0.0, seed=1)
    with pytest.raises(SplitError):
        make_split_indices(100, 0.6, seed=1)


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    labels = (np.arange(2000) % 10).astype(np.uint8)
    result = write_split(tmp_path, 2000, 0.1, seed=99, labels=labels)
    assert result.n_train == 1800 and result.n_val == 200
    train, val = load_split(tmp_path)
    assert train.size == 1800 and val.size == 200


def test_class_loss_rejected(tmp_path: Path) -> None:
    # all labels identical except one sample -> some class missing from a side
    labels = np.zeros(100, dtype=np.uint8)
    with pytest.raises(SplitError, match="lost classes"):
        write_split(tmp_path, 100, 0.1, seed=1, labels=labels)


def test_tampered_indices_detected(tmp_path: Path) -> None:
    write_split(tmp_path, 500, 0.1, seed=5)
    idx_file = tmp_path / "mnist_split_indices.npz"
    data = bytearray(idx_file.read_bytes())
    data[-1] ^= 0xFF
    idx_file.write_bytes(bytes(data))
    with pytest.raises(SplitError, match="checksum mismatch"):
        load_split(tmp_path)


def test_missing_artifacts_rejected(tmp_path: Path) -> None:
    with pytest.raises(SplitError, match="missing"):
        load_split(tmp_path)
