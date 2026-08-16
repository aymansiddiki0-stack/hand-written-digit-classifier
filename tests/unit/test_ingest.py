"""MNIST ingestion tests. No network access: synthetic IDX fixtures."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from digit_classifier.data.ingest import (
    EXPECTED_COUNTS,
    FILES,
    MnistIngestionError,
    ingest_mnist,
    parse_idx_images,
    parse_idx_labels,
)


def make_images_payload(count: int, rows: int = 28, cols: int = 28) -> bytes:
    header = struct.pack(">IIII", 2051, count, rows, cols)
    body = (np.arange(count * rows * cols, dtype=np.uint64) % 256).astype(np.uint8).tobytes()
    return header + body


def make_labels_payload(count: int, bad_value: int | None = None) -> bytes:
    header = struct.pack(">II", 2049, count)
    labels = (np.arange(count, dtype=np.uint64) % 10).astype(np.uint8)
    if bad_value is not None and count:
        labels[0] = bad_value
    return header + labels.tobytes()


def write_fixture_dataset(raw_dir: Path) -> None:
    """A structurally valid 'MNIST' with canonical counts but tiny synthetic pixels."""
    payloads = {
        "train_images": make_images_payload(EXPECTED_COUNTS["train"]),
        "train_labels": make_labels_payload(EXPECTED_COUNTS["train"]),
        "test_images": make_images_payload(EXPECTED_COUNTS["test"]),
        "test_labels": make_labels_payload(EXPECTED_COUNTS["test"]),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in FILES.items():
        (raw_dir / filename).write_bytes(gzip.compress(payloads[key]))


# --- parser unit tests -------------------------------------------------------


def test_parse_valid_images() -> None:
    arr = parse_idx_images(make_images_payload(5))
    assert arr.shape == (5, 28, 28)
    assert arr.dtype == np.uint8


def test_parse_rejects_bad_magic() -> None:
    payload = struct.pack(">IIII", 1234, 1, 28, 28) + bytes(784)
    with pytest.raises(MnistIngestionError, match="magic"):
        parse_idx_images(payload)


def test_parse_rejects_wrong_dimensions() -> None:
    payload = struct.pack(">IIII", 2051, 1, 27, 28) + bytes(27 * 28)
    with pytest.raises(MnistIngestionError, match="dimensions"):
        parse_idx_images(payload)


def test_parse_rejects_truncated_images() -> None:
    payload = make_images_payload(3)[:-10]
    with pytest.raises(MnistIngestionError, match="truncated"):
        parse_idx_images(payload)


def test_parse_rejects_out_of_range_labels() -> None:
    with pytest.raises(MnistIngestionError, match="labels outside"):
        parse_idx_labels(make_labels_payload(4, bad_value=11))


def test_parse_rejects_truncated_labels() -> None:
    with pytest.raises(MnistIngestionError, match="truncated"):
        parse_idx_labels(make_labels_payload(4)[:-1])


# --- ingestion behaviour tests ----------------------------------------------


def test_ingest_valid_fixture_writes_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    write_fixture_dataset(raw)
    result = ingest_mnist(raw, interim, download=False)
    assert result.manifest_path.is_file()
    import json

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["counts"] == {"train": 60000, "test": 10000}
    assert set(manifest["raw_files"]) == set(FILES)
    for record in manifest["raw_files"].values():
        assert len(record["sha256"]) == 64
    train = np.load(result.train_images)
    assert train.shape == (60000, 28, 28)


def test_ingest_is_idempotent_and_reuses_valid_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    write_fixture_dataset(raw)
    first = ingest_mnist(raw, interim, download=False)
    second = ingest_mnist(raw, interim, download=False)
    import json

    m2 = json.loads(second.manifest_path.read_text())
    assert all(rec["reused_existing"] for rec in m2["raw_files"].values())
    assert first.train_images == second.train_images


def test_ingest_rejects_corrupt_gzip_without_download(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    write_fixture_dataset(raw)
    # Corrupt one file mid-stream (simulates interrupted/partial download kept on disk)
    target = raw / FILES["train_images"]
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    del data[-2000:]
    target.write_bytes(bytes(data))
    with pytest.raises(MnistIngestionError, match="missing or invalid"):
        ingest_mnist(raw, interim, download=False)


def test_ingest_rejects_count_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    write_fixture_dataset(raw)
    # Replace train labels with a valid file of the wrong count
    (raw / FILES["train_labels"]).write_bytes(gzip.compress(make_labels_payload(59999)))
    with pytest.raises(MnistIngestionError, match="!= label count"):
        ingest_mnist(raw, interim, download=False)


def test_ingest_rejects_wrong_canonical_count(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    write_fixture_dataset(raw)
    # Images and labels agree with each other but not with EXPECTED_COUNTS
    (raw / FILES["train_images"]).write_bytes(gzip.compress(make_images_payload(59999)))
    (raw / FILES["train_labels"]).write_bytes(gzip.compress(make_labels_payload(59999)))
    with pytest.raises(MnistIngestionError, match="expected 60000 samples, found 59999"):
        ingest_mnist(raw, interim, download=False)


def test_ingest_missing_file_with_download_disabled(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    write_fixture_dataset(raw)
    (raw / FILES["test_labels"]).unlink()
    with pytest.raises(MnistIngestionError, match="missing or invalid"):
        ingest_mnist(raw, interim, download=False)
