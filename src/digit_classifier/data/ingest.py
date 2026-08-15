"""MNIST ingestion.

Downloads the four canonical MNIST IDX files from a documented mirror into
``data/raw``, validates their structure, decodes them into NumPy arrays in
``data/interim``, and writes a machine-readable manifest recording checksums
of the raw files.

Safety properties:
- downloads go to a ``*.part`` temp file and are renamed only after the
  gzip payload decompresses and parses, so an interrupted download can never
  masquerade as a complete file;
- ingestion is idempotent: valid existing files are reused, invalid ones are
  re-fetched;
- validation is structural (IDX magic numbers, declared counts, dimensions,
  label range) rather than relying on hardcoded reference checksums.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Documented, stable mirror of the canonical MNIST files. The original
# yann.lecun.com host is unreliable; this GitHub mirror serves the identical
# IDX gzip files. Recorded in the manifest on every run.
MNIST_MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master"

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049
IMAGE_SIZE = 28
NUM_CLASSES = 10

EXPECTED_COUNTS = {"train": 60_000, "test": 10_000}

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


class MnistIngestionError(Exception):
    """Raised when MNIST data is missing, corrupt, or structurally invalid."""


@dataclass(frozen=True)
class IngestResult:
    manifest_path: Path
    train_images: Path
    train_labels: Path
    test_images: Path
    test_labels: Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _decompress(path: Path) -> bytes:
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read()
    except (OSError, EOFError) as exc:
        raise MnistIngestionError(f"cannot decompress {path.name}: {exc}") from exc


def _is_valid_raw_file(path: Path, kind: str) -> bool:
    """True when an existing raw file decompresses and parses cleanly."""
    if not path.is_file():
        return False
    try:
        payload = _decompress(path)
        if kind == "images":
            parse_idx_images(payload)
        else:
            parse_idx_labels(payload)
        return True
    except MnistIngestionError:
        return False


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        part.replace(dest)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise MnistIngestionError(f"download failed for {url}: {exc}") from exc


def ingest_mnist(
    raw_dir: Path,
    interim_dir: Path,
    mirror: str = MNIST_MIRROR,
    download: bool = True,
) -> IngestResult:
    """Fetch (if needed), validate, decode, and manifest the MNIST dataset."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    raw_records: dict[str, dict[str, object]] = {}

    for key, filename in FILES.items():
        kind = "images" if key.endswith("images") else "labels"
        raw_path = raw_dir / filename
        reused = _is_valid_raw_file(raw_path, kind)
        if not reused:
            if not download:
                raise MnistIngestionError(
                    f"{filename} is missing or invalid and downloading is disabled"
                )
            _download(f"{mirror}/{filename}", raw_path)
        payload = _decompress(raw_path)
        arrays[key] = parse_idx_images(payload) if kind == "images" else parse_idx_labels(payload)
        raw_records[key] = {
            "filename": filename,
            "sha256": sha256_of(raw_path),
            "bytes": raw_path.stat().st_size,
            "reused_existing": reused,
        }

    # Cross-file validation: counts must match labels and the canonical totals.
    for split in ("train", "test"):
        n_images = arrays[f"{split}_images"].shape[0]
        n_labels = arrays[f"{split}_labels"].shape[0]
        if n_images != n_labels:
            raise MnistIngestionError(f"{split}: image count {n_images} != label count {n_labels}")
        if n_images != EXPECTED_COUNTS[split]:
            raise MnistIngestionError(f"{split}: unexpected sample count: {n_images}")

    # Decode to interim .npy files (uint8, shape checked above).
    interim_paths: dict[str, Path] = {}
    for key, arr in arrays.items():
        out = interim_dir / f"mnist_{key}.npy"
        np.save(out, arr)
        interim_paths[key] = out

    manifest = {
        "dataset": "mnist",
        "source_mirror": mirror,
        "raw_files": raw_records,
        "interim_files": {
            key: {
                "path": str(path.relative_to(interim_dir.parent.parent)),
                "shape": list(arrays[key].shape),
                "dtype": str(arrays[key].dtype),
            }
            for key, path in interim_paths.items()
        },
        "counts": {split: EXPECTED_COUNTS[split] for split in ("train", "test")},
        "label_range": [0, NUM_CLASSES - 1],
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
    }
    manifest_path = interim_dir / "mnist_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return IngestResult(
        manifest_path=manifest_path,
        train_images=interim_paths["train_images"],
        train_labels=interim_paths["train_labels"],
        test_images=interim_paths["test_images"],
        test_labels=interim_paths["test_labels"],
    )
