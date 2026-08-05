"""Environment and configuration smoke tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import digit_classifier
from digit_classifier.config import find_project_root, load_config


def test_package_imports() -> None:
    assert digit_classifier.__version__


def test_project_root_contains_params() -> None:
    root = find_project_root()
    assert (root / "params.yaml").is_file()


def test_load_real_config() -> None:
    cfg = load_config()
    assert cfg.training.device == "cpu"
    assert 0.0 <= cfg.evaluation.confidence_threshold <= 1.0
    assert cfg.privacy.store_raw_camera_frames is False
    assert cfg.service.max_upload_bytes > 0


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/params.yaml"))


def test_malformed_values_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "params.yaml"
    bad.write_text(
        """
data:
  raw_dir: data/raw
  interim_dir: data/interim
  processed_dir: data/processed
  camera_dir: data/camera
  validation_fraction: 0.9
  split_seed: 1
training: {seed: 1, device: gpu9000, batch_size: -4}
evaluation: {confidence_threshold: 2.0}
service: {host: 127.0.0.1, port: 0, max_upload_bytes: 0}
privacy: {store_raw_camera_frames: false}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as exc:
        load_config(bad)
    fields = {e["loc"] for e in exc.value.errors()}
    # every malformed field must be reported, not just the first one
    assert ("data", "validation_fraction") in fields
    assert ("training", "device") in fields
    assert ("training", "batch_size") in fields
    assert ("evaluation", "confidence_threshold") in fields
    assert ("service", "port") in fields
    assert ("service", "max_upload_bytes") in fields


def test_non_mapping_config_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "params.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)
