"""Typed configuration loaded from params.yaml.

Both roles read configuration through this module so the pipeline and the
service cannot silently disagree about paths, thresholds, or limits.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    camera_dir: Path
    validation_fraction: float = Field(gt=0.0, lt=0.5)
    split_seed: int


class TrainingConfig(BaseModel):
    seed: int
    device: str
    batch_size: int = Field(gt=0)

    @field_validator("device")
    @classmethod
    def known_device(cls, v: str) -> str:
        if v not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported device: {v!r}")
        return v


class EvaluationConfig(BaseModel):
    confidence_threshold: float = Field(ge=0.0, le=1.0)


class ServiceConfig(BaseModel):
    host: str
    port: int = Field(gt=0, lt=65536)
    max_upload_bytes: int = Field(gt=0)
    model_id: str | None = None


class PrivacyConfig(BaseModel):
    store_raw_camera_frames: bool = False


class TelemetryConfig(BaseModel):
    enabled: bool = True
    events_path: Path


class BaselineConfig(BaseModel):
    hidden_units: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)


class AppConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    service: ServiceConfig
    privacy: PrivacyConfig
    telemetry: TelemetryConfig
    baseline: BaselineConfig


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward until params.yaml is found."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "params.yaml").is_file():
            return candidate
    raise FileNotFoundError("params.yaml not found in any parent directory")


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate configuration.

    Raises pydantic.ValidationError for malformed values and
    FileNotFoundError when the file is absent, so callers fail fast
    instead of running with a half-configured system.
    """
    config_path = path or (find_project_root() / "params.yaml")
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration file is not a mapping: {config_path}")
    return AppConfig.model_validate(raw)
