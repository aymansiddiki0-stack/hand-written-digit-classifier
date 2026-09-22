"""Prediction endpoint tests against a real fixture-trained model.

Covers the response contract, direct MNIST-like preprocessing, upload
rejections, uncertainty, and corrupt serving artifacts.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from digit_classifier.config import load_config
from digit_classifier.models.cnn import MODEL_KIND, CompactCnn
from digit_classifier.training.train import sha256_file, train_model
from services.api.app import create_app
from tests.unit.test_training import tiny_dataset


@pytest.fixture(scope="module")
def fixture_models_dir(tmp_path_factory) -> Path:
    """Train a tiny CNN on synthetic data and select it as current."""
    artifacts = tmp_path_factory.mktemp("artifacts")
    x, y = tiny_dataset(n_per_class=6)
    model = CompactCnn(dropout_conv=0.0, dropout_fc=0.0)
    result = train_model(
        model,
        {"train": (x, y), "val": (x, y)},
        model_kind=MODEL_KIND,
        run_id="fixture-cnn",
        seed=3,
        epochs=12,
        batch_size=16,
        learning_rate=0.005,
        artifacts_dir=artifacts,
        extra_config={"dropout_conv": 0.0, "dropout_fc": 0.0},
    )
    models_dir = artifacts / "models"
    payload = {
        "model_id": "fixture-cnn",
        "model_kind": MODEL_KIND,
        "checkpoint_file": result.checkpoint_path.name,
        "checkpoint_sha256": sha256_file(result.checkpoint_path),
        "model_config": {"dropout_conv": 0.0, "dropout_fc": 0.0},
        "selected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_accuracy": result.best_val_accuracy,
    }
    (models_dir / "current.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return models_dir


def png_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def digit_pattern(digit: int = 3) -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.uint8)
    image[:, digit * 2 : digit * 2 + 3] = 220
    return image


def post_image(client: TestClient, data: bytes, content_type: str = "image/png"):
    return client.post("/v1/predictions", files={"image": ("digit.png", data, content_type)})


@pytest.fixture()
def client(fixture_models_dir: Path):
    with TestClient(create_app(models_dir=fixture_models_dir)) as c:
        yield c


def test_ready_flips_to_200_with_model(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {
        "ready": True,
        "model_loaded": True,
        "detail": "Service can serve predictions.",
    }
    info = client.get("/v1/model")
    assert info.status_code == 200
    assert info.json()["model_id"] == "fixture-cnn"
    assert info.json()["model_kind"] == MODEL_KIND


def test_prediction_contract_and_fixture_pattern(client: TestClient) -> None:
    resp = post_image(client, png_bytes(digit_pattern(3)))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "prediction",
        "confidence",
        "uncertain",
        "model_id",
        "preprocessing_id",
        "inference_ms",
        "request_id",
    }
    assert body["prediction"] == 3
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_id"] == "fixture-cnn"
    assert body["preprocessing_id"] == "mnist-direct-001"
    assert body["inference_ms"] >= 0.0


def test_request_ids_are_unique(client: TestClient) -> None:
    ids = {post_image(client, png_bytes(digit_pattern())).json()["request_id"] for _ in range(5)}
    assert len(ids) == 5


def test_non_28x28_input_is_resized(client: TestClient) -> None:
    large = np.repeat(np.repeat(digit_pattern(4), 2, axis=0), 2, axis=1)
    resp = post_image(client, png_bytes(large))
    assert resp.status_code == 200
    assert resp.json()["prediction"] == 4


def test_unsupported_media_type(client: TestClient) -> None:
    resp = post_image(client, b"GIF89a....", content_type="image/gif")
    assert resp.status_code == 415
    assert resp.json()["error_code"] == "unsupported_media_type"


def test_oversized_payload_rejected(fixture_models_dir: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(deep=True)
    cfg.service.max_upload_bytes = 1000
    with TestClient(create_app(config=cfg, models_dir=fixture_models_dir)) as client:
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 255, size=(400, 400), dtype=np.uint8).astype(np.uint8)
        data = png_bytes(noisy)
        assert len(data) > 1000
        resp = post_image(client, data)
    assert resp.status_code == 413
    assert resp.json()["error_code"] == "payload_too_large"


def test_corrupt_bytes_rejected(client: TestClient) -> None:
    resp = post_image(client, b"\x89PNG\r\n\x1a\ntotally-not-a-real-png")
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "undecodable_image"


def test_empty_body_rejected(client: TestClient) -> None:
    resp = post_image(client, b"")
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "empty_body"


def test_blank_image_rejected(client: TestClient) -> None:
    blank = np.zeros((28, 28), dtype=np.uint8)
    resp = post_image(client, png_bytes(blank))
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "blank_image"


def test_tiny_image_rejected(client: TestClient) -> None:
    tiny = np.zeros((20, 20), dtype=np.uint8)
    tiny[5:15, 5:15] = 255
    resp = post_image(client, png_bytes(tiny))
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "image_too_small"


def test_uncertain_flag_when_threshold_is_max(fixture_models_dir: Path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.evaluation.confidence_threshold = 1.0
    with TestClient(create_app(config=cfg, models_dir=fixture_models_dir)) as client:
        resp = post_image(client, png_bytes(digit_pattern(7)))
    assert resp.status_code == 200
    assert resp.json()["uncertain"] is True


def test_corrupted_checkpoint_leaves_service_alive_but_not_ready(
    fixture_models_dir: Path, tmp_path: Path
) -> None:
    import shutil

    broken = tmp_path / "models"
    shutil.copytree(fixture_models_dir, broken)
    meta = json.loads((broken / "current.json").read_text())
    checkpoint = broken / meta["checkpoint_file"]
    data = bytearray(checkpoint.read_bytes())
    data[len(data) // 2] ^= 0xFF
    checkpoint.write_bytes(bytes(data))
    with TestClient(create_app(models_dir=broken)) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert "checksum mismatch" in ready.json()["detail"]
        prediction = client.post("/v1/predictions")
        assert prediction.status_code == 503
