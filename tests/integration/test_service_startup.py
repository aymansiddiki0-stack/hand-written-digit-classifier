"""Service startup integration tests.

Boots the real uvicorn process (not TestClient). Two artifact states are
exercised via the DIGIT_MODELS_DIR override:
- an empty models dir -> alive but honestly not-ready;
- the project's real selected model -> ready, and a real HTTP prediction.
"""

from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(models_dir: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    env = dict(os.environ, DIGIT_MODELS_DIR=str(models_dir))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.api.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                return proc, base
        except httpx.HTTPError as exc:
            last_error = exc
            time.sleep(0.25)
    proc.terminate()
    raise RuntimeError(f"server did not become healthy: {last_error}")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture
def empty_models_server(tmp_path: Path):
    proc, base = start_server(tmp_path)
    yield base
    stop_server(proc)


@pytest.fixture
def real_model_server():
    if not (REAL_MODELS_DIR / "current.json").is_file():
        pytest.skip("no selected model artifact; run select_model first")
    if not (PROJECT_ROOT / "data" / "interim" / "mnist_test_images.npy").is_file():
        pytest.skip("MNIST interim data not ingested; run ingestion first")
    proc, base = start_server(REAL_MODELS_DIR)
    yield base
    stop_server(proc)


def test_alive_but_not_ready_without_model(empty_models_server: str) -> None:
    assert httpx.get(f"{empty_models_server}/health", timeout=5.0).status_code == 200
    ready = httpx.get(f"{empty_models_server}/ready", timeout=5.0)
    assert ready.status_code == 503
    assert ready.json()["model_loaded"] is False


def test_ready_and_real_prediction_with_selected_model(real_model_server: str) -> None:
    ready = httpx.get(f"{real_model_server}/ready", timeout=10.0)
    assert ready.status_code == 200
    assert ready.json()["model_loaded"] is True

    info = httpx.get(f"{real_model_server}/v1/model", timeout=10.0).json()
    assert info["loaded"] is True and info["model_kind"] == "compact-cnn"

    # A real MNIST test image over real HTTP.
    from PIL import Image

    images = np.load(PROJECT_ROOT / "data" / "interim" / "mnist_test_images.npy")
    labels = np.load(PROJECT_ROOT / "data" / "interim" / "mnist_test_labels.npy")
    buf = io.BytesIO()
    Image.fromarray(images[0], mode="L").save(buf, format="PNG")
    resp = httpx.post(
        f"{real_model_server}/v1/predictions",
        files={"image": ("digit.png", buf.getvalue(), "image/png")},
        timeout=30.0,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == int(labels[0])
    assert body["model_id"] == info["model_id"]
    assert body["inference_ms"] > 0


def test_repeated_requests_stay_stable(empty_models_server: str) -> None:
    for _ in range(25):
        assert httpx.get(f"{empty_models_server}/health", timeout=5.0).status_code == 200
