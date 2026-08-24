"""Service startup integration tests.

Boots the real uvicorn process (not TestClient) against an empty models
directory to prove it starts and reports its readiness honestly.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


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


def test_alive_but_not_ready_without_model(empty_models_server: str) -> None:
    assert httpx.get(f"{empty_models_server}/health", timeout=5.0).status_code == 200
    ready = httpx.get(f"{empty_models_server}/ready", timeout=5.0)
    assert ready.status_code == 503
    assert ready.json()["model_loaded"] is False
