"""Local API entrypoint: `uv run python -m services.api.main`."""

from __future__ import annotations

import uvicorn

from digit_classifier.config import load_config


def run() -> None:
    cfg = load_config()
    uvicorn.run(
        "services.api.app:create_app",
        factory=True,
        host=cfg.service.host,
        port=cfg.service.port,
    )


if __name__ == "__main__":
    run()
