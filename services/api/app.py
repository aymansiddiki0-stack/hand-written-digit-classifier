"""FastAPI application shell."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from digit_classifier import __version__


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="Digit Classifier API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    return app
