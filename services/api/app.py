"""FastAPI application shell."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from digit_classifier import __version__


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    detail: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="Digit Classifier API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
    )
    app.state.model_loaded = False

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/ready", response_model=ReadyResponse)
    def ready(response: Response) -> ReadyResponse:
        if not app.state.model_loaded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyResponse(
                ready=False,
                model_loaded=False,
                detail="No model is loaded; predictions are unavailable.",
            )
        return ReadyResponse(
            ready=True,
            model_loaded=True,
            detail="Service can serve predictions.",
        )

    return app
