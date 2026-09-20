"""FastAPI application.

Lifecycle: the model is loaded once at startup from
``<artifacts>/models/current.json``. A load failure leaves the service alive
(/health 200) but honestly not-ready (/ready 503 with a bounded reason).
"""

from __future__ import annotations

import io
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Response, UploadFile, status
from PIL import Image
from pydantic import BaseModel

from digit_classifier import __version__
from digit_classifier.config import AppConfig, find_project_root, load_config
from digit_classifier.inference.predictor import ModelLoadError, load_current_model

logger = logging.getLogger("digit_classifier.api")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    detail: str


class ModelInfoResponse(BaseModel):
    model_id: str | None
    model_kind: str | None
    loaded: bool
    detail: str


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    uncertain: bool
    model_id: str
    preprocessing_id: str
    inference_ms: float
    request_id: str


def create_app(config: AppConfig | None = None, models_dir: Path | None = None) -> FastAPI:
    cfg = config or load_config()
    resolved_models_dir = models_dir or find_project_root() / "artifacts" / "models"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the model exactly once during startup, never per prediction.
        try:
            app.state.predictor = load_current_model(resolved_models_dir)
            app.state.model_load_error = None
            logger.info("model loaded: %s", app.state.predictor.model_id)
        except ModelLoadError as exc:
            app.state.predictor = None
            # Bounded reason: message text only, file names but no directories.
            app.state.model_load_error = str(exc)
            logger.error("model load failed: %s", exc)
        yield

    app = FastAPI(
        title="Digit Classifier API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.predictor = None
    app.state.model_load_error = "Model loading has not been attempted yet."

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/ready", response_model=ReadyResponse)
    def ready(response: Response) -> ReadyResponse:
        if app.state.predictor is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyResponse(
                ready=False,
                model_loaded=False,
                detail=f"No model available: {app.state.model_load_error}",
            )
        return ReadyResponse(ready=True, model_loaded=True, detail="Service can serve predictions.")

    @app.get("/v1/model", response_model=ModelInfoResponse)
    def model_info(response: Response) -> ModelInfoResponse:
        predictor = app.state.predictor
        if predictor is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ModelInfoResponse(
                model_id=None,
                model_kind=None,
                loaded=False,
                detail=f"No model available: {app.state.model_load_error}",
            )
        return ModelInfoResponse(
            model_id=predictor.model_id,
            model_kind=predictor.model_kind,
            loaded=True,
            detail="Model loaded.",
        )

    @app.post("/v1/predictions", response_model=PredictionResponse)
    async def predict(image: UploadFile) -> PredictionResponse:
        predictor = app.state.predictor
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model is loaded; predictions are unavailable.",
            )

        payload = await image.read()
        with Image.open(io.BytesIO(payload)) as source:
            grayscale = source.convert("L").resize((28, 28))
            image_u8 = np.asarray(grayscale, dtype=np.uint8)

        output = predictor.predict(image_u8)
        return PredictionResponse(
            prediction=output.digit,
            confidence=round(output.confidence, 4),
            uncertain=output.confidence < cfg.evaluation.confidence_threshold,
            model_id=predictor.model_id,
            preprocessing_id="mnist-direct-001",
            inference_ms=output.inference_ms,
            request_id=str(uuid.uuid4()),
        )

    return app
