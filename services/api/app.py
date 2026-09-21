"""FastAPI application.

Lifecycle: the model is loaded once at startup from
``<artifacts>/models/current.json``. A load failure leaves the service alive
(/health 200) but honestly not-ready (/ready 503 with a bounded reason).

Errors are typed and bounded: no stack traces, no local filesystem paths.
"""

from __future__ import annotations

import io
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, Response, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from digit_classifier import __version__
from digit_classifier.config import AppConfig, find_project_root, load_config
from digit_classifier.inference.predictor import ModelLoadError, load_current_model

logger = logging.getLogger("digit_classifier.api")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}


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


class ErrorResponse(BaseModel):
    error_code: str
    detail: str
    request_id: str


def _error(status_code: int, error_code: str, detail: str, request_id: str) -> JSONResponse:
    body = ErrorResponse(error_code=error_code, detail=detail, request_id=request_id)
    return JSONResponse(status_code=status_code, content=body.model_dump())


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

    @app.post(
        "/v1/predictions",
        response_model=PredictionResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def predict(image: UploadFile | None = None):  # noqa: B008
        request_id = str(uuid.uuid4())
        predictor = app.state.predictor
        if predictor is None:
            return _error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "model_unavailable",
                "No model is loaded; predictions are unavailable.",
                request_id,
            )
        if image is None:
            return _error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "missing_image",
                "Send the image as multipart form field 'image'.",
                request_id,
            )
        if (image.content_type or "") not in ALLOWED_CONTENT_TYPES:
            return _error(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                f"Supported content types: {sorted(ALLOWED_CONTENT_TYPES)}.",
                request_id,
            )
        max_bytes = cfg.service.max_upload_bytes
        payload = await image.read(max_bytes + 1)
        if len(payload) > max_bytes:
            return _error(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "payload_too_large",
                f"Image exceeds the {max_bytes} byte limit.",
                request_id,
            )
        if not payload:
            return _error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "empty_body",
                "The uploaded image is empty.",
                request_id,
            )
        try:
            with Image.open(io.BytesIO(payload)) as source:
                source.load()
                grayscale = source.convert("L")
                if min(grayscale.size) < 28:
                    return _error(
                        status.HTTP_422_UNPROCESSABLE_ENTITY,
                        "image_too_small",
                        "Image must be at least 28x28 pixels.",
                        request_id,
                    )
                image_u8 = np.asarray(grayscale.resize((28, 28)), dtype=np.uint8)
        except (UnidentifiedImageError, OSError):
            return _error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "undecodable_image",
                "The uploaded bytes are not a decodable PNG or JPEG image.",
                request_id,
            )
        if int(image_u8.max()) == int(image_u8.min()):
            return _error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "blank_image",
                "The uploaded image contains no visible digit.",
                request_id,
            )
        try:
            output = predictor.predict(image_u8)
        except (RuntimeError, ValueError):
            logger.exception("inference failed (request %s)", request_id)
            return _error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "inference_failed",
                "The model failed to produce a valid prediction.",
                request_id,
            )
        return PredictionResponse(
            prediction=output.digit,
            confidence=round(output.confidence, 4),
            uncertain=output.confidence < cfg.evaluation.confidence_threshold,
            model_id=predictor.model_id,
            preprocessing_id="mnist-direct-001",
            inference_ms=output.inference_ms,
            request_id=request_id,
        )

    return app
