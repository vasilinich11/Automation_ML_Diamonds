"""FastAPI-приложение для предсказания стоимости бриллиантов.

Запуск: ``uvicorn diamonds_mlops.api.app:app`` или ``diamonds serve``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from sklearn.pipeline import Pipeline

from diamonds_mlops import __version__
from diamonds_mlops.api.schemas import (
    DiamondFeatures,
    ErrorResponse,
    HealthResponse,
    InfrastructureMetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
    RootResponse,
)
from diamonds_mlops.config import Settings, get_settings
from diamonds_mlops.features import FEATURE_COLUMNS, add_features
from diamonds_mlops.logging_config import configure_logging
from diamonds_mlops.model_training import load_model
from diamonds_mlops.monitoring import get_infrastructure_metrics

logger = logging.getLogger(__name__)

MODEL_NOT_TRAINED_DETAIL = "Model is not trained yet. Run `diamonds train` before calling /predict."

router = APIRouter()


def get_app_settings(request: Request) -> Settings:
    """Достать настройки, с которыми было создано приложение."""
    return cast("Settings", request.app.state.settings)


def get_model(request: Request) -> Pipeline:
    """Вернуть загруженную модель; при необходимости лениво подгрузить её с диска.

    Raises:
        HTTPException: 503, если артефакт модели ещё не создан.
    """
    state = request.app.state
    if state.model is None:
        state.model = load_model(get_app_settings(request).model_path)
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=MODEL_NOT_TRAINED_DETAIL
        )
    return cast("Pipeline", state.model)


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


@router.get("/", response_model=RootResponse)
def root() -> RootResponse:
    """Базовая информация об API."""
    return RootResponse(message="Diamonds price prediction API", docs="/docs", version=__version__)


@router.get("/health", response_model=HealthResponse)
def health(request: Request, settings: SettingsDep) -> HealthResponse:
    """Статус сервиса, наличие модели и загрузка ресурсов хоста."""
    model_loaded = request.app.state.model is not None or settings.model_path.exists()
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        infrastructure=InfrastructureMetricsResponse(**get_infrastructure_metrics()),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "The model artifact is not available yet.",
            "content": {"application/json": {"example": {"detail": MODEL_NOT_TRAINED_DETAIL}}},
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def predict(features: DiamondFeatures, request: Request) -> PredictionResponse:
    """Предсказать цену бриллианта по его характеристикам.

    Модель запрашивается внутри обработчика, чтобы невалидный запрос получал 422
    независимо от того, обучена ли модель.
    """
    model = get_model(request)
    frame = add_features(pd.DataFrame([features.model_dump()]))
    try:
        prediction = float(model.predict(frame[FEATURE_COLUMNS])[0])
    except Exception as exc:
        logger.exception("Prediction failed for payload %s", features.model_dump())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check input values and model artifact compatibility.",
        ) from exc
    return PredictionResponse(
        predicted_price=round(prediction, 2), message="Prediction completed successfully."
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(settings: SettingsDep) -> ModelInfoResponse:
    """Путь к модели, список признаков и сохранённые метрики."""
    metrics: dict[str, float] | None = None
    if settings.metrics_path.exists():
        metrics = json.loads(settings.metrics_path.read_text(encoding="utf-8"))
    model_exists = settings.model_path.exists()
    return ModelInfoResponse(
        model_path=str(settings.model_path),
        model_exists=model_exists,
        metrics=metrics,
        features=FEATURE_COLUMNS,
        message=(
            "Model artifact found."
            if model_exists
            else "Model artifact is missing. Run training before prediction."
        ),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Создать экземпляр FastAPI (application factory).

    Модель загружается один раз при старте приложения, а не при импорте модуля.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        app.state.model = load_model(settings.model_path)
        yield
        app.state.model = None

    app = FastAPI(
        title="Diamonds Price Prediction API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model = None
    app.include_router(router)
    return app


app = create_app()
