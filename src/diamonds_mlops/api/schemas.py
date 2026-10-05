"""Pydantic-схемы запросов и ответов API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InfrastructureMetricsResponse(BaseModel):
    """Загрузка ресурсов хоста в процентах."""

    cpu_percent: float
    ram_percent: float
    disk_percent: float


class RootResponse(BaseModel):
    """Базовая информация об API."""

    message: str
    docs: str
    version: str


class HealthResponse(BaseModel):
    """Статус сервиса, модели и инфраструктуры."""

    status: Literal["ok"]
    model_loaded: bool
    infrastructure: InfrastructureMetricsResponse

    model_config = ConfigDict(protected_namespaces=())


class DiamondFeatures(BaseModel):
    """Характеристики бриллианта для предсказания цены."""

    carat: float = Field(gt=0, examples=[1.0])
    cut: str = Field(min_length=1, examples=["Ideal"])
    color: str = Field(min_length=1, examples=["E"])
    clarity: str = Field(min_length=1, examples=["VS1"])
    depth: float = Field(gt=0, examples=[61.0])
    table: float = Field(gt=0, examples=[57.0])
    x: float = Field(gt=0, examples=[6.0])
    y: float = Field(gt=0, examples=[6.1])
    z: float = Field(gt=0, examples=[3.8])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "carat": 1.0,
                "cut": "Ideal",
                "color": "E",
                "clarity": "VS1",
                "depth": 61.0,
                "table": 57.0,
                "x": 6.0,
                "y": 6.1,
                "z": 3.8,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Результат предсказания."""

    predicted_price: float
    message: str


class ModelInfoResponse(BaseModel):
    """Информация об артефакте модели и его метриках."""

    model_path: str
    model_exists: bool
    metrics: dict[str, float] | None
    features: list[str]
    message: str

    model_config = ConfigDict(protected_namespaces=())


class ErrorResponse(BaseModel):
    """Тело ответа с ошибкой."""

    detail: str
