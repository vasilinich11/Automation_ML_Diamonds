"""Diamonds MLOps: ETL, обучение модели, FastAPI и мониторинг для предсказания цены бриллиантов."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("diamonds-mlops")
except PackageNotFoundError:  # пакет запущен из исходников без установки
    __version__ = "0.0.0+unknown"
