"""Единая настройка логирования для CLI и API.

Модули библиотеки только получают логгер через ``logging.getLogger(__name__)``,
а конфигурируют логирование исключительно точки входа (CLI, FastAPI).
"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | int = logging.INFO) -> None:
    """Настроить корневой логгер с единым форматом сообщений."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True)
