"""Централизованная конфигурация проекта.

Все пути к данным и артефактам, а также гиперпараметры собраны в одном месте
вместо констант, разбросанных по модулям. Любое значение можно переопределить
переменной окружения с префиксом ``DIAMONDS_`` или через файл ``.env``::

    DIAMONDS_PROJECT_ROOT=/app
    DIAMONDS_MODEL_FILE=/artifacts/model.joblib
    DIAMONDS_N_ESTIMATORS=300

Относительные пути считаются от ``project_root``, абсолютные используются как есть.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Настройки пайплайна, модели, мониторинга и API."""

    model_config = SettingsConfigDict(
        env_prefix="DIAMONDS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Пути
    project_root: Path = DEFAULT_PROJECT_ROOT
    raw_data_file: Path = Path("data/raw/diamonds.csv")
    processed_dir_name: Path = Path("data/processed")
    model_file: Path = Path("models/diamond_price_model.joblib")
    metrics_file: Path = Path("models/metrics.json")
    baseline_file: Path = Path("reports/monitoring/baseline.json")
    figures_dir_name: Path = Path("reports/figures")

    # Данные
    random_state: int = 42
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    sample_rows: int = Field(default=500, gt=0)

    # Модель
    n_estimators: int = Field(default=120, gt=0)
    min_samples_leaf: int = Field(default=2, gt=0)
    n_jobs: int = -1

    # Мониторинг
    drift_threshold: float = Field(default=0.25, gt=0.0)
    rmse_degradation_threshold: float = Field(default=0.15, gt=0.0)
    r2_drop_threshold: float = Field(default=0.10, gt=0.0)

    # Логирование
    log_level: str = "INFO"

    def resolve(self, path: Path) -> Path:
        """Вернуть абсолютный путь: относительные пути считаются от корня проекта."""
        return path if path.is_absolute() else self.project_root / path

    @property
    def raw_data_path(self) -> Path:
        """Путь к исходному CSV-файлу с данными."""
        return self.resolve(self.raw_data_file)

    @property
    def processed_dir(self) -> Path:
        """Каталог для train/test выборок после ETL."""
        return self.resolve(self.processed_dir_name)

    @property
    def model_path(self) -> Path:
        """Путь к сериализованному sklearn-пайплайну."""
        return self.resolve(self.model_file)

    @property
    def metrics_path(self) -> Path:
        """Путь к JSON с метриками последнего обучения."""
        return self.resolve(self.metrics_file)

    @property
    def baseline_path(self) -> Path:
        """Путь к baseline для мониторинга дрейфа и деградации."""
        return self.resolve(self.baseline_file)

    @property
    def figures_dir(self) -> Path:
        """Каталог для графиков отчёта."""
        return self.resolve(self.figures_dir_name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Вернуть закешированный экземпляр настроек (читается из окружения один раз)."""
    return Settings()
