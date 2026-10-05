# Changelog

All notable changes to this MLOps project are documented here.

## 1.1.0 - 2026-10-06

### Changed

- Code moved to an installable src-layout package `src/diamonds_mlops` (previously a package named `src`).
- Centralized configuration in `config.py` (`pydantic-settings`, `DIAMONDS_*` environment variables, `.env.example`).
- `logging` instead of `print`; logging is configured only in entry points.
- FastAPI app built with an application factory; the model is loaded in `lifespan` instead of at import time; schemas moved to `api/schemas.py`.
- Single CLI `diamonds` (`pipeline`, `prepare-data`, `train`, `figures`, `monitor`, `serve`) replaces `run_pipeline.py` and per-module `__main__` blocks.
- Infrastructure metrics have a single implementation in `monitoring.py` (`infrastructure_monitoring.py` removed).
- Dependencies managed by Poetry (`pyproject.toml`, `poetry.lock`, `poetry.toml`); `requirements.txt` is exported from the lock file.
- Multi-stage Dockerfile: the virtual environment is built from `poetry.lock`, the model is trained at build time, the service runs as a non-root user.
- CI runs pre-commit, tests on Python 3.11-3.13 with coverage, Docker build and a container smoke test.
- `README_plan.md` re-encoded from UTF-16 to UTF-8; line endings normalized to LF via `.gitattributes`.

### Added

- Linters and type checking: ruff (lint + format) and mypy in strict mode.
- pre-commit hooks: file hygiene, ruff, mypy, `poetry check --lock`, `poetry export`, pytest on pre-push.
- In-project virtual environment (`.venv`), `.python-version`, VS Code settings, Makefile.
- `DataValidationError`, typed results (`NamedTuple`, `TypedDict`), Google-style docstrings.
- Tests for configuration and CLI; test coverage threshold of 85%.

### Removed

- `VERSION` file: the version is defined only in `pyproject.toml`.

## 1.0.0 - 2026-05-28

### Added

- ETL pipeline for the Diamonds regression dataset.
- Feature engineering: `volume`, `density`, `depth_to_width`.
- RandomForestRegressor training pipeline with RMSE, MAE, R2, and MAPE metrics.
- FastAPI application with health, prediction, and model info endpoints.
- Lightweight monitoring for drift, degradation, CPU, RAM, and disk usage.
- Pytest suite for data processing, model training, API behavior, and monitoring.
- Dockerfile and Docker Compose configuration.
- GitHub Actions workflow for dependency installation, compile check, tests, and Docker build.
- README, technical review report, and project structure documentation.
