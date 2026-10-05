# Multi-stage сборка:
#   builder - ставит Poetry, создаёт виртуальное окружение /app/.venv строго по poetry.lock
#             (только runtime-зависимости) и обучает модель на sample dataset;
#   runtime - чистый slim-образ без Poetry и компиляторов: копируется только .venv,
#             исходники и артефакты модели. Сервис работает от непривилегированного пользователя.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

ARG POETRY_VERSION=2.5.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

RUN python -m pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Сначала только файлы зависимостей: слой с .venv кешируется, пока не меняется poetry.lock.
COPY pyproject.toml poetry.lock poetry.toml README.md ./
RUN poetry install --only main --no-root --no-ansi && rm -rf "${POETRY_CACHE_DIR}"

# Затем сам пакет (устанавливается в .venv вместе с консольной командой `diamonds`).
COPY src ./src
RUN poetry install --only main --no-ansi

# Обучение модели во время сборки, чтобы /predict работал сразу после запуска контейнера.
ENV DIAMONDS_PROJECT_ROOT=/app
RUN /app/.venv/bin/diamonds pipeline


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    DIAMONDS_PROJECT_ROOT=/app \
    DIAMONDS_LOG_LEVEL=INFO

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/src ./src
COPY --from=builder --chown=app:app /app/models ./models
COPY --from=builder --chown=app:app /app/reports/monitoring ./reports/monitoring

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "diamonds_mlops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
