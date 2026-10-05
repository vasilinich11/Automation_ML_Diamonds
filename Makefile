# Короткие команды для работы с проектом. Все команды выполняются в .venv через Poetry.
# На Windows без make можно запускать те же команды из правой части напрямую.

POETRY ?= poetry
RUN := $(POETRY) run

.DEFAULT_GOAL := help
.PHONY: help install lock lint format typecheck test pipeline train figures serve monitor \
        docker-build docker-up requirements clean

help:  ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Создать .venv, установить зависимости по poetry.lock и git-хуки
	$(POETRY) install
	$(RUN) pre-commit install

lock:  ## Обновить poetry.lock после изменения pyproject.toml
	$(POETRY) lock

lint:  ## Все проверки pre-commit (ruff, mypy, poetry check, гигиена файлов)
	$(RUN) pre-commit run --all-files

format:  ## Автоисправление и форматирование кода
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck:  ## Проверка типов mypy
	$(RUN) mypy

test:  ## Тесты с отчётом о покрытии
	$(RUN) pytest --cov

pipeline:  ## ETL + обучение модели
	$(RUN) diamonds pipeline

train:  ## Только обучение модели
	$(RUN) diamonds train

figures:  ## Графики для отчёта
	$(RUN) diamonds figures

serve:  ## Запустить API с автоперезагрузкой
	$(RUN) diamonds serve --reload

monitor:  ## Загрузка CPU/RAM/диска
	$(RUN) diamonds monitor

requirements:  ## Пересобрать requirements.txt из poetry.lock
	$(RUN) pre-commit run poetry-export --all-files

docker-build:  ## Собрать Docker-образ
	docker compose build

docker-up:  ## Запустить API и Prometheus в Docker
	docker compose up

clean:  ## Удалить кеши инструментов
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
