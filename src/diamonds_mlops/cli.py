"""Единая точка входа командной строки.

Примеры::

    diamonds pipeline        # ETL + обучение
    diamonds prepare-data    # только ETL
    diamonds train           # только обучение
    diamonds figures         # графики для отчёта
    diamonds monitor         # загрузка CPU/RAM/диска
    diamonds serve --port 8000

Без установки пакета то же самое доступно через ``python -m diamonds_mlops``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import Any

from diamonds_mlops import __version__
from diamonds_mlops.config import Settings, get_settings
from diamonds_mlops.logging_config import configure_logging


def _print_json(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def _cmd_prepare_data(_: argparse.Namespace, settings: Settings) -> None:
    from diamonds_mlops.data_processing import run_data_pipeline

    split = run_data_pipeline(settings=settings)
    _print_json({"train_rows": len(split.train), "test_rows": len(split.test)})


def _cmd_train(_: argparse.Namespace, settings: Settings) -> None:
    from diamonds_mlops.model_training import load_processed_data, train_model

    split = load_processed_data(settings=settings)
    _print_json(train_model(split.train, split.test, settings=settings).metrics)


def _cmd_pipeline(_: argparse.Namespace, settings: Settings) -> None:
    from diamonds_mlops.pipeline import run_pipeline

    _print_json(run_pipeline(settings).metrics)


def _cmd_figures(_: argparse.Namespace, settings: Settings) -> None:
    from diamonds_mlops.visualization import generate_report_figures

    _print_json([str(path) for path in generate_report_figures(settings=settings)])


def _cmd_monitor(_: argparse.Namespace, __: Settings) -> None:
    from diamonds_mlops.monitoring import get_infrastructure_metrics

    _print_json(get_infrastructure_metrics())


def _cmd_serve(args: argparse.Namespace, settings: Settings) -> None:
    import uvicorn

    uvicorn.run(
        "diamonds_mlops.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


Command = Callable[[argparse.Namespace, Settings], None]

COMMANDS: dict[str, tuple[Command, str]] = {
    "prepare-data": (_cmd_prepare_data, "ETL: загрузка, очистка, признаки, train/test split"),
    "train": (_cmd_train, "Обучить модель на подготовленных данных"),
    "pipeline": (_cmd_pipeline, "Полный пайплайн: ETL + обучение"),
    "figures": (_cmd_figures, "Построить графики для отчёта (нужна группа viz)"),
    "monitor": (_cmd_monitor, "Показать загрузку CPU/RAM/диска"),
    "serve": (_cmd_serve, "Запустить FastAPI через uvicorn"),
}


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов со всеми подкомандами."""
    parser = argparse.ArgumentParser(prog="diamonds", description="Diamonds MLOps CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default=None, help="Переопределить DIAMONDS_LOG_LEVEL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, (_, help_text) in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text)
        if name == "serve":
            subparser.add_argument("--host", default="127.0.0.1")
            subparser.add_argument("--port", type=int, default=8000)
            subparser.add_argument("--reload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Разобрать аргументы и выполнить выбранную команду."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    if args.log_level:
        settings = settings.model_copy(update={"log_level": args.log_level})
    configure_logging(settings.log_level)

    handler, _ = COMMANDS[args.command]
    handler(args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
