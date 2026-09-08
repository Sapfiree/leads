#!/usr/bin/env python3
"""
CLI для очистки заявок клиентов.

Примеры:
    python clean_leads.py input.xlsx
    python clean_leads.py input.xlsx --output-dir output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cleaner import CleanerError, run_pipeline


def _configure_stdio() -> None:
    """Корректный вывод кириллицы в консоли Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Очистка и нормализация заявок клиентов.\n\n"
            "Как указать свой файл:\n"
            "  1) Положи Excel/CSV в эту папку (или укажи полный путь).\n"
            "  2) Запусти: python clean_leads.py ИМЯ_ФАЙЛА --output-dir output\n"
            "  3) Результат смотри в папке output/.\n"
            "Не переименовывай .xlsx в .csv вручную — оставь исходное расширение."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python clean_leads.py leads.xlsx --output-dir output\n"
            "  python clean_leads.py sample_input.csv --output-dir output\n"
            "  python clean_leads.py \"D:\\data\\заявки.xlsx\" --output-dir output\n"
        ),
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Путь к входному файлу заявок (.xlsx, .xls, .csv, .txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Папка для результатов (по умолчанию: output)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    try:
        result = run_pipeline(input_path, output_dir)
    except CleanerError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr)
        return 130

    for warning in result.warnings:
        print(f"Предупреждение: {warning}")

    stats = result.stats.to_dict()
    print("Обработка завершена успешно.")
    print(f"Входной файл: {input_path}")
    print(f"Папка результата: {output_dir.resolve()}")
    print(f"  - clean_leads.xlsx")
    print(f"  - problem_leads.xlsx")
    print(f"  - processing_report.json")
    print("Статистика:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
