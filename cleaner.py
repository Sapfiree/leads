"""
Модуль очистки и нормализации заявок клиентов.

Правила:
- Телефон: строго +7 и ровно 10 цифр; иначе — невалидный.
- Дата: YYYY-MM-DD; нераспознанная дата помечается как «битая дата».
- Имя: trim, сжатие пробелов, Title Case (включая дефисы).
- Дедупликация: по нормализованному телефону, оставляем первую строку.
- Строка с валидным телефоном и именем, но битой датой — в clean и в problem.
- Без валидного телефона — только в problem (дедупликация невозможна).
- Дубликаты — только в problem.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from dateutil import parser as date_parser

REQUIRED_COLUMNS = ("Имя", "Телефон", "Дата заявки", "Источник")

PROBLEM_NO_PHONE = "некорректный телефон"
PROBLEM_NO_NAME = "отсутствует имя"
PROBLEM_BAD_DATE = "битая дата"
PROBLEM_DUPLICATE = "дубликат телефона"

STATUS_CLEAN = "чистая"
STATUS_CLEAN_WITH_ISSUES = "в чистых с замечаниями"
STATUS_PROBLEM = "проблемная"
STATUS_DUPLICATE = "дубликат"

# Российские месяцы для текстовых дат
_RU_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}

_PHONE_DIGITS_RE = re.compile(r"\D+")
_VALID_PHONE_RE = re.compile(r"^\+7\d{10}$")
_EMPTY_NAME_MARKERS = frozenset(
    {
        "",
        "-",
        "—",
        "–",
        ".",
        "нет",
        "н/д",
        "n/a",
        "na",
        "none",
        "null",
        "без имени",
        "unknown",
    }
)


class CleanerError(Exception):
    """Понятная ошибка обработки для CLI."""


@dataclass
class ProcessingStats:
    total_rows: int = 0
    clean_unique: int = 0
    problem_rows: int = 0
    invalid_phones: int = 0
    bad_dates: int = 0
    missing_names: int = 0
    duplicates: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "всего_строк_на_входе": self.total_rows,
            "чистых_уникальных_заявок": self.clean_unique,
            "строк_с_проблемами": self.problem_rows,
            "невалидных_телефонов": self.invalid_phones,
            "битых_дат": self.bad_dates,
            "отсутствующих_имён": self.missing_names,
            "найденных_дублей": self.duplicates,
        }


@dataclass
class CleaningResult:
    clean_df: pd.DataFrame
    problem_df: pd.DataFrame
    stats: ProcessingStats
    warnings: list[str] = field(default_factory=list)


def normalize_phone(raw: Any) -> str | None:
    """
    Нормализует телефон к формату +7XXXXXXXXXX.
    Возвращает None, если номер нельзя однозначно привести к российскому.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    # Excel иногда читает телефон как float (79991234567.0)
    if isinstance(raw, float):
        if raw.is_integer():
            raw = str(int(raw))
        else:
            return None
    elif isinstance(raw, int):
        raw = str(raw)
    else:
        raw = str(raw).strip()

    if not raw or raw.lower() in {"nan", "none", "null", "-"}:
        return None

    digits = _PHONE_DIGITS_RE.sub("", raw)
    if not digits:
        return None

    # 11 цифр: 8XXXXXXXXXX или 7XXXXXXXXXX
    if len(digits) == 11 and digits[0] in {"7", "8"}:
        local = digits[1:]
        return f"+7{local}"

    # 10 цифр без кода страны
    if len(digits) == 10:
        return f"+7{digits}"

    # Уже с +7 и лишними символами, но после очистки 11 цифр с 7
    if len(digits) == 12 and digits.startswith("77"):
        # неоднозначно — не угадываем
        return None

    return None


def is_valid_phone(phone: str | None) -> bool:
    return bool(phone and _VALID_PHONE_RE.fullmatch(phone))


def normalize_name(raw: Any) -> str | None:
    """Нормализует имя: trim, сжатие пробелов, Title Case с учётом дефисов."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    text = str(raw).strip()
    if not text:
        return None

    text = re.sub(r"\s+", " ", text)
    if text.lower() in _EMPTY_NAME_MARKERS:
        return None

    parts = text.split(" ")
    normalized_parts: list[str] = []
    for part in parts:
        if "-" in part:
            hyphen_bits = [
                bit[:1].upper() + bit[1:].lower() if bit else ""
                for bit in part.split("-")
            ]
            normalized_parts.append("-".join(hyphen_bits))
        else:
            normalized_parts.append(part[:1].upper() + part[1:].lower() if part else "")

    result = " ".join(normalized_parts).strip()
    return result or None


def _parse_russian_text_date(text: str) -> pd.Timestamp | None:
    """Парсит даты вида «10 февраля 2024» / «10 февраля 2024 г.»."""
    cleaned = text.lower().replace("г.", "").replace("года", "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    match = re.match(
        r"^(\d{1,2})\s+([а-яё]+)\s+(\d{4})$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    day_s, month_s, year_s = match.groups()
    month = _RU_MONTHS.get(month_s.lower())
    if month is None:
        return None
    try:
        return pd.Timestamp(year=int(year_s), month=month, day=int(day_s))
    except (ValueError, TypeError):
        return None


def normalize_date(raw: Any) -> str | None:
    """
    Нормализует дату к YYYY-MM-DD.
    Возвращает None, если дату распознать нельзя (без угадываний).
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    if isinstance(raw, pd.Timestamp):
        if pd.isna(raw):
            return None
        return raw.strftime("%Y-%m-%d")

    if hasattr(raw, "strftime") and not isinstance(raw, str):
        try:
            return pd.Timestamp(raw).strftime("%Y-%m-%d")
        except (ValueError, TypeError, OverflowError):
            return None

    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", "-", "n/a"}:
        return None

    # Число Excel serial date
    if re.fullmatch(r"\d+(\.\d+)?", text):
        try:
            as_float = float(text)
            # Excel serial примерно 1..60000 для разумных дат
            if 1 <= as_float <= 100000:
                excel_date = pd.to_datetime(as_float, unit="D", origin="1899-12-30")
                return excel_date.strftime("%Y-%m-%d")
        except (ValueError, TypeError, OverflowError):
            pass

    ru_date = _parse_russian_text_date(text)
    if ru_date is not None:
        return ru_date.strftime("%Y-%m-%d")

    # Явные форматы день-месяц-год и ISO
    known_formats = (
        "%d.%m.%Y",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%d-%m-%Y",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
    )
    for fmt in known_formats:
        try:
            return pd.to_datetime(text, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

    # dateutil как запасной вариант, dayfirst=True для европейских дат
    try:
        parsed = date_parser.parse(text, dayfirst=True, yearfirst=False, fuzzy=False)
        return pd.Timestamp(parsed).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError):
        return None


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def detect_delimiter(sample: str) -> str:
    """Определяет разделитель CSV/TXT по первой содержательной строке."""
    candidates = [";", ",", "\t", "|"]
    counts = {sep: sample.count(sep) for sep in candidates}
    best_sep, best_count = max(counts.items(), key=lambda item: item[1])
    if best_count == 0:
        return ","
    # Если несколько разделителей с одинаковым максимумом — предпочитаем ;
    tied = [sep for sep, count in counts.items() if count == best_count]
    if len(tied) > 1:
        for preferred in (";", ",", "\t", "|"):
            if preferred in tied:
                return preferred
    return best_sep


def _detect_excel_format(raw_bytes: bytes) -> str | None:
    """Определяет Excel по сигнатуре, даже если расширение неверное."""
    if raw_bytes.startswith(b"PK"):
        return "xlsx"
    # OLE Compound Document — старый .xls
    if raw_bytes.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
        return "xls"
    return None


def _read_excel(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, dtype=object)
    except Exception as exc:  # noqa: BLE001
        raise CleanerError(f"Ошибка чтения Excel «{path}»: {exc}") from exc


def _read_text_table(path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    encodings = ("utf-8-sig", "utf-8", "cp1251")
    raw_bytes = path.read_bytes()

    excel_kind = _detect_excel_format(raw_bytes)
    if excel_kind is not None:
        warnings.append(
            f"Файл «{path.name}» по содержимому является Excel (.{excel_kind}), "
            f"хотя расширение «{path.suffix}». Читаю как Excel. "
            "Лучше переименовать в .xlsx/.xls или сохранить через «Файл → Сохранить как → CSV»."
        )
        return _read_excel(path), warnings

    text: str | None = None
    used_encoding: str | None = None

    for encoding in encodings:
        try:
            text = raw_bytes.decode(encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if text is None or used_encoding is None:
        raise CleanerError(
            f"Не удалось прочитать файл «{path}»: это не текстовый CSV/TXT "
            "в кодировке UTF-8, UTF-8-SIG или cp1251. "
            "Если это Excel — укажите файл с расширением .xlsx/.xls "
            "(простое переименование в .csv не конвертирует формат)."
        )

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise CleanerError(f"Файл «{path}» пуст: нет данных для обработки.")

    delimiter = detect_delimiter(lines[0])
    if delimiter == "," and ";" not in lines[0] and lines[0].count(",") == 0:
        warnings.append(
            "Разделитель в текстовом файле не обнаружен явно; используется запятая (,)."
        )
    elif delimiter != ";":
        warnings.append(
            f"Для текстового файла выбран разделитель «{delimiter}» "
            f"(кодировка: {used_encoding})."
        )

    try:
        df = pd.read_csv(
            path,
            sep=delimiter,
            encoding=used_encoding,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:  # noqa: BLE001 — отдаём понятное сообщение
        raise CleanerError(f"Ошибка чтения CSV/TXT «{path}»: {exc}") from exc

    return df, warnings


def load_input(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Загружает входной файл (.xlsx/.xls/.csv/.txt) в DataFrame."""
    if not path.exists():
        raise CleanerError(f"Файл не найден: «{path}».")

    if not path.is_file():
        raise CleanerError(f"Указанный путь не является файлом: «{path}».")

    suffix = path.suffix.lower()
    warnings: list[str] = []

    if suffix in {".xlsx", ".xls"}:
        df = _read_excel(path)
    elif suffix in {".csv", ".txt"}:
        df, warnings = _read_text_table(path)
    else:
        # На случай расширения вроде .CSV.XLSX или без расширения — пробуем по сигнатуре
        raw_bytes = path.read_bytes()[:8]
        excel_kind = _detect_excel_format(raw_bytes)
        if excel_kind is not None:
            warnings.append(
                f"Расширение «{suffix or '(нет)'}» нестандартное, "
                f"но файл похож на Excel (.{excel_kind}). Читаю как Excel."
            )
            df = _read_excel(path)
        else:
            raise CleanerError(
                f"Неподдерживаемое расширение «{suffix}». "
                "Допустимы: .xlsx, .xls, .csv, .txt."
            )

    if df.empty:
        raise CleanerError(f"Файл «{path}» пуст: нет строк данных.")

    # Нормализуем имена колонок (пробелы по краям)
    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise CleanerError(
            "Во входном файле отсутствуют обязательные колонки: "
            + ", ".join(f"«{c}»" for c in missing)
            + f". Найдены: {', '.join(f'«{c}»' for c in df.columns)}."
        )

    return df, warnings


def process_leads(df: pd.DataFrame) -> CleaningResult:
    """
    Очищает и нормализует заявки.

    Детерминированное правило:
    1) Нормализуем поля.
    2) Дедупликация по валидному телефону — оставляем первую строку исходного файла.
    3) В clean попадают строки с валидным телефоном и именем (дата может быть битой).
    4) В problem — любая строка с хотя бы одной проблемой (включая «чистые с битой датой»).
    """
    stats = ProcessingStats(total_rows=len(df))
    clean_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    seen_phones: dict[str, int] = {}  # phone -> исходная строка первой записи

    # Нумерация как в Excel/CSV: строка 1 — заголовок, данные с 2
    for position, (_, row) in enumerate(df.iterrows()):
        source_row = position + 2

        raw_name = row.get("Имя")
        raw_phone = row.get("Телефон")
        raw_date = row.get("Дата заявки")
        raw_source = row.get("Источник")

        norm_name = normalize_name(raw_name)
        norm_phone = normalize_phone(raw_phone)
        norm_date = normalize_date(raw_date)

        problems: list[str] = []

        if not is_valid_phone(norm_phone):
            problems.append(PROBLEM_NO_PHONE)
            stats.invalid_phones += 1
            norm_phone = None

        if norm_name is None:
            problems.append(PROBLEM_NO_NAME)
            stats.missing_names += 1

        if norm_date is None:
            problems.append(PROBLEM_BAD_DATE)
            stats.bad_dates += 1

        duplicate_of: int | None = None
        is_duplicate = False
        if norm_phone is not None:
            if norm_phone in seen_phones:
                is_duplicate = True
                duplicate_of = seen_phones[norm_phone]
                problems.append(PROBLEM_DUPLICATE)
                stats.duplicates += 1
            else:
                seen_phones[norm_phone] = source_row

        base = {
            "Исходная строка": source_row,
            "Имя": None if _is_blank(raw_name) else raw_name,
            "Телефон": None if _is_blank(raw_phone) else raw_phone,
            "Дата заявки": None if _is_blank(raw_date) else raw_date,
            "Источник": None if _is_blank(raw_source) else raw_source,
            "Имя_норм": norm_name,
            "Телефон_норм": norm_phone,
            "Дата_норм": norm_date,
        }

        # Чистая уникальная заявка: валидный телефон + имя + не дубликат
        can_be_clean = (
            norm_phone is not None
            and norm_name is not None
            and not is_duplicate
        )

        if can_be_clean:
            clean_rows.append(
                {
                    "Исходная строка": source_row,
                    "Имя": norm_name,
                    "Телефон": norm_phone,
                    "Дата заявки": norm_date,  # может быть None при битой дате
                    "Источник": None if _is_blank(raw_source) else str(raw_source).strip(),
                }
            )
            stats.clean_unique += 1

        if problems:
            if can_be_clean and problems == [PROBLEM_BAD_DATE]:
                status = STATUS_CLEAN_WITH_ISSUES
            elif is_duplicate:
                status = STATUS_DUPLICATE
            else:
                status = STATUS_PROBLEM

            problem_entry = {
                **base,
                "Проблемы": "; ".join(problems),
                "Статус": status,
                "Дубликат строки": duplicate_of,
            }
            problem_rows.append(problem_entry)
            stats.problem_rows += 1

    clean_df = pd.DataFrame(
        clean_rows,
        columns=["Исходная строка", "Имя", "Телефон", "Дата заявки", "Источник"],
    )
    problem_columns = [
        "Исходная строка",
        "Имя",
        "Телефон",
        "Дата заявки",
        "Источник",
        "Имя_норм",
        "Телефон_норм",
        "Дата_норм",
        "Проблемы",
        "Статус",
        "Дубликат строки",
    ]
    problem_df = pd.DataFrame(problem_rows, columns=problem_columns)

    return CleaningResult(clean_df=clean_df, problem_df=problem_df, stats=stats)


def _force_text_columns(worksheet, column_names: tuple[str, ...]) -> None:
    """Помечает колонки как текст, чтобы Excel не съедал ведущий '+' у телефонов."""
    headers = {
        cell.value: cell.column
        for cell in worksheet[1]
        if cell.value is not None
    }
    for name in column_names:
        col_idx = headers.get(name)
        if col_idx is None:
            continue
        for row in range(2, worksheet.max_row + 1):
            cell = worksheet.cell(row=row, column=col_idx)
            if cell.value is None:
                continue
            cell.value = str(cell.value)
            cell.number_format = "@"


def save_results(
    result: CleaningResult,
    output_dir: Path,
    *,
    clean_name: str = "clean_leads.xlsx",
    problem_name: str = "problem_leads.xlsx",
    report_name: str = "processing_report.json",
) -> dict[str, Path]:
    """Сохраняет clean/problem Excel и JSON-отчёт."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / clean_name
    problem_path = output_dir / problem_name
    report_path = output_dir / report_name

    stats_df = pd.DataFrame(
        [{"Показатель": k, "Значение": v} for k, v in result.stats.to_dict().items()]
    )

    try:
        with pd.ExcelWriter(clean_path, engine="openpyxl") as writer:
            result.clean_df.to_excel(writer, sheet_name="Чистые заявки", index=False)
            stats_df.to_excel(writer, sheet_name="Статистика", index=False)
            _force_text_columns(writer.book["Чистые заявки"], ("Телефон",))

        with pd.ExcelWriter(problem_path, engine="openpyxl") as writer:
            result.problem_df.to_excel(writer, sheet_name="Проблемные строки", index=False)
            _force_text_columns(
                writer.book["Проблемные строки"],
                ("Телефон", "Телефон_норм"),
            )

        report_path.write_text(
            json.dumps(result.stats.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise CleanerError(f"Ошибка записи результата в «{output_dir}»: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise CleanerError(f"Ошибка записи Excel/JSON: {exc}") from exc

    return {
        "clean": clean_path,
        "problem": problem_path,
        "report": report_path,
    }


def run_pipeline(input_path: Path, output_dir: Path) -> CleaningResult:
    """Полный пайплайн: загрузка → очистка → сохранение."""
    df, warnings = load_input(input_path)
    result = process_leads(df)
    result.warnings.extend(warnings)
    save_results(result, output_dir)
    return result
