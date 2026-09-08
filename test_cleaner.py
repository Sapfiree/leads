"""Автоматические тесты модуля очистки заявок."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cleaner import (
    PROBLEM_BAD_DATE,
    PROBLEM_DUPLICATE,
    PROBLEM_NO_NAME,
    PROBLEM_NO_PHONE,
    CleanerError,
    is_valid_phone,
    load_input,
    normalize_date,
    normalize_name,
    normalize_phone,
    process_leads,
    save_results,
)


# ---------------------------------------------------------------------------
# 1–2. Телефоны
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8 (999) 123-45-67", "+79991234567"),
        ("+7 999 123 45 67", "+79991234567"),
        ("7 999 123-45-67", "+79991234567"),
        ("9991234567", "+79991234567"),
        ("89991234567", "+79991234567"),
        ("+7(999)123-45-67", "+79991234567"),
        (89991234567, "+79991234567"),
        (9991234567.0, "+79991234567"),
    ],
)
def test_normalize_phone_valid(raw, expected):
    assert normalize_phone(raw) == expected
    assert is_valid_phone(expected)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "12345",
        "123456789",  # 9 цифр
        "123456789012",  # 12 цифр
        "abc",
        "-",
        "+1 999 123 45 67",
        "999123456",  # 9 цифр
    ],
)
def test_normalize_phone_invalid(raw):
    assert normalize_phone(raw) is None


# ---------------------------------------------------------------------------
# 3. Имена
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  иВАН   иВАНОВ  ", "Иван Иванов"),
        ("анна-мария", "Анна-Мария"),
        ("ПЕТР СЕРГЕЕВИЧ", "Петр Сергеевич"),
        ("мария", "Мария"),
        ("Анна-мария Петрова", "Анна-Мария Петрова"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "-", "нет", "n/a", "None"])
def test_normalize_name_empty(raw):
    assert normalize_name(raw) is None


# ---------------------------------------------------------------------------
# 4–5. Даты
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.02.2024", "2024-02-10"),
        ("10-02-2024", "2024-02-10"),
        ("10/02/2024", "2024-02-10"),
        ("2024-02-10", "2024-02-10"),
        ("2024-02-10 14:30", "2024-02-10"),
        ("15 февраля 2024", "2024-02-15"),
        (pd.Timestamp("2024-02-10 14:30:00"), "2024-02-10"),
    ],
)
def test_normalize_date_valid(raw, expected):
    assert normalize_date(raw) == expected


def test_normalize_date_unrecognized():
    assert normalize_date("не дата") is None
    assert normalize_date("абвгд") is None
    assert normalize_date("") is None
    assert normalize_date(None) is None


# ---------------------------------------------------------------------------
# 6. Дедупликация
# ---------------------------------------------------------------------------


def test_deduplication_keeps_first():
    df = pd.DataFrame(
        [
            {
                "Имя": "Иван Иванов",
                "Телефон": "89991234567",
                "Дата заявки": "10.02.2024",
                "Источник": "сайт",
            },
            {
                "Имя": "Иван Дубль",
                "Телефон": "+7 999 123 45 67",
                "Дата заявки": "11.02.2024",
                "Источник": "Telegram",
            },
            {
                "Имя": "Мария",
                "Телефон": "89001112233",
                "Дата заявки": "12.02.2024",
                "Источник": "email",
            },
        ]
    )
    result = process_leads(df)

    assert len(result.clean_df) == 2
    assert list(result.clean_df["Телефон"]) == ["+79991234567", "+79001112233"]
    assert result.clean_df.iloc[0]["Имя"] == "Иван Иванов"
    assert result.stats.duplicates == 1

    dup = result.problem_df[
        result.problem_df["Проблемы"].str.contains(PROBLEM_DUPLICATE)
    ]
    assert len(dup) == 1
    assert dup.iloc[0]["Дубликат строки"] == 2
    assert dup.iloc[0]["Исходная строка"] == 3


# ---------------------------------------------------------------------------
# 7. Несколько проблем в одной строке
# ---------------------------------------------------------------------------


def test_multiple_problems_in_one_row():
    df = pd.DataFrame(
        [
            {
                "Имя": "  ",
                "Телефон": "123",
                "Дата заявки": "xxx",
                "Источник": "сайт",
            }
        ]
    )
    result = process_leads(df)
    assert len(result.clean_df) == 0
    assert len(result.problem_df) == 1
    problems = result.problem_df.iloc[0]["Проблемы"]
    assert PROBLEM_NO_PHONE in problems
    assert PROBLEM_NO_NAME in problems
    assert PROBLEM_BAD_DATE in problems


# ---------------------------------------------------------------------------
# 8. Сохранение проблемных строк + правило битой даты
# ---------------------------------------------------------------------------


def test_broken_date_goes_to_clean_and_problem(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "Имя": "Кирилл Орлов",
                "Телефон": "89005554433",
                "Дата заявки": "не дата",
                "Источник": "звонок",
            }
        ]
    )
    result = process_leads(df)

    assert len(result.clean_df) == 1
    assert result.clean_df.iloc[0]["Телефон"] == "+79005554433"
    assert pd.isna(result.clean_df.iloc[0]["Дата заявки"]) or result.clean_df.iloc[0][
        "Дата заявки"
    ] is None

    assert len(result.problem_df) == 1
    assert PROBLEM_BAD_DATE in result.problem_df.iloc[0]["Проблемы"]
    assert result.problem_df.iloc[0]["Статус"] == "в чистых с замечаниями"

    paths = save_results(result, tmp_path)
    assert paths["clean"].exists()
    assert paths["problem"].exists()
    assert paths["report"].exists()

    clean_xlsx = pd.read_excel(paths["clean"], sheet_name="Чистые заявки")
    problem_xlsx = pd.read_excel(paths["problem"], sheet_name="Проблемные строки")
    assert len(clean_xlsx) == 1
    assert len(problem_xlsx) == 1


def test_no_phone_not_in_clean():
    df = pd.DataFrame(
        [
            {
                "Имя": "Олег",
                "Телефон": "12345",
                "Дата заявки": "10.02.2024",
                "Источник": "email",
            }
        ]
    )
    result = process_leads(df)
    assert len(result.clean_df) == 0
    assert len(result.problem_df) == 1
    assert PROBLEM_NO_PHONE in result.problem_df.iloc[0]["Проблемы"]


# ---------------------------------------------------------------------------
# 9. Отсутствие обязательной колонки
# ---------------------------------------------------------------------------


def test_missing_required_column(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("Имя;Телефон;Источник\nИван;89991234567;сайт\n", encoding="utf-8")
    with pytest.raises(CleanerError) as exc_info:
        load_input(path)
    assert "обязательные колонки" in str(exc_info.value).lower() or "Дата заявки" in str(
        exc_info.value
    )


def test_file_not_found():
    with pytest.raises(CleanerError) as exc_info:
        load_input(Path("definitely_missing_file_12345.xlsx"))
    assert "не найден" in str(exc_info.value).lower()


def test_excel_misnamed_as_csv(tmp_path: Path):
    """Excel, переименованный в .csv, должен читаться как Excel."""
    path = tmp_path / "leads.csv"
    df_in = pd.DataFrame(
        [
            {
                "Имя": "Иван",
                "Телефон": "89991234567",
                "Дата заявки": "10.02.2024",
                "Источник": "сайт",
            }
        ]
    )
    df_in.to_excel(path, index=False)
    df, warnings = load_input(path)
    assert len(df) == 1
    assert any("Excel" in w for w in warnings)


def test_sample_input_pipeline():
    sample = Path(__file__).parent / "sample_input.csv"
    df, _ = load_input(sample)
    result = process_leads(df)

    # Чистые телефоны строго +7XXXXXXXXXX
    for phone in result.clean_df["Телефон"]:
        assert is_valid_phone(phone)

    # Нет дублей в clean
    assert result.clean_df["Телефон"].is_unique

    # Ожидаемые чистые: Иван, Кирилл(битая дата), Елена, Василий, Наталия, Алексей(битая дата)
    # нет: дубли Ивана, без телефона, без имени, Олег(плохой телефон), нет(имя), дубль Василия
    assert result.stats.clean_unique == 6
    assert result.stats.duplicates == 5  # 4 дубля к Ивану + 1 к Василию
    assert result.stats.problem_rows >= 1
    assert len(result.clean_df) + result.stats.duplicates <= result.stats.total_rows
