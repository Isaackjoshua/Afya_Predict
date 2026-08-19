"""Epidemiological week arithmetic."""

from __future__ import annotations

from datetime import date

from src.core.timeutils import (
    epi_week_end,
    epi_week_start,
    last_n_weeks,
    shift_week,
    sort_weeks,
    to_epi_week,
    week_range,
    weeks_between,
)


def test_iso_week_label_round_trips():
    assert to_epi_week(date(2026, 8, 19)) == "2026-W34"
    assert epi_week_start("2026-W34") == date(2026, 8, 17)
    assert epi_week_end("2026-W34") == date(2026, 8, 23)


def test_shift_crosses_the_year_boundary():
    assert shift_week("2025-W01", -1) == "2024-W52"
    assert shift_week("2024-W52", 1) == "2025-W01"


def test_week_range_is_inclusive_and_ordered():
    weeks = week_range("2024-W01", "2024-W05")
    assert weeks == ["2024-W01", "2024-W02", "2024-W03", "2024-W04", "2024-W05"]
    assert week_range("2024-W05", "2024-W01") == []


def test_weeks_between_is_signed():
    assert weeks_between("2024-W01", "2024-W10") == 9
    assert weeks_between("2024-W10", "2024-W01") == -9


def test_last_n_weeks_ends_on_the_anchor():
    weeks = last_n_weeks(4, end="2024-W10")
    assert weeks == ["2024-W07", "2024-W08", "2024-W09", "2024-W10"]


def test_sort_weeks_handles_year_rollover():
    assert sort_weeks(["2025-W02", "2024-W51", "2025-W01"]) == [
        "2024-W51", "2025-W01", "2025-W02",
    ]
