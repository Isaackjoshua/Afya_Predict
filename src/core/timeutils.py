"""ISO-8601 epidemiological week helpers.

The whole platform normalises to a `district x epi-week` grid, so week
arithmetic lives in exactly one place.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, List, Optional

import pandas as pd

WEEK_FORMAT = "%G-W%V"


def to_epi_week(day: date) -> str:
    """Return the ISO week label (e.g. `2026-W07`) containing `day`."""
    iso = day.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def epi_week_start(week: str) -> date:
    """Monday of the given ISO week label."""
    year_str, week_str = week.split("-W")
    return date.fromisocalendar(int(year_str), int(week_str), 1)


def epi_week_end(week: str) -> date:
    return epi_week_start(week) + timedelta(days=6)


def shift_week(week: str, delta: int) -> str:
    """Move `delta` weeks forwards (positive) or backwards (negative)."""
    return to_epi_week(epi_week_start(week) + timedelta(weeks=delta))


def week_range(start: str, end: str) -> List[str]:
    """Inclusive list of ISO week labels from `start` to `end`."""
    cursor, stop = epi_week_start(start), epi_week_start(end)
    if stop < cursor:
        return []
    out: List[str] = []
    while cursor <= stop:
        out.append(to_epi_week(cursor))
        cursor += timedelta(weeks=1)
    return out


def weeks_between(start: str, end: str) -> int:
    """Signed number of weeks from `start` to `end`."""
    return (epi_week_start(end) - epi_week_start(start)).days // 7


def last_n_weeks(n: int, end: Optional[str] = None) -> List[str]:
    end_week = end or to_epi_week(date.today())
    return week_range(shift_week(end_week, -(n - 1)), end_week)


def week_series_to_dates(weeks: Iterable[str]) -> pd.Series:
    """Convert week labels to their Monday dates (handy for plotting)."""
    values = list(weeks)
    return pd.Series([epi_week_start(w) for w in values], index=values, name="week_start")


def sort_weeks(weeks: Iterable[str]) -> List[str]:
    return sorted(set(weeks), key=epi_week_start)


def week_of_year(week: str) -> int:
    return int(week.split("-W")[1])


def year_of_week(week: str) -> int:
    return int(week.split("-W")[0])
