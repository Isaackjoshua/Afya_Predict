"""Seasonal encoding: Fourier terms, cyclic features and school/holiday flags.

Tanzania's disease seasons are driven by two rainy seasons in the north and one
in the south, so a single annual sine is not enough — the builder adds Fourier
harmonics up to the requested order and lets the model choose.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.core.timeutils import week_of_year, year_of_week

DEFAULT_FOURIER_ORDER = 3


def add_seasonality(
    frame: pd.DataFrame, fourier_order: int = DEFAULT_FOURIER_ORDER, add_trend: bool = True
) -> pd.DataFrame:
    """Add Fourier harmonics, cyclic week encoding and an optional linear trend."""
    weeks = frame.index.get_level_values("week")
    woy = np.array([week_of_year(w) for w in weeks], dtype=float)
    years = np.array([year_of_week(w) for w in weeks], dtype=float)
    phase = 2 * np.pi * woy / 52.1775

    new: Dict[str, np.ndarray] = {
        "week_of_year": woy,
        "week_sin": np.sin(phase),
        "week_cos": np.cos(phase),
    }
    for k in range(1, fourier_order + 1):
        new[f"fourier_sin{k}"] = np.sin(k * phase)
        new[f"fourier_cos{k}"] = np.cos(k * phase)

    # Tanzanian season labels, useful both as features and for explanations.
    new["season_masika"] = ((woy >= 9) & (woy <= 21)).astype(float)   # long rains Mar-May
    new["season_vuli"] = ((woy >= 40) & (woy <= 52)).astype(float)    # short rains Oct-Dec
    new["season_dry"] = ((woy >= 22) & (woy <= 39)).astype(float)     # cool dry Jun-Sep

    if add_trend:
        elapsed = (years - years.min()) * 52.1775 + woy
        new["time_index"] = elapsed / 52.1775  # in years

    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)


def seasonal_feature_names(fourier_order: int = DEFAULT_FOURIER_ORDER) -> List[str]:
    names = ["week_of_year", "week_sin", "week_cos", "season_masika", "season_vuli", "season_dry"]
    for k in range(1, fourier_order + 1):
        names.extend([f"fourier_sin{k}", f"fourier_cos{k}"])
    return names


def seasonal_baseline(series: pd.Series, min_years: int = 2) -> pd.Series:
    """Mean value per week-of-year across the history (the naive #1 baseline)."""
    weeks = series.index.get_level_values("week") if series.index.nlevels > 1 else series.index
    woy = pd.Index([week_of_year(str(w)) for w in weeks], name="week_of_year")
    grouped = series.groupby(woy).mean()
    years = len({year_of_week(str(w)) for w in weeks})
    if years < min_years:
        return pd.Series(dtype=float)
    return grouped
