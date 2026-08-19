"""Rolling statistics, anomalies and z-scores.

Absolute rainfall matters less than *rainfall anomaly*: 90 mm is a drought in
Kyela and a flood in Longido. Every rolling operation is computed inside a
district so no signal leaks across the grid, and every window is strictly
backward-looking so nothing leaks from the future either.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (4, 8, 12)


def _grouped(frame: pd.DataFrame, variable: str):
    return frame.groupby(level="district")[variable]


def add_rolling_means(
    frame: pd.DataFrame, variables: Sequence[str], windows: Iterable[int] = DEFAULT_WINDOWS
) -> pd.DataFrame:
    """Trailing means, e.g. `rainfall_mm_roll4`."""
    frame = frame.sort_index()
    new: Dict[str, pd.Series] = {}
    for variable in variables:
        if variable not in frame.columns:
            continue
        for window in windows:
            new[f"{variable}_roll{window}"] = _grouped(frame, variable).transform(
                lambda s, w=window: s.rolling(w, min_periods=max(2, w // 2)).mean()
            )
    return _concat(frame, new)


def add_rolling_sums(
    frame: pd.DataFrame, variables: Sequence[str], windows: Iterable[int] = DEFAULT_WINDOWS
) -> pd.DataFrame:
    """Trailing sums — accumulated rainfall is what fills breeding sites."""
    frame = frame.sort_index()
    new: Dict[str, pd.Series] = {}
    for variable in variables:
        if variable not in frame.columns:
            continue
        for window in windows:
            new[f"{variable}_sum{window}"] = _grouped(frame, variable).transform(
                lambda s, w=window: s.rolling(w, min_periods=max(2, w // 2)).sum()
            )
    return _concat(frame, new)


def add_anomalies(
    frame: pd.DataFrame, variables: Sequence[str], baseline_weeks: int = 52
) -> pd.DataFrame:
    """Deviation from the district's own trailing baseline, absolute and z-scored.

    The baseline is a rolling window rather than a fixed climatology so the
    reference drifts with the climate instead of freezing a historical normal.
    """
    frame = frame.sort_index()
    new: Dict[str, pd.Series] = {}
    for variable in variables:
        if variable not in frame.columns:
            continue
        grouped = _grouped(frame, variable)
        mean = grouped.transform(
            lambda s: s.rolling(baseline_weeks, min_periods=8).mean().shift(1)
        )
        std = grouped.transform(
            lambda s: s.rolling(baseline_weeks, min_periods=8).std().shift(1)
        )
        anomaly = frame[variable] - mean
        new[f"{variable}_anomaly"] = anomaly
        new[f"{variable}_zscore"] = anomaly / std.replace(0, np.nan)
    return _concat(frame, new)


def add_deltas(frame: pd.DataFrame, variables: Sequence[str], periods: Iterable[int] = (1, 4)) -> pd.DataFrame:
    """Week-on-week and month-on-month change — captures the *rate* of rise."""
    frame = frame.sort_index()
    new: Dict[str, pd.Series] = {}
    for variable in variables:
        if variable not in frame.columns:
            continue
        grouped = _grouped(frame, variable)
        for period in periods:
            new[f"{variable}_delta{period}"] = grouped.transform(
                lambda s, p=period: s.diff(p)
            )
    return _concat(frame, new)


def add_expanding_percentile(frame: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """Where the current value sits in the district's own history (0-1).

    Expanding rather than rolling so early weeks are still usable, and shifted
    by one so the current observation never ranks itself.
    """
    frame = frame.sort_index()
    new: Dict[str, pd.Series] = {}
    for variable in variables:
        if variable not in frame.columns:
            continue
        new[f"{variable}_pctile"] = _grouped(frame, variable).transform(
            lambda s: s.expanding(min_periods=8).apply(
                lambda w: float((w[:-1] <= w[-1]).mean()) if len(w) > 1 else np.nan, raw=True
            )
        )
    return _concat(frame, new)


def rolling_feature_names(variables: Sequence[str], windows: Iterable[int] = DEFAULT_WINDOWS) -> list:
    names = []
    for variable in variables:
        names.extend(f"{variable}_roll{w}" for w in windows)
        names.extend([f"{variable}_anomaly", f"{variable}_zscore"])
    return names


def _concat(frame: pd.DataFrame, new: Dict[str, pd.Series]) -> pd.DataFrame:
    if not new:
        return frame
    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)
