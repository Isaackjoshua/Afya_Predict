"""One Health cross-domain features (shortcoming #5).

Human, animal and environmental surveillance normally sit in separate systems.
Africa CDC's Event Management System found 69% of recorded events affected
humans, yet animal and environmental events were tracked separately — so the
spillover signal that precedes a zoonotic outbreak is visible in the data but
invisible to the analyst.

These features put all three domains in one row and score their *joint*
anomalies, which is where spillover shows up first.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ANIMAL_VARIABLES = (
    "livestock_outbreak_events",
    "livestock_mortality",
    "zoonotic_alert_index",
    "livestock_density",
)
ENVIRONMENT_VARIABLES = (
    "rainfall_mm",
    "temperature_c",
    "ndvi",
    "surface_water_index",
    "pm25_ug_m3",
)


def add_one_health_features(
    frame: pd.DataFrame,
    case_column: Optional[str] = None,
    animal_variables: Sequence[str] = ANIMAL_VARIABLES,
    environment_variables: Sequence[str] = ENVIRONMENT_VARIABLES,
    window: int = 26,
    lags: Sequence[int] = (1, 2, 4, 8),
) -> pd.DataFrame:
    """Add animal-domain lags plus a cross-domain anomaly-convergence score."""
    frame = frame.sort_index()
    new: Dict[str, pd.Series] = {}

    present_animal = [v for v in animal_variables if v in frame.columns]
    for variable in present_animal:
        grouped = frame.groupby(level="district")[variable]
        for lag in lags:
            new[f"{variable}_lag{lag}"] = grouped.shift(lag)
        # Animal outbreaks are rare and bursty: a trailing sum is more
        # informative than an instantaneous count.
        new[f"{variable}_sum8"] = grouped.transform(
            lambda s: s.rolling(8, min_periods=2).sum()
        )

    frame = _concat(frame, new)

    animal_z = _domain_zscore(frame, present_animal, window)
    env_z = _domain_zscore(frame, [v for v in environment_variables if v in frame.columns], window)
    human_z = (
        _domain_zscore(frame, [case_column], window)
        if case_column and case_column in frame.columns
        else None
    )

    extra: Dict[str, pd.Series] = {}
    if animal_z is not None:
        extra["one_health_animal_anomaly"] = animal_z
    if env_z is not None:
        extra["one_health_environment_anomaly"] = env_z
    if animal_z is not None and env_z is not None:
        # Spillover risk is highest when animal *and* environmental signals move
        # together — either alone is far weaker evidence.
        extra["zoonotic_spillover_score"] = (
            np.tanh(animal_z.clip(lower=0)) * np.tanh(env_z.clip(lower=0))
        )
    if human_z is not None and animal_z is not None:
        # Animal signal leading human signal is the classic spillover ordering.
        lead = animal_z.groupby(level="district").shift(4)
        extra["animal_leads_human"] = np.tanh(lead.clip(lower=0)) * np.tanh(human_z.clip(lower=0))

    return _concat(frame, extra)


def _domain_zscore(frame: pd.DataFrame, variables: Sequence[str], window: int) -> Optional[pd.Series]:
    """Mean rolling z-score across a domain's variables."""
    usable = [v for v in variables if v and v in frame.columns]
    if not usable:
        return None
    scores: List[pd.Series] = []
    for variable in usable:
        grouped = frame.groupby(level="district")[variable]
        mean = grouped.transform(lambda s: s.rolling(window, min_periods=6).mean().shift(1))
        std = grouped.transform(lambda s: s.rolling(window, min_periods=6).std().shift(1))
        scores.append((frame[variable] - mean) / std.replace(0, np.nan))
    stacked = pd.concat(scores, axis=1)
    return stacked.mean(axis=1, skipna=True)


def cross_domain_alerts(
    frame: pd.DataFrame, threshold: float = 0.35, top_n: int = 20
) -> pd.DataFrame:
    """District-weeks where animal and environmental anomalies converge.

    Surfaced on the dashboard as early zoonotic-spillover watch items, ahead of
    any human case signal.
    """
    if "zoonotic_spillover_score" not in frame.columns:
        return pd.DataFrame(columns=["district", "week", "zoonotic_spillover_score"])
    scores = frame["zoonotic_spillover_score"].dropna()
    hits = scores[scores >= threshold].sort_values(ascending=False).head(top_n)
    return (
        hits.rename("zoonotic_spillover_score")
        .reset_index()
        .loc[:, ["district", "week", "zoonotic_spillover_score"]]
    )


def _concat(frame: pd.DataFrame, new: Dict[str, pd.Series]) -> pd.DataFrame:
    if not new:
        return frame
    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)
