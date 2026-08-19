"""Spatial accuracy: did the disease actually spread where we said it would?

Temporal accuracy alone can hide a model that predicts the right national curve
in entirely the wrong districts. These metrics score the *geography* of the
forecast: hit rate on the top-k risk districts, rank correlation across the
grid, and how well predicted importation matched observed new-district onsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.geo import distance_matrix
from src.core.types import RegionConfig


@dataclass
class SpatialAccuracy:
    """Geographic skill of a single week's forecast."""

    week: str
    top_k: int
    hit_rate: float
    rank_correlation: float
    mean_distance_error_km: float
    predicted_hotspots: List[str]
    actual_hotspots: List[str]

    def summary(self) -> Dict[str, object]:
        return {
            "week": self.week,
            "top_k": self.top_k,
            "hit_rate": round(self.hit_rate, 4),
            "rank_correlation": round(self.rank_correlation, 4),
            "mean_distance_error_km": round(self.mean_distance_error_km, 1),
            "predicted_hotspots": self.predicted_hotspots,
            "actual_hotspots": self.actual_hotspots,
        }


def hotspot_hit_rate(
    predicted: pd.Series, actual: pd.Series, k: int = 10
) -> float:
    """Share of the actual top-k hotspot districts that were predicted top-k."""
    if predicted.empty or actual.empty:
        return float("nan")
    predicted_top = set(predicted.sort_values(ascending=False).head(k).index)
    actual_top = set(actual.sort_values(ascending=False).head(k).index)
    if not actual_top:
        return float("nan")
    return len(predicted_top & actual_top) / len(actual_top)


def rank_correlation(predicted: pd.Series, actual: pd.Series) -> float:
    """Spearman correlation of district rankings — did we order them right?"""
    aligned = pd.concat([predicted.rename("p"), actual.rename("a")], axis=1).dropna()
    if len(aligned) < 3 or aligned["p"].nunique() < 2 or aligned["a"].nunique() < 2:
        return float("nan")
    return float(aligned["p"].corr(aligned["a"], method="spearman"))


def displacement_error(
    predicted: pd.Series, actual: pd.Series, region: RegionConfig, k: int = 5
) -> float:
    """Mean km between each actual hotspot and its nearest predicted hotspot.

    A model that names the neighbouring district is far more useful than one
    that names a council 700 km away; plain hit rate treats both as a miss.
    """
    if predicted.empty or actual.empty:
        return float("nan")
    distances = distance_matrix(region)
    predicted_top = [d for d in predicted.sort_values(ascending=False).head(k).index if d in distances.index]
    actual_top = [d for d in actual.sort_values(ascending=False).head(k).index if d in distances.index]
    if not predicted_top or not actual_top:
        return float("nan")
    return float(np.mean([distances.loc[a, predicted_top].min() for a in actual_top]))


def evaluate_spatial(
    predicted: pd.Series,
    actual: pd.Series,
    region: RegionConfig,
    week: str = "",
    k: int = 10,
) -> SpatialAccuracy:
    """Full spatial scorecard for one week."""
    return SpatialAccuracy(
        week=week,
        top_k=k,
        hit_rate=hotspot_hit_rate(predicted, actual, k),
        rank_correlation=rank_correlation(predicted, actual),
        mean_distance_error_km=displacement_error(predicted, actual, region, k),
        predicted_hotspots=list(predicted.sort_values(ascending=False).head(k).index),
        actual_hotspots=list(actual.sort_values(ascending=False).head(k).index),
    )


def evaluate_spatial_series(
    frame: pd.DataFrame,
    region: RegionConfig,
    predicted_column: str = "predicted",
    actual_column: str = "actual",
    k: int = 10,
) -> pd.DataFrame:
    """Spatial scorecard for every week in a district x week frame."""
    rows = []
    for week, group in frame.groupby(level="week"):
        series = group.droplevel("week")
        rows.append(
            evaluate_spatial(
                series[predicted_column], series[actual_column], region, week=str(week), k=k
            ).summary()
        )
    return pd.DataFrame(rows)


def importation_accuracy(
    predicted_importation: pd.Series,
    new_onset_districts: Sequence[str],
    k: int = 10,
) -> Dict[str, float]:
    """Did high predicted importation precede actual new-district onsets?

    This is the direct test of shortcoming #10: not "was the curve right" but
    "did we name the districts the disease reached next".
    """
    if predicted_importation.empty:
        return {"precision_at_k": float("nan"), "recall_at_k": float("nan"), "n_onsets": 0}
    ranked = list(predicted_importation.sort_values(ascending=False).head(k).index)
    onsets = set(new_onset_districts)
    hits = len(set(ranked) & onsets)
    return {
        "precision_at_k": hits / k if k else float("nan"),
        "recall_at_k": hits / len(onsets) if onsets else float("nan"),
        "n_onsets": len(onsets),
        "hits": hits,
        "top_k": ranked,
    }


def detect_new_onsets(
    incidence: pd.DataFrame, threshold: float, week: str, lookback: int = 4
) -> List[str]:
    """Districts crossing the threshold this week after being quiet before it."""
    if week not in incidence.index:
        return []
    position = list(incidence.index).index(week)
    if position == 0:
        return []
    current = incidence.loc[week]
    window = incidence.iloc[max(0, position - lookback):position]
    quiet_before = (window < threshold).all()
    crossing = current >= threshold
    return list(current.index[crossing & quiet_before])
