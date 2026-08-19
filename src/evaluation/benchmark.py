"""Naive baselines and the deployment gate (critical rule #10).

A model that cannot beat "same week last year" has learned nothing worth
deploying, however good its R² looks. Three baselines are mandatory:

1. **seasonal naive** — the same epidemiological week one year ago;
2. **rolling mean** — the trailing four-week average;
3. **autoregressive** — an AR(1)-style persistence forecast.

`benchmark_against_naive` returns a pass/fail verdict, and the training scripts
refuse to promote a model that fails it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.evaluation.metrics import regression_metrics, skill_score

log = get_logger("evaluation.benchmark")

SEASON_LENGTH = 52


# ------------------------------------------------------------- the baselines
def seasonal_naive(series: pd.Series, horizon: int, season: int = SEASON_LENGTH) -> pd.Series:
    """Forecast for week t+h = the value at t+h-52."""
    return series.shift(season - horizon)


def rolling_mean_baseline(series: pd.Series, window: int = 4) -> pd.Series:
    """Forecast = the trailing `window`-week mean, known at forecast time."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def autoregressive_baseline(series: pd.Series, alpha: float = 0.7) -> pd.Series:
    """Exponentially smoothed persistence — a strong, cheap competitor."""
    return series.shift(1).ewm(alpha=1 - alpha, adjust=False).mean()


BASELINES = {
    "seasonal_naive": "same epidemiological week last year",
    "rolling_mean_4w": "trailing 4-week mean",
    "autoregressive": "exponentially smoothed persistence",
}


def build_baselines(
    actual_history: pd.Series, horizon: int, season: int = SEASON_LENGTH
) -> pd.DataFrame:
    """All three baseline forecasts, aligned to `actual_history`'s index."""
    return pd.DataFrame(
        {
            "seasonal_naive": seasonal_naive(actual_history, horizon, season),
            "rolling_mean_4w": rolling_mean_baseline(actual_history),
            "autoregressive": autoregressive_baseline(actual_history),
        }
    )


# ------------------------------------------------------------------- verdict
@dataclass
class BaselineComparison:
    """Model-versus-baseline result and the deployment verdict."""

    model_mae: float
    baseline_mae: Dict[str, float] = field(default_factory=dict)
    skill: Dict[str, float] = field(default_factory=dict)
    beaten: Dict[str, bool] = field(default_factory=dict)
    n: int = 0

    @property
    def passes(self) -> bool:
        """Rule #10: the model must beat every baseline, not just the easiest."""
        return bool(self.beaten) and all(self.beaten.values())

    @property
    def failed_against(self) -> List[str]:
        return [name for name, won in self.beaten.items() if not won]

    def verdict(self) -> str:
        if not self.beaten:
            return "NOT EVALUATED — no overlapping baseline observations"
        if self.passes:
            best = min(self.skill, key=lambda k: self.skill[k])
            return (
                f"PASS — beats all {len(self.beaten)} naive baselines "
                f"(narrowest margin {self.skill[best]:+.1%} against {best})"
            )
        return (
            f"FAIL — does not beat {', '.join(self.failed_against)}; "
            "not fit for deployment under rule #10"
        )

    def to_frame(self) -> pd.DataFrame:
        rows = [{"model": "afya_predict", "mae": self.model_mae, "skill_vs_baseline": 0.0,
                 "beaten": None, "description": "fused multi-source model"}]
        for name, mae in self.baseline_mae.items():
            rows.append(
                {
                    "model": name,
                    "mae": mae,
                    "skill_vs_baseline": self.skill.get(name, float("nan")),
                    "beaten": self.beaten.get(name),
                    "description": BASELINES.get(name, ""),
                }
            )
        return pd.DataFrame(rows)


def benchmark_against_naive(
    actual: pd.Series,
    predicted: pd.Series,
    history: Optional[pd.Series] = None,
    horizon: int = 4,
    season: int = SEASON_LENGTH,
    margin: float = 0.0,
) -> BaselineComparison:
    """Compare a model's MAE against all three naive baselines.

    `actual`/`predicted` are indexed by week (or district-week); `history` is
    the full observed case series the baselines are built from.
    """
    history = history if history is not None else actual
    baselines = build_baselines(history, horizon, season).reindex(actual.index)

    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted"), baselines], axis=1)
    aligned = aligned.dropna(subset=["actual", "predicted"])
    if aligned.empty:
        return BaselineComparison(model_mae=float("nan"))

    model_mae = float(np.mean(np.abs(aligned["predicted"] - aligned["actual"])))
    comparison = BaselineComparison(model_mae=model_mae, n=len(aligned))

    for name in BASELINES:
        subset = aligned.dropna(subset=[name])
        if len(subset) < 5:
            continue
        baseline_mae = float(np.mean(np.abs(subset[name] - subset["actual"])))
        # Recompute the model's MAE on the same rows for a fair comparison.
        model_on_subset = float(np.mean(np.abs(subset["predicted"] - subset["actual"])))
        comparison.baseline_mae[name] = baseline_mae
        comparison.skill[name] = skill_score(model_on_subset, baseline_mae)
        comparison.beaten[name] = model_on_subset < baseline_mae * (1 - margin)
    return comparison


def benchmark_by_district(
    frame: pd.DataFrame,
    actual_column: str = "actual",
    predicted_column: str = "predicted",
    horizon: int = 4,
) -> pd.DataFrame:
    """Per-district benchmark table.

    A national average can hide districts where the model is worse than useless,
    which is exactly the failure external validation is supposed to catch.
    """
    rows = []
    for district, group in frame.groupby(level="district"):
        series = group.droplevel("district")
        comparison = benchmark_against_naive(
            series[actual_column], series[predicted_column], horizon=horizon
        )
        rows.append(
            {
                "district": district,
                "n": comparison.n,
                "model_mae": comparison.model_mae,
                **{f"skill_vs_{k}": v for k, v in comparison.skill.items()},
                "passes": comparison.passes,
            }
        )
    return pd.DataFrame(rows).sort_values("model_mae").reset_index(drop=True)
