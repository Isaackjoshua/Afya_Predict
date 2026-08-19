"""Probability and interval calibration.

A forecast that says "95% interval" and covers 60% of outcomes is worse than no
interval at all, because it invites false confidence in resource decisions.
This module measures the gap and produces the reliability curve the dashboard
plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CalibrationReport:
    """Reliability of the interval and of the risk score."""

    coverage: float
    nominal: float
    mean_width: float
    reliability: pd.DataFrame
    expected_calibration_error: float
    n: int

    @property
    def is_calibrated(self) -> bool:
        """Within 10 percentage points of nominal is the usual working bar."""
        return abs(self.coverage - self.nominal) <= 0.10

    def summary(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "coverage": round(self.coverage, 4),
            "nominal": self.nominal,
            "coverage_gap": round(self.coverage - self.nominal, 4),
            "mean_width": round(self.mean_width, 3),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "is_calibrated": self.is_calibrated,
        }


def interval_calibration(
    actual: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    nominal: float = 0.95,
    n_bins: int = 10,
) -> CalibrationReport:
    """Empirical coverage of the prediction intervals, overall and by magnitude."""
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    a, lo, hi = a[mask], lo[mask], hi[mask]
    if a.size == 0:
        return CalibrationReport(
            coverage=float("nan"), nominal=nominal, mean_width=float("nan"),
            reliability=pd.DataFrame(), expected_calibration_error=float("nan"), n=0,
        )

    inside = (a >= lo) & (a <= hi)
    centre = (lo + hi) / 2.0
    # Bin by forecast magnitude: coverage often degrades at the high end, which
    # is precisely where the decisions are made.
    edges = np.unique(np.quantile(centre, np.linspace(0, 1, n_bins + 1)))
    rows = []
    errors = []
    if len(edges) > 1:
        bins = np.clip(np.digitize(centre, edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            selected = bins == b
            if not selected.any():
                continue
            observed = float(inside[selected].mean())
            rows.append(
                {
                    "bin": b,
                    "forecast_low": float(edges[b]),
                    "forecast_high": float(edges[b + 1]),
                    "n": int(selected.sum()),
                    "coverage": observed,
                    "nominal": nominal,
                    "gap": observed - nominal,
                    "mean_width": float((hi - lo)[selected].mean()),
                }
            )
            errors.append(abs(observed - nominal) * selected.sum())

    ece = float(np.sum(errors) / a.size) if errors else float("nan")
    return CalibrationReport(
        coverage=float(inside.mean()),
        nominal=nominal,
        mean_width=float((hi - lo).mean()),
        reliability=pd.DataFrame(rows),
        expected_calibration_error=ece,
        n=int(a.size),
    )


def risk_score_reliability(
    risk_scores: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10
) -> pd.DataFrame:
    """Reliability diagram: predicted risk score against observed frequency.

    A well-calibrated 0.8 risk score should be followed by an outbreak about
    80% of the time.
    """
    s = np.asarray(risk_scores, dtype=float)
    y = np.asarray(outcomes, dtype=bool)
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    if s.size == 0:
        return pd.DataFrame(columns=["bin_low", "bin_high", "n", "mean_score", "observed_rate", "gap"])

    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        selected = bins == b
        if not selected.any():
            continue
        mean_score = float(s[selected].mean())
        observed = float(y[selected].mean())
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": int(selected.sum()),
                "mean_score": mean_score,
                "observed_rate": observed,
                "gap": observed - mean_score,
            }
        )
    return pd.DataFrame(rows)


def brier_score(risk_scores: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Mean squared error of the probabilistic risk score (lower is better)."""
    s = np.asarray(risk_scores, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(np.mean((s[mask] - y[mask]) ** 2))


def recalibration_factor(
    actual: Sequence[float], lower: Sequence[float], upper: Sequence[float], nominal: float = 0.95
) -> float:
    """Multiplier that would bring interval coverage back to nominal.

    Applied by the retraining loop so intervals stay honest as data quality
    changes across seasons.
    """
    report = interval_calibration(actual, lower, upper, nominal)
    if not np.isfinite(report.coverage) or report.coverage <= 0:
        return 1.0
    if report.coverage >= nominal:
        # Over-covering: intervals can safely narrow, but never below 60%.
        return float(max(0.6, report.coverage and nominal / report.coverage))
    return float(min(3.0, nominal / max(report.coverage, 1e-6)))
