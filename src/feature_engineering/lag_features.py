"""Lagged driver features and per-district lag fitting (shortcoming #8, rule #3).

Rainfall-to-malaria lags reported in the literature span 2–8 weeks in some
ecologies and 1–3 months in others; a coefficient transplanted from the Kenyan
highlands is wrong on the Dar es Salaam coast. So the YAML's `optimal_lag_weeks`
is only a *prior*. For each district the fitter scans `lag_weeks_range`, scores
each candidate against the district's own case history, and keeps the winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.core.types import FeatureSpec

log = get_logger("features.lags")

#: Minimum usable observations before a district is allowed its own lag fit;
#: below this it inherits the pooled (national) lag.
MIN_OBS_FOR_LOCAL_FIT = 60


@dataclass
class LagFit:
    """The outcome of scanning one proxy's candidate lags in one district."""

    proxy: str
    district: str
    lag_weeks: int
    score: float
    scores_by_lag: Dict[int, float]
    source: str = "fitted"  # fitted | pooled | prior

    @property
    def is_local(self) -> bool:
        return self.source == "fitted"


def lag_column(variable: str, lag: int) -> str:
    return f"{variable}_lag{lag}"


def add_lag_features(
    panel_values: pd.DataFrame,
    variables: Sequence[str],
    lags: Iterable[int],
) -> pd.DataFrame:
    """Add `<variable>_lag<k>` columns, shifting inside each district."""
    frame = panel_values.sort_index()
    lags = sorted({int(lag) for lag in lags})
    new_columns: Dict[str, pd.Series] = {}
    for variable in variables:
        if variable not in frame.columns:
            continue
        grouped = frame.groupby(level="district")[variable]
        for lag in lags:
            if lag == 0:
                new_columns[lag_column(variable, 0)] = frame[variable]
            else:
                new_columns[lag_column(variable, lag)] = grouped.shift(lag)
    if not new_columns:
        return frame
    return pd.concat([frame, pd.DataFrame(new_columns, index=frame.index)], axis=1)


def _score_lag(driver: pd.Series, target: pd.Series, lag: int) -> float:
    """Absolute Spearman correlation between a lagged driver and the target.

    Spearman rather than Pearson because several proxy relationships are
    monotone but not linear (saturating rainfall, bell-curve temperature), and
    rank correlation is robust to the reporting spikes DHIS2 data carries.
    """
    shifted = driver.shift(lag)
    pair = pd.concat([shifted, target], axis=1).dropna()
    if len(pair) < 12:
        return 0.0
    a, b = pair.iloc[:, 0], pair.iloc[:, 1]
    if a.nunique() < 3 or b.nunique() < 3:
        return 0.0
    rho = a.corr(b, method="spearman")
    return 0.0 if not np.isfinite(rho) else abs(float(rho))


def fit_lag_for_district(
    driver: pd.Series,
    target: pd.Series,
    spec: FeatureSpec,
    district: str,
) -> Optional[LagFit]:
    """Scan `spec.lag_candidates` and return the best-scoring lag."""
    candidates = spec.lag_candidates
    if not candidates:
        return None
    usable = pd.concat([driver, target], axis=1).dropna()
    if len(usable) < MIN_OBS_FOR_LOCAL_FIT:
        return None
    scores = {lag: _score_lag(driver, target, lag) for lag in candidates}
    best_lag = max(scores, key=lambda k: scores[k])
    if scores[best_lag] <= 0.0:
        return None
    return LagFit(
        proxy=spec.name,
        district=district,
        lag_weeks=int(best_lag),
        score=float(scores[best_lag]),
        scores_by_lag={int(k): round(float(v), 4) for k, v in scores.items()},
    )


def fit_optimal_lags(
    panel_values: pd.DataFrame,
    target_column: str,
    specs: Sequence[FeatureSpec],
    variable_for_proxy: Dict[str, str],
    districts: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, LagFit]]:
    """Fit each proxy's lag per district, falling back to pooled then prior.

    Returns `{district: {proxy_name: LagFit}}`. Districts with too little
    history borrow the pooled national fit — transfer learning in its simplest
    form, and the reason a data-poor council still gets a defensible lag.
    """
    districts = list(districts or panel_values.index.get_level_values("district").unique())
    out: Dict[str, Dict[str, LagFit]] = {district: {} for district in districts}

    # Pooled fit first: concatenate every district's series so the national
    # signal is available as a fallback.
    for spec in specs:
        variable = variable_for_proxy.get(spec.name)
        if variable is None or variable not in panel_values.columns:
            continue
        if len(spec.lag_candidates) <= 1:
            lag = spec.optimal_lag_weeks
            for district in districts:
                out[district][spec.name] = LagFit(
                    proxy=spec.name, district=district, lag_weeks=lag,
                    score=0.0, scores_by_lag={lag: 0.0}, source="prior",
                )
            continue

        pooled_scores: Dict[int, List[float]] = {lag: [] for lag in spec.lag_candidates}
        local_fits: Dict[str, LagFit] = {}
        for district in districts:
            try:
                driver = panel_values.xs(district, level="district")[variable]
                target = panel_values.xs(district, level="district")[target_column]
            except KeyError:
                continue
            fit = fit_lag_for_district(driver, target, spec, district)
            if fit is not None:
                local_fits[district] = fit
                for lag, score in fit.scores_by_lag.items():
                    pooled_scores[lag].append(score)

        if any(pooled_scores.values()):
            averaged = {
                lag: float(np.mean(values)) if values else 0.0
                for lag, values in pooled_scores.items()
            }
            pooled_lag = max(averaged, key=lambda k: averaged[k])
            pooled_score = averaged[pooled_lag]
        else:
            averaged = {spec.optimal_lag_weeks: 0.0}
            pooled_lag, pooled_score = spec.optimal_lag_weeks, 0.0

        for district in districts:
            if district in local_fits:
                out[district][spec.name] = local_fits[district]
            else:
                out[district][spec.name] = LagFit(
                    proxy=spec.name,
                    district=district,
                    lag_weeks=int(pooled_lag),
                    score=float(pooled_score),
                    scores_by_lag={int(k): round(float(v), 4) for k, v in averaged.items()},
                    source="pooled",
                )
    return out


def lag_fit_report(fits: Dict[str, Dict[str, LagFit]]) -> pd.DataFrame:
    """Tabulate fitted lags — the evidence that rule #3 is actually honoured."""
    rows = []
    for district, by_proxy in fits.items():
        for proxy, fit in by_proxy.items():
            rows.append(
                {
                    "district": district,
                    "proxy": proxy,
                    "lag_weeks": fit.lag_weeks,
                    "score": round(fit.score, 4),
                    "source": fit.source,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["district", "proxy", "lag_weeks", "score", "source"])
    return pd.DataFrame(rows).sort_values(["proxy", "district"]).reset_index(drop=True)


def lag_dispersion(fits: Dict[str, Dict[str, LagFit]]) -> pd.DataFrame:
    """How much each proxy's fitted lag varies across districts.

    A wide spread is the empirical proof that transplanted coefficients would
    have been wrong — exactly the failure mode shortcoming #8 describes.
    """
    report = lag_fit_report(fits)
    if report.empty:
        return report
    return (
        report.groupby("proxy")["lag_weeks"]
        .agg(min_lag="min", median_lag="median", max_lag="max", unique_lags="nunique")
        .reset_index()
    )
