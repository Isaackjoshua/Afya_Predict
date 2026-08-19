"""Mobility-derived features and importation risk (shortcoming #10).

Predicting *when* risk rises is only half the problem; agencies need to know
*where it will spread next*. These features weight each district's exposure by
actual travel flow, using empirical CDR data when a telco agreement exists and
a gravity model otherwise (rule #8).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.geo import gravity_matrix, radiation_matrix
from src.core.logging import get_logger
from src.core.types import RegionConfig

log = get_logger("features.mobility")


def get_travel_matrix(
    region: RegionConfig,
    week: Optional[str] = None,
    model: str = "gravity",
    settings=None,
) -> pd.DataFrame:
    """Best available origin -> destination flow matrix for the region."""
    try:
        from src.data_ingestion.adapters.cdr_mobility import CDRMobilityAdapter

        adapter = CDRMobilityAdapter(region=region, settings=settings)
        matrix = adapter.travel_matrix(week=week, model=model)
        if matrix is not None and not matrix.empty:
            return matrix
    except Exception as exc:  # noqa: BLE001
        log.warning("CDR travel matrix unavailable (%s); using analytical fallback", exc)
    return radiation_matrix(region) if model == "radiation" else gravity_matrix(region)


def importation_pressure(
    frame: pd.DataFrame,
    region: RegionConfig,
    case_column: str,
    travel_matrix: Optional[pd.DataFrame] = None,
    lag_weeks: int = 1,
) -> pd.Series:
    """Incidence in other districts, weighted by their travel flow into each one.

    `pressure_j(t) = sum_i T[i, j] * incidence_i(t - lag)`

    `T` is row-normalised, so this is a flow-weighted average of the incidence a
    district's inbound travellers were exposed to.
    """
    matrix = travel_matrix if travel_matrix is not None else get_travel_matrix(region)
    populations = pd.Series({d.name: float(d.population) for d in region.districts})

    districts = frame.index.get_level_values("district")
    incidence = frame[case_column] / districts.map(populations).to_numpy() * 1000.0
    wide = incidence.unstack("district").sort_index()
    if lag_weeks:
        wide = wide.shift(lag_weeks)

    common = [d for d in wide.columns if d in matrix.index]
    matrix = matrix.loc[common, common]
    wide = wide[common].fillna(0.0)

    # (weeks x districts) @ (origins x destinations) -> weeks x destinations
    pressure = pd.DataFrame(wide.to_numpy() @ matrix.to_numpy(), index=wide.index, columns=common)
    return _restack(pressure, frame.index).rename("importation_pressure")


def add_mobility_features(
    frame: pd.DataFrame,
    region: RegionConfig,
    case_column: Optional[str] = None,
    travel_matrix: Optional[pd.DataFrame] = None,
    lags: Sequence[int] = (1, 2, 4),
) -> pd.DataFrame:
    """Add flow-normalised mobility volumes and lagged importation pressure."""
    new: Dict[str, pd.Series] = {}
    populations = pd.Series({d.name: float(d.population) for d in region.districts})
    districts = frame.index.get_level_values("district")
    pop_aligned = pd.Series(districts.map(populations).to_numpy(), index=frame.index)

    # Raw trip counts scale with district size; per-capita rates do not.
    for variable in ("mobility_inbound", "mobility_outbound", "mobility_internal"):
        if variable in frame.columns:
            new[f"{variable}_per_capita"] = frame[variable] / pop_aligned.replace(0, np.nan)

    if "mobility_inbound" in frame.columns and "mobility_outbound" in frame.columns:
        total = frame["mobility_inbound"] + frame["mobility_outbound"]
        new["mobility_net_inflow"] = frame["mobility_inbound"] - frame["mobility_outbound"]
        new["mobility_net_inflow_ratio"] = new["mobility_net_inflow"] / total.replace(0, np.nan)

    frame = _concat(frame, new)

    if case_column and case_column in frame.columns:
        matrix = travel_matrix if travel_matrix is not None else get_travel_matrix(region)
        pressure_new: Dict[str, pd.Series] = {}
        for lag in lags:
            pressure_new[f"importation_pressure_lag{lag}"] = importation_pressure(
                frame, region, case_column, travel_matrix=matrix, lag_weeks=lag
            )
        frame = _concat(frame, pressure_new)
    return frame


def top_source_districts(
    frame: pd.DataFrame,
    region: RegionConfig,
    district: str,
    week: str,
    case_column: str,
    travel_matrix: Optional[pd.DataFrame] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Rank the districts contributing most importation risk to `district`.

    Powers the `source_districts` field of a prediction: not just "risk is
    high" but "it is arriving from Kahama and Geita".
    """
    matrix = travel_matrix if travel_matrix is not None else get_travel_matrix(region)
    if district not in matrix.columns:
        return pd.DataFrame(columns=["district", "flow_weight", "active_cases", "contributed_risk"])

    populations = pd.Series({d.name: float(d.population) for d in region.districts})
    try:
        week_slice = frame.xs(week, level="week")
    except KeyError:
        return pd.DataFrame(columns=["district", "flow_weight", "active_cases", "contributed_risk"])

    cases = week_slice[case_column].fillna(0.0)
    incidence = cases / cases.index.map(populations).to_numpy() * 1000.0
    inflow = matrix[district].drop(labels=[district], errors="ignore")
    common = [d for d in inflow.index if d in incidence.index]

    contributions = (inflow.loc[common] * incidence.loc[common]).sort_values(ascending=False)
    total = float(contributions.sum())
    rows = [
        {
            "district": name,
            "flow_weight": round(float(inflow[name]), 6),
            "active_cases": float(cases.get(name, 0.0)),
            "contributed_risk": round(float(value / total) if total > 0 else 0.0, 4),
        }
        for name, value in contributions.head(top_n).items()
        if value > 0
    ]
    return pd.DataFrame(rows)


def _restack(wide: pd.DataFrame, index: pd.MultiIndex) -> pd.Series:
    stacked = wide.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["week", "district"])
    return stacked.reorder_levels(["district", "week"]).reindex(index)


def _concat(frame: pd.DataFrame, new: Dict[str, pd.Series]) -> pd.DataFrame:
    if not new:
        return frame
    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)
