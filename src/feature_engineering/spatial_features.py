"""Neighbour-district aggregates and connectivity features.

Two distinct spatial mechanisms are encoded here:

* **Contiguity** — a district surrounded by rising incidence is at risk even if
  its own counts are flat. Neighbour aggregates borrow strength across the grid,
  the same idea that makes Bayesian hierarchical models work with sparse data
  (shortcoming #7).
* **Connectivity** — importation pressure weighted by *travel*, not distance,
  which is handled in `mobility_features` and `models/spatial_diffusion`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.geo import distance_matrix, neighbour_map
from src.core.types import RegionConfig

DEFAULT_K = 5


def add_neighbour_features(
    frame: pd.DataFrame,
    region: RegionConfig,
    variables: Sequence[str],
    k: int = DEFAULT_K,
    max_km: Optional[float] = 300.0,
) -> pd.DataFrame:
    """Add `<variable>_nbr_mean` and `<variable>_nbr_max` over the k nearest districts."""
    neighbours = neighbour_map(region, k=k, max_km=max_km)
    new: Dict[str, pd.Series] = {}
    for variable in variables:
        if variable not in frame.columns:
            continue
        wide = frame[variable].unstack("district")
        means = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
        maxima = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
        for district in wide.columns:
            pool = [n for n in neighbours.get(district, []) if n in wide.columns]
            if not pool:
                means[district] = np.nan
                maxima[district] = np.nan
                continue
            means[district] = wide[pool].mean(axis=1)
            maxima[district] = wide[pool].max(axis=1)
        new[f"{variable}_nbr_mean"] = _restack(means, frame.index)
        new[f"{variable}_nbr_max"] = _restack(maxima, frame.index)
    return _concat(frame, new)


def add_neighbour_gradient(
    frame: pd.DataFrame, region: RegionConfig, variables: Sequence[str], k: int = DEFAULT_K
) -> pd.DataFrame:
    """District value minus its neighbourhood mean — a local hotspot signal."""
    frame = add_neighbour_features(frame, region, variables, k=k)
    new: Dict[str, pd.Series] = {}
    for variable in variables:
        column = f"{variable}_nbr_mean"
        if variable in frame.columns and column in frame.columns:
            new[f"{variable}_nbr_gradient"] = frame[variable] - frame[column]
    return _concat(frame, new)


def add_connectivity_features(frame: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    """Static structural attributes: centrality, remoteness, urban flag.

    These are what let a model learn that a transport hub behaves differently
    from an isolated council even when their climate is identical.
    """
    distances = distance_matrix(region)
    populations = pd.Series({d.name: float(d.population) for d in region.districts})
    # Population-weighted accessibility: sum_j P_j / d_ij.
    inverse = 1.0 / distances.replace(0, np.nan)
    accessibility = (inverse.mul(populations, axis=1)).sum(axis=1, skipna=True)

    attributes = pd.DataFrame(
        {
            "accessibility_index": accessibility / accessibility.max(),
            "remoteness_km": distances.replace(0, np.nan).min(axis=1),
            "urban_flag": pd.Series({d.name: float(d.urban) for d in region.districts}),
            "wash_access_static": pd.Series({d.name: d.wash_access for d in region.districts}),
            "log_population": np.log1p(populations),
        }
    )
    districts = frame.index.get_level_values("district")
    aligned = attributes.reindex(districts)
    aligned.index = frame.index
    return pd.concat([frame, aligned], axis=1)


def neighbour_case_pressure(
    frame: pd.DataFrame, region: RegionConfig, case_column: str, k: int = DEFAULT_K
) -> pd.Series:
    """Distance-weighted incidence in surrounding districts (per 1,000)."""
    populations = pd.Series({d.name: float(d.population) for d in region.districts})
    incidence = frame[case_column] / frame.index.get_level_values("district").map(
        populations
    ).to_numpy() * 1000.0
    wide = incidence.unstack("district")
    distances = distance_matrix(region)
    neighbours = neighbour_map(region, k=k)

    out = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for district in wide.columns:
        pool = [n for n in neighbours.get(district, []) if n in wide.columns]
        if not pool:
            out[district] = 0.0
            continue
        weights = 1.0 / distances.loc[district, pool].clip(lower=10.0)
        weights = weights / weights.sum()
        out[district] = (wide[pool] * weights).sum(axis=1)
    return _restack(out, frame.index).rename("neighbour_case_pressure")


def _restack(wide: pd.DataFrame, index: pd.MultiIndex) -> pd.Series:
    stacked = wide.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["week", "district"])
    return stacked.reorder_levels(["district", "week"]).reindex(index)


def _concat(frame: pd.DataFrame, new: Dict[str, pd.Series]) -> pd.DataFrame:
    if not new:
        return frame
    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)
