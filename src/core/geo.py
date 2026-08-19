"""Lightweight geospatial helpers.

Deliberately dependency-free (numpy only): geopandas/rasterio are optional
extras, and the gravity-model fallback for spatial diffusion must work on a
low-spec offline deployment (rule #6, #8).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.types import District, RegionConfig

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def distance_matrix(region: RegionConfig) -> pd.DataFrame:
    """Symmetric district x district great-circle distance matrix (km)."""
    names = region.district_names
    lats = np.radians([d.lat for d in region.districts])
    lons = np.radians([d.lon for d in region.districts])
    dphi = lats[:, None] - lats[None, :]
    dlmb = lons[:, None] - lons[None, :]
    a = np.sin(dphi / 2) ** 2 + np.cos(lats)[:, None] * np.cos(lats)[None, :] * np.sin(dlmb / 2) ** 2
    dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=names, columns=names)


def neighbours(
    region: RegionConfig, district: str, k: int = 5, max_km: Optional[float] = None
) -> List[str]:
    """The `k` nearest districts, optionally capped at `max_km`."""
    dist = distance_matrix(region)[district].drop(labels=[district])
    if max_km is not None:
        dist = dist[dist <= max_km]
    return list(dist.sort_values().head(k).index)


def neighbour_map(
    region: RegionConfig, k: int = 5, max_km: Optional[float] = None
) -> Dict[str, List[str]]:
    dist = distance_matrix(region)
    out: Dict[str, List[str]] = {}
    for name in region.district_names:
        series = dist[name].drop(labels=[name])
        if max_km is not None:
            series = series[series <= max_km]
        out[name] = list(series.sort_values().head(k).index)
    return out


def gravity_matrix(
    region: RegionConfig,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.8,
    min_km: float = 10.0,
) -> pd.DataFrame:
    """Row-normalised gravity flow matrix `T[i, j]` = flow from i into j.

    `T[i, j] ~ P_i^alpha * P_j^beta / d_ij^gamma`. Used as the CDR fallback
    (rule #8): when real mobility data is available it replaces this matrix,
    but the platform never depends on a telco negotiation to function.
    """
    names = region.district_names
    pops = np.array([max(d.population, 1) for d in region.districts], dtype=float)
    dist = distance_matrix(region).to_numpy()
    dist = np.maximum(dist, min_km)
    flow = (pops[:, None] ** alpha) * (pops[None, :] ** beta) / (dist**gamma)
    np.fill_diagonal(flow, 0.0)
    row_sums = flow.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return pd.DataFrame(flow / row_sums, index=names, columns=names)


def radiation_matrix(region: RegionConfig) -> pd.DataFrame:
    """Row-normalised radiation-model flow matrix (parameter-free alternative).

    `T[i, j] ~ m_i n_j / ((m_i + s_ij)(m_i + n_j + s_ij))` where `s_ij` is the
    population in the circle of radius d_ij around i, excluding i and j.
    """
    names = region.district_names
    pops = np.array([max(d.population, 1) for d in region.districts], dtype=float)
    dist = distance_matrix(region).to_numpy()
    n = len(names)
    flow = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(dist[i])
        cumulative = np.cumsum(pops[order])
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        for j in range(n):
            if i == j:
                continue
            s_ij = cumulative[rank[j]] - pops[i] - pops[j]
            s_ij = max(s_ij, 0.0)
            m, nn = pops[i], pops[j]
            flow[i, j] = (m * nn) / ((m + s_ij) * (m + nn + s_ij))
    row_sums = flow.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return pd.DataFrame(flow / row_sums, index=names, columns=names)


def population_series(region: RegionConfig) -> pd.Series:
    return pd.Series(
        {d.name: float(d.population) for d in region.districts}, name="population"
    )


def district_frame(region: RegionConfig) -> pd.DataFrame:
    """District attribute table (lat/lon/population/urban/wash/density)."""
    return pd.DataFrame([d.model_dump() for d in region.districts]).set_index("name")


def bbox_contains(region: RegionConfig, lat: float, lon: float) -> bool:
    if region.bbox is None:
        return True
    b = region.bbox
    return b.south <= lat <= b.north and b.west <= lon <= b.east


def subset_region(region: RegionConfig, districts: Sequence[str]) -> RegionConfig:
    """Return a copy of `region` restricted to the named districts."""
    keep: List[District] = [d for d in region.districts if d.name in set(districts)]
    data = region.model_dump()
    data["districts"] = [d.model_dump() for d in keep]
    return RegionConfig.model_validate(data)
