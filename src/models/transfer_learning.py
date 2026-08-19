"""Cross-district knowledge transfer (shortcoming #8).

The tension this module resolves: a model transplanted from a data-rich
district to a data-poor one is wrong (different ecology, different lags), but a
data-poor district cannot fit its own model either.

The compromise the platform takes is explicit and auditable:

1. a **pooled** model is always fitted across all districts;
2. a district with enough history gets its **own** model;
3. a district without gets the pooled model, plus a *local calibration* — a
   scalar bias/scale correction fitted on whatever local data does exist, which
   needs far fewer observations than a full refit;
4. donors are chosen by **ecological similarity**, not proximity, so a highland
   council borrows from other highland councils rather than from its
   geographically nearest lowland neighbour.

Every borrowed prediction is labelled as such, so an official can see that a
forecast rests on transferred structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.geo import distance_matrix, district_frame
from src.core.logging import get_logger
from src.core.types import RegionConfig

log = get_logger("models.transfer")

#: Attributes used to judge whether two districts are ecologically comparable.
SIMILARITY_FEATURES = ("lat", "lon", "population", "density_km2", "wash_access", "urban")

#: Rows of local history below which a full local fit is not attempted.
MIN_ROWS_FOR_LOCAL_FIT = 78
#: Rows below which even a calibration is not trustworthy.
MIN_ROWS_FOR_CALIBRATION = 12


@dataclass
class LocalCalibration:
    """Scalar `a + b * prediction` correction fitted on sparse local data."""

    district: str
    intercept: float = 0.0
    slope: float = 1.0
    n_rows: int = 0
    r2_before: float = 0.0
    r2_after: float = 0.0
    applied: bool = False

    def apply(self, predictions: np.ndarray) -> np.ndarray:
        if not self.applied:
            return predictions
        return np.clip(self.intercept + self.slope * predictions, 0.0, None)

    def describe(self) -> str:
        if not self.applied:
            return f"{self.district}: pooled model used without local calibration"
        return (
            f"{self.district}: pooled model calibrated locally "
            f"({self.intercept:+.2f} + {self.slope:.2f}x on {self.n_rows} weeks of local data)"
        )


@dataclass
class TransferPlan:
    """Which districts fit locally, which borrow, and from whom."""

    local: List[str] = field(default_factory=list)
    borrowing: Dict[str, List[str]] = field(default_factory=dict)
    calibrations: Dict[str, LocalCalibration] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = []
        for district in self.local:
            rows.append({"district": district, "strategy": "local_fit", "donors": "", "calibrated": False})
        for district, donors in self.borrowing.items():
            calibration = self.calibrations.get(district)
            rows.append(
                {
                    "district": district,
                    "strategy": "transfer",
                    "donors": ", ".join(donors[:3]),
                    "calibrated": bool(calibration and calibration.applied),
                }
            )
        return pd.DataFrame(rows).sort_values(["strategy", "district"]).reset_index(drop=True)


def similarity_matrix(region: RegionConfig) -> pd.DataFrame:
    """District x district ecological similarity in [0, 1].

    Standardises the attributes, takes a Euclidean distance in that space, and
    maps it to a similarity. Geography contributes, but so do density, urbanity
    and WASH coverage — the things that actually change transmission.
    """
    frame = district_frame(region)
    available = [c for c in SIMILARITY_FEATURES if c in frame.columns]
    values = frame[available].astype(float)
    standardised = (values - values.mean()) / values.std().replace(0, 1.0)
    matrix = standardised.to_numpy()
    squared = ((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
    distances = np.sqrt(squared)
    scale = np.median(distances[distances > 0]) or 1.0
    similarity = np.exp(-distances / scale)
    np.fill_diagonal(similarity, 1.0)
    return pd.DataFrame(similarity, index=frame.index, columns=frame.index)


def find_donors(
    region: RegionConfig,
    district: str,
    candidates: Sequence[str],
    k: int = 5,
    min_similarity: float = 0.3,
) -> List[str]:
    """The `k` most ecologically similar data-rich districts."""
    if district not in region.district_names:
        return []
    similarity = similarity_matrix(region)[district]
    pool = [c for c in candidates if c != district and c in similarity.index]
    if not pool:
        return []
    ranked = similarity.loc[pool].sort_values(ascending=False)
    return list(ranked[ranked >= min_similarity].head(k).index) or list(ranked.head(k).index)


def build_transfer_plan(
    matrix,
    region: RegionConfig,
    min_rows: int = MIN_ROWS_FOR_LOCAL_FIT,
) -> TransferPlan:
    """Decide, per district, whether to fit locally or borrow."""
    plan = TransferPlan()
    counts: Dict[str, int] = {}
    for district in matrix.districts:
        counts[district] = len(matrix.for_district(district).dropna_rows().X)

    plan.local = [d for d, n in counts.items() if n >= min_rows]
    for district, n in counts.items():
        if n >= min_rows:
            continue
        donors = find_donors(region, district, plan.local or list(counts))
        plan.borrowing[district] = donors
        log.debug("%s has %d usable weeks; borrowing from %s", district, n, ", ".join(donors[:3]))
    return plan


def fit_local_calibration(
    predictions: np.ndarray,
    actuals: np.ndarray,
    district: str,
    min_rows: int = MIN_ROWS_FOR_CALIBRATION,
) -> LocalCalibration:
    """Fit `actual ~ a + b * predicted` on whatever local data exists.

    Two parameters instead of a hundred: this is what makes local adaptation
    feasible on a council with a year of patchy reporting.
    """
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    x, y = predictions[mask], actuals[mask]
    calibration = LocalCalibration(district=district, n_rows=int(mask.sum()))
    if calibration.n_rows < min_rows or np.std(x) < 1e-9:
        return calibration

    denominator = float(np.sum((x - x.mean()) ** 2))
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / denominator) if denominator > 0 else 1.0
    intercept = float(y.mean() - slope * x.mean())

    total = float(np.sum((y - y.mean()) ** 2)) or 1e-9
    calibration.r2_before = float(1 - np.sum((y - x) ** 2) / total)
    corrected = intercept + slope * x
    calibration.r2_after = float(1 - np.sum((y - corrected) ** 2) / total)

    # Only apply a calibration that actually helps, and reject wild slopes that
    # a dozen noisy points can easily produce.
    if calibration.r2_after > calibration.r2_before and 0.2 <= slope <= 5.0:
        calibration.intercept = intercept
        calibration.slope = slope
        calibration.applied = True
    return calibration


def transfer_lag_structure(
    lag_fits: Dict[str, Dict],
    region: RegionConfig,
    target_district: str,
    donors: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Borrow a lag structure from similar districts, weighted by similarity.

    Used when a district has enough data to calibrate but not enough for its own
    lag scan — the structure is transferred, the level is fitted locally.
    """
    donors = list(donors or find_donors(region, target_district, list(lag_fits)))
    if not donors:
        return {}
    similarity = similarity_matrix(region)[target_district]
    out: Dict[str, int] = {}
    proxies = {p for donor in donors for p in lag_fits.get(donor, {})}
    for proxy in proxies:
        lags, weights = [], []
        for donor in donors:
            fit = lag_fits.get(donor, {}).get(proxy)
            if fit is None:
                continue
            lags.append(fit.lag_weeks)
            weights.append(float(similarity.get(donor, 0.0)) * max(fit.score, 1e-3))
        if lags:
            out[proxy] = int(round(float(np.average(lags, weights=weights))))
    return out
