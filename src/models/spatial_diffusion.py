"""Mobility-weighted spatial diffusion (shortcoming #10).

Most published models answer "will malaria rise next month?". Agencies also
need "where will it arrive next?", because pre-positioning supplies in the
wrong district is as costly as not pre-positioning at all.

The model here is a discrete-time reaction-diffusion process on the district
graph: each week a share of a district's infection pressure travels along the
flow matrix, decays with the disease's serial interval, and adds to the
receiving district's local risk. With CDR data the flow matrix is empirical;
without it, gravity or radiation (rule #8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.geo import gravity_matrix, radiation_matrix
from src.core.logging import get_logger
from src.core.timeutils import shift_week
from src.core.types import DiseaseConfig, RegionConfig, SourceRisk

log = get_logger("models.diffusion")

#: Approximate serial interval per transmission mode, in weeks. Controls how
#: fast imported infection converts into local transmission.
SERIAL_INTERVAL_WEEKS: Dict[str, float] = {
    "vector_borne": 4.0,
    "waterborne": 1.0,
    "airborne": 1.5,
    "sexual": 12.0,
    "zoonotic": 2.0,
    "other": 2.0,
}


@dataclass
class DiffusionForecast:
    """Projected importation risk by district and week."""

    risk: pd.DataFrame            # weeks x districts, 0-1
    contributions: Dict[str, List[SourceRisk]]
    travel_model: str

    def top_districts(self, week: Optional[str] = None, n: int = 10) -> pd.Series:
        row = self.risk.loc[week] if week else self.risk.iloc[-1]
        return row.sort_values(ascending=False).head(n)


class SpatialDiffusionModel:
    """Project where infection pressure travels next."""

    def __init__(
        self,
        config: DiseaseConfig,
        region: RegionConfig,
        travel_matrix: Optional[pd.DataFrame] = None,
    ) -> None:
        self.config = config
        self.region = region
        self.travel_model = config.spatial.diffusion_model
        self.travel = travel_matrix if travel_matrix is not None else self._build_matrix()
        self.populations = pd.Series(
            {d.name: float(d.population) for d in region.districts}, name="population"
        )

    def _build_matrix(self) -> pd.DataFrame:
        from src.feature_engineering.mobility_features import get_travel_matrix

        return get_travel_matrix(self.region, model=self.travel_model)

    # -- core --------------------------------------------------------------
    @property
    def decay(self) -> float:
        """Weekly retention of imported pressure, from the serial interval."""
        interval = SERIAL_INTERVAL_WEEKS.get(self.config.transmission_mode.value, 2.0)
        return float(np.exp(-1.0 / max(interval, 0.5)))

    def step(self, incidence: pd.Series, carried: Optional[pd.Series] = None) -> pd.Series:
        """Advance the diffusion one week.

        `incidence` is current per-1,000 incidence by district; `carried` is
        importation pressure surviving from previous weeks.
        """
        districts = list(self.travel.index)
        current = incidence.reindex(districts).fillna(0.0).to_numpy()
        previous = (
            carried.reindex(districts).fillna(0.0).to_numpy()
            if carried is not None
            else np.zeros(len(districts))
        )
        # Flow-weighted incidence arriving from every origin, plus decayed carry-over.
        arriving = self.travel.to_numpy().T @ current
        return pd.Series(arriving + self.decay * previous, index=districts)

    def project(
        self,
        incidence: pd.Series,
        start_week: str,
        horizon_weeks: int = 8,
        local_growth: float = 1.0,
    ) -> DiffusionForecast:
        """Roll the diffusion forward and normalise to a 0-1 risk score."""
        districts = list(self.travel.index)
        current = incidence.reindex(districts).fillna(0.0)
        carried = pd.Series(0.0, index=districts)

        rows: Dict[str, pd.Series] = {}
        for step in range(1, horizon_weeks + 1):
            carried = self.step(current, carried)
            week = shift_week(start_week, step)
            reference = max(self.config.alerts.high, 1e-6)
            rows[week] = np.tanh(carried / reference) * self.config.spatial.importation_weight * 2.0
            # Imported pressure seeds local transmission for the next step.
            current = current * local_growth + carried * (1 - self.decay)

        risk = pd.DataFrame(rows).T.clip(0.0, 1.0)
        contributions = {
            district: self.contributors(incidence, district) for district in districts
        }
        return DiffusionForecast(
            risk=risk, contributions=contributions, travel_model=self.travel_model
        )

    def contributors(
        self, incidence: pd.Series, district: str, top_n: int = 5
    ) -> List[SourceRisk]:
        """Which districts are sending risk into `district`, and how much."""
        if district not in self.travel.columns:
            return []
        inflow = self.travel[district].drop(labels=[district], errors="ignore")
        aligned = incidence.reindex(inflow.index).fillna(0.0)
        contributions = (inflow * aligned).sort_values(ascending=False)
        total = float(contributions.sum())
        out: List[SourceRisk] = []
        for name, value in contributions.head(top_n).items():
            if value <= 0:
                continue
            out.append(
                SourceRisk(
                    district=str(name),
                    flow_weight=round(float(inflow[name]), 6),
                    active_cases=float(aligned[name] * self.populations.get(name, 0.0) / 1000.0),
                    contributed_risk=round(float(value / total) if total > 0 else 0.0, 4),
                )
            )
        return out

    # -- diagnostics -------------------------------------------------------
    def importation_ranking(self, incidence: pd.Series) -> pd.DataFrame:
        """One-step importation pressure for every district, ranked."""
        pressure = self.step(incidence)
        frame = pressure.rename("importation_pressure").to_frame()
        frame["local_incidence"] = incidence.reindex(frame.index).fillna(0.0)
        frame["imported_share"] = frame["importation_pressure"] / (
            frame["importation_pressure"] + frame["local_incidence"]
        ).replace(0, np.nan)
        return frame.sort_values("importation_pressure", ascending=False)

    def at_risk_districts(
        self, incidence: pd.Series, threshold_ratio: float = 0.6
    ) -> List[str]:
        """Districts whose risk is mostly *arriving* rather than growing locally.

        These are the ones a purely temporal model misses entirely: quiet today,
        seeded already.
        """
        ranking = self.importation_ranking(incidence)
        candidates = ranking[
            (ranking["imported_share"] >= threshold_ratio)
            & (ranking["importation_pressure"] > 0)
        ]
        return list(candidates.index)
