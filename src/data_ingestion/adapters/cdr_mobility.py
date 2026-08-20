"""Aggregated Call Detail Record mobility adapter (shortcoming #10).

Real CDR access requires a negotiated agreement with Vodacom/Airtel/Tigo and
arrives as **pre-aggregated** origin-destination counts — never subscriber-level
records. This adapter reads whatever aggregate drops land in `CDR_DATA_DIR` and,
when none exist, falls back to a gravity-model travel matrix (rule #8), so the
spatial layer never depends on a commercial negotiation.

Emitted variables
-----------------
`mobility_inbound`  : trips arriving in the district that week
`mobility_outbound` : trips leaving the district that week
`mobility_internal` : trips that start and end inside the district

The full origin-destination matrix is also exposed via :meth:`travel_matrix`
for the spatial diffusion model.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.geo import gravity_matrix, population_series, radiation_matrix
from src.data_ingestion.base_adapter import BaseAdapter

OD_COLUMNS = ["origin", "destination", "week", "trips"]


class CDRMobilityAdapter(BaseAdapter):
    source_name = "cdr_mobility"
    variables = ("mobility_inbound", "mobility_outbound", "mobility_internal")
    update_frequency_days = 7
    native_resolution = "council OD pairs"

    def is_configured(self) -> bool:
        directory = Path(self.settings.cdr_data_dir)
        if not directory.exists():
            return False
        return any(directory.glob("*.csv")) or any(directory.glob("*.parquet"))

    # -- live -------------------------------------------------------------
    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Read telco OD drops and reduce them to per-district flow totals."""
        od = self.read_od_files(weeks)
        if od.empty:
            raise FileNotFoundError("no usable CDR origin-destination files found")
        return self._od_to_tidy(od, weeks, quality=1.0)

    def read_od_files(self, weeks: Optional[List[str]] = None) -> pd.DataFrame:
        """Load and concatenate every OD file in the drop directory."""
        directory = Path(self.settings.cdr_data_dir)
        frames = []
        for path in sorted(list(directory.glob("*.csv")) + list(directory.glob("*.parquet"))):
            try:
                frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("unreadable CDR file %s: %s", path, exc)
                continue
            frame.columns = [c.strip().lower() for c in frame.columns]
            missing = [c for c in OD_COLUMNS if c not in frame.columns]
            if missing:
                self.log.warning("CDR file %s missing columns %s; skipped", path.name, missing)
                continue
            frames.append(frame[OD_COLUMNS])
        if not frames:
            return pd.DataFrame(columns=OD_COLUMNS)
        od = pd.concat(frames, ignore_index=True)
        if weeks is not None:
            od = od[od["week"].isin(set(weeks))]
        return od

    def _od_to_tidy(self, od: pd.DataFrame, weeks: List[str], quality: float) -> pd.DataFrame:
        known = set(self.region.district_names)
        od = od[od["origin"].isin(known) & od["destination"].isin(known)]
        internal = od[od["origin"] == od["destination"]]
        external = od[od["origin"] != od["destination"]]

        inbound = external.groupby(["destination", "week"])["trips"].sum()
        outbound = external.groupby(["origin", "week"])["trips"].sum()
        within = internal.groupby(["origin", "week"])["trips"].sum()

        records = []
        for (district, week), value in inbound.items():
            records.append({"district": district, "week": week, "variable": "mobility_inbound",
                            "value": float(value), "quality": quality})
        for (district, week), value in outbound.items():
            records.append({"district": district, "week": week, "variable": "mobility_outbound",
                            "value": float(value), "quality": quality})
        for (district, week), value in within.items():
            records.append({"district": district, "week": week, "variable": "mobility_internal",
                            "value": float(value), "quality": quality})
        return self.tidy(records)

    # -- travel matrix -----------------------------------------------------
    def travel_matrix(self, week: Optional[str] = None, model: str = "gravity") -> pd.DataFrame:
        """Row-normalised origin -> destination flow matrix for `week`.

        Uses empirical CDR data when available, otherwise the configured
        analytical fallback (rule #8).
        """
        if self.is_configured():
            od = self.read_od_files([week] if week else None)
            if not od.empty:
                matrix = od.pivot_table(
                    index="origin", columns="destination", values="trips", aggfunc="sum"
                )
                matrix = matrix.reindex(
                    index=self.region.district_names, columns=self.region.district_names
                ).fillna(0.0)
                # Zero the diagonal without writing through `.values`: pandas
                # hands back a read-only view under copy-on-write, and even
                # before that a mixed-dtype frame returned a *copy*, so the
                # in-place fill either raised or silently did nothing.
                flows = matrix.to_numpy(dtype=float, copy=True)
                np.fill_diagonal(flows, 0.0)
                matrix = pd.DataFrame(flows, index=matrix.index, columns=matrix.columns)
                totals = matrix.sum(axis=1).replace(0, 1.0)
                return matrix.div(totals, axis=0)
        if model == "radiation":
            return radiation_matrix(self.region)
        return gravity_matrix(self.region)

    # -- synthetic --------------------------------------------------------
    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Gravity-model flows modulated by holiday/harvest travel peaks."""
        matrix = gravity_matrix(self.region)
        pops = population_series(self.region)
        # Baseline: ~6% of a district's population makes an inter-district trip
        # in a given week; urban hubs both send and receive more.
        base_out = pops * 0.06

        records = []
        for week in weeks:
            phase = self._season_phase(week)
            # December holidays and the June-August harvest raise travel.
            seasonal = 1.0 + 0.25 * np.cos(phase - np.radians(350)) + 0.12 * np.cos(2 * phase)
            flows = matrix.mul(base_out * seasonal, axis=0)
            inbound = flows.sum(axis=0)
            outbound = flows.sum(axis=1)
            for district in self.region.districts:
                rng = self._rng(district.name, week)
                jitter = float(np.exp(rng.normal(0, 0.06)))
                internal = float(district.population) * (0.35 if district.urban else 0.18) * jitter
                for variable, value in (
                    ("mobility_inbound", float(inbound[district.name]) * jitter),
                    ("mobility_outbound", float(outbound[district.name]) * jitter),
                    ("mobility_internal", internal),
                ):
                    records.append(
                        {
                            "district": district.name,
                            "week": week,
                            "variable": variable,
                            "value": round(max(value, 0.0), 1),
                            "quality": 0.55,  # gravity fallback, not measurement
                        }
                    )
        return self.tidy(records)
