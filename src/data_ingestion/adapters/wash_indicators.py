"""WASH (water, sanitation and hygiene) indicator adapter.

Sources: the WHO/UNICEF Joint Monitoring Programme (JMP) national/subnational
series, augmented by satellite-derived surface-water extent. WASH access is the
dominant structural driver of cholera risk and a deprivation proxy for TB/HIV.

Values are annual, so the adapter emits a smoothly interpolated weekly series
rather than a step function — a step would create a spurious January signal.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.core.timeutils import week_of_year, year_of_week
from src.data_ingestion.base_adapter import BaseAdapter

#: Roughly +0.8 percentage points of improved-water access per year nationally.
ANNUAL_IMPROVEMENT = 0.008
BASE_YEAR = 2022


class WASHIndicatorsAdapter(BaseAdapter):
    source_name = "wash_indicators"
    variables = ("wash_access", "improved_sanitation", "surface_water_index")
    update_frequency_days = 365
    native_resolution = "council (JMP subnational)"

    def is_configured(self) -> bool:
        return (self.settings.raw_dir / "wash" / "jmp_subnational.csv").exists()

    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Read a JMP subnational extract (`district,year,water,sanitation`)."""
        path = self.settings.raw_dir / "wash" / "jmp_subnational.csv"
        frame = pd.read_csv(path)
        frame.columns = [c.strip().lower() for c in frame.columns]
        required = {"district", "year", "water", "sanitation"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        indexed = frame.set_index(["district", "year"])
        records = []
        for week in weeks:
            year = year_of_week(week)
            for district in self.region.districts:
                key = (district.name, year)
                if key not in indexed.index:
                    continue
                row = indexed.loc[key]
                records.append({"district": district.name, "week": week, "variable": "wash_access",
                                "value": float(row["water"]), "quality": 1.0})
                records.append({"district": district.name, "week": week,
                                "variable": "improved_sanitation",
                                "value": float(row["sanitation"]), "quality": 1.0})
        return self.tidy(records)

    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Interpolate the config table's access levels with a seasonal water index."""
        records = []
        for district in self.region.districts:
            rng = self._rng(district.name)
            local_drift = ANNUAL_IMPROVEMENT * float(np.clip(rng.normal(1.0, 0.35), 0.2, 2.0))
            for week in weeks:
                years = (year_of_week(week) - BASE_YEAR) + (week_of_year(week) - 1) / 52.0
                access = float(np.clip(district.wash_access + local_drift * years, 0.05, 0.99))
                sanitation = float(np.clip(access - 0.16, 0.02, 0.97))
                # Surface water peaks with the rains: more open sources in use.
                phase = self._season_phase(week)
                surface = 0.5 + 0.35 * np.cos(phase - np.radians(60))
                for variable, value in (
                    ("wash_access", access),
                    ("improved_sanitation", sanitation),
                    ("surface_water_index", float(np.clip(surface, 0.0, 1.0))),
                ):
                    records.append(
                        {
                            "district": district.name,
                            "week": week,
                            "variable": variable,
                            "value": round(value, 4),
                            "quality": 0.75,
                        }
                    )
        return self.tidy(records)
