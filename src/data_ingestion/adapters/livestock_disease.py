"""Livestock and wildlife disease adapter — the One Health arm (shortcoming #5).

Human, animal and environmental surveillance normally live in separate systems;
the 2014 Ebola response was delayed by exactly that incompatibility. This
adapter puts animal signals on the *same* district x week grid as human cases so
that cross-domain anomaly detection can catch spillover before it reaches
facilities.

Upstream candidates: Tanzania's Ministry of Livestock and Fisheries reports,
WOAH/WAHIS event notifications, and district veterinary officer submissions.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.data_ingestion.base_adapter import BaseAdapter

WAHIS_URL = "https://wahis.woah.org/api/v1/pi/event/filtered-list"


class LivestockDiseaseAdapter(BaseAdapter):
    source_name = "livestock_disease"
    variables = (
        "livestock_outbreak_events",
        "livestock_mortality",
        "zoonotic_alert_index",
        "livestock_density",
    )
    update_frequency_days = 7
    native_resolution = "council (veterinary reports)"

    def is_configured(self) -> bool:
        return (self.settings.raw_dir / "livestock" / "reports.csv").exists()

    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Read veterinary-office submissions (`district,week,disease,cases,deaths`)."""
        path = self.settings.raw_dir / "livestock" / "reports.csv"
        frame = pd.read_csv(path)
        frame.columns = [c.strip().lower() for c in frame.columns]
        required = {"district", "week", "cases", "deaths"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        frame = frame[frame["week"].isin(set(weeks))]
        records = []
        for _, row in frame.iterrows():
            records.append({"district": str(row["district"]), "week": str(row["week"]),
                            "variable": "livestock_outbreak_events",
                            "value": float(row["cases"]), "quality": 0.9})
            records.append({"district": str(row["district"]), "week": str(row["week"]),
                            "variable": "livestock_mortality",
                            "value": float(row["deaths"]), "quality": 0.9})
        return self.tidy(records)

    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Rare, clustered animal events concentrated in pastoralist districts."""
        pastoral = {"Monduli", "Longido", "Ngorongoro", "Simanjiro", "Kiteto", "Chunya",
                    "Manyoni", "Meatu", "Kilindi", "Handeni"}
        records = []
        for district in self.region.districts:
            rng = self._rng(district.name)
            intensity = 1.0 if district.name in pastoral else (0.15 if district.urban else 0.45)
            herd_density = intensity * float(np.clip(rng.normal(45, 15), 2, 120))
            for week in weeks:
                phase = self._season_phase(week)
                # Rift Valley fever style risk rises after heavy rains.
                seasonal = 0.6 + 0.9 * max(0.0, np.cos(phase - np.radians(80)))
                events = float(rng.poisson(0.06 * intensity * seasonal))
                mortality = events * float(np.clip(rng.normal(7, 4), 0, 60))
                zoonotic = float(np.clip(0.05 * events + 0.02 * seasonal * intensity, 0, 1))
                for variable, value in (
                    ("livestock_outbreak_events", events),
                    ("livestock_mortality", mortality),
                    ("zoonotic_alert_index", zoonotic),
                    ("livestock_density", herd_density),
                ):
                    records.append(
                        {"district": district.name, "week": week, "variable": variable,
                         "value": round(float(value), 3), "quality": 0.6}
                    )
        return self.tidy(records)
