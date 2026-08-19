"""MODIS NDVI adapter — vegetation greenness.

NDVI is a free 250 m–1 km, 16-day composite from NASA LPDAAC (AppEEARS). It
proxies mosquito resting habitat and soil moisture for malaria, standing surface
water for cholera, and agricultural stress for migration-driven risk.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.data_ingestion.base_adapter import BaseAdapter


class MODISNDVIAdapter(BaseAdapter):
    source_name = "modis"
    variables = ("ndvi",)
    update_frequency_days = 16
    native_resolution = "250m-1km"

    APPEEARS_URL = "https://appeears.earthdatacloud.nasa.gov/api"

    def is_configured(self) -> bool:
        return self.settings.has_earthdata()

    # -- live -------------------------------------------------------------
    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Submit a point sample task to AppEEARS and poll for the result."""
        import time

        import requests

        from src.core.timeutils import epi_week_end, epi_week_start

        auth = requests.post(
            f"{self.APPEEARS_URL}/login",
            auth=(self.settings.earthdata_username, self.settings.earthdata_password),
            timeout=60,
        )
        auth.raise_for_status()
        token = auth.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        start, end = epi_week_start(weeks[0]), epi_week_end(weeks[-1])
        task = {
            "task_type": "point",
            "task_name": f"afya-ndvi-{start:%Y%m%d}",
            "params": {
                "dates": [{"startDate": f"{start:%m-%d-%Y}", "endDate": f"{end:%m-%d-%Y}"}],
                "layers": [{"product": "MOD13Q1.061", "layer": "_250m_16_days_NDVI"}],
                "coordinates": [
                    {"latitude": d.lat, "longitude": d.lon, "id": d.name}
                    for d in self.region.districts
                ],
            },
        }
        submit = requests.post(f"{self.APPEEARS_URL}/task", json=task, headers=headers, timeout=120)
        submit.raise_for_status()
        task_id = submit.json()["task_id"]

        for _ in range(60):  # poll for up to ~10 minutes
            status = requests.get(
                f"{self.APPEEARS_URL}/task/{task_id}", headers=headers, timeout=60
            ).json()
            if status.get("status") == "done":
                break
            time.sleep(10)
        else:
            raise TimeoutError(f"AppEEARS task {task_id} did not finish in time")

        bundle = requests.get(
            f"{self.APPEEARS_URL}/bundle/{task_id}", headers=headers, timeout=60
        ).json()
        csv_file = next(f for f in bundle["files"] if f["file_name"].endswith(".csv"))
        content = requests.get(
            f"{self.APPEEARS_URL}/bundle/{task_id}/{csv_file['file_id']}",
            headers=headers,
            timeout=300,
        )
        content.raise_for_status()

        from io import StringIO

        frame = pd.read_csv(StringIO(content.text))
        frame["date"] = pd.to_datetime(frame["Date"])
        frame["week"] = frame["date"].dt.strftime("%G-W%V")
        value_col = next(c for c in frame.columns if "NDVI" in c and "QC" not in c)
        weekly = frame.groupby(["ID", "week"], as_index=False)[value_col].mean()
        records = [
            {
                "district": str(row["ID"]),
                "week": str(row["week"]),
                "variable": "ndvi",
                "value": float(row[value_col]) * 0.0001,  # MODIS scale factor
                "quality": 1.0,
            }
            for _, row in weekly.iterrows()
            if str(row["week"]) in set(weeks)
        ]
        return self.tidy(records)

    # -- synthetic --------------------------------------------------------
    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Greenness lagging rainfall by ~3 weeks, damped in urban districts."""
        from src.core.timeutils import shift_week, week_range
        from src.data_ingestion.adapters.chirps_rainfall import CHIRPSRainfallAdapter

        lag = 3
        extended = week_range(shift_week(weeks[0], -(lag + 8)), weeks[-1])
        rain = CHIRPSRainfallAdapter(self.region, self.settings).synthesize(extended)
        wide = rain.pivot_table(index="week", columns="district", values="value").reindex(extended)
        # NDVI responds to accumulated moisture, not a single week's rain.
        accumulated = wide.rolling(window=8, min_periods=1).mean().shift(lag)

        records = []
        for district in self.region.districts:
            rng = self._rng(district.name)
            urban_damping = 0.72 if district.urban else 1.0
            baseline = 0.30 + 0.22 * urban_damping
            series = accumulated[district.name]
            scale = max(float(series.max() or 1.0), 1e-6)
            for week in weeks:
                moisture = float(series.get(week, np.nan))
                if not np.isfinite(moisture):
                    moisture = float(series.mean())
                ndvi = baseline + 0.34 * urban_damping * (moisture / scale) + rng.normal(0, 0.012)
                records.append(
                    {
                        "district": district.name,
                        "week": week,
                        "variable": "ndvi",
                        "value": round(float(np.clip(ndvi, 0.02, 0.95)), 4),
                        "quality": 0.7,
                    }
                )
        return self.tidy(records)
