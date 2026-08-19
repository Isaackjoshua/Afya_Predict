"""ERA5 reanalysis adapter — temperature, humidity and wind.

ERA5 is ECMWF's global 0.25-degree reanalysis, free through the Copernicus
Climate Data Store. Because reanalysis has no gaps, it anchors the temperature
and humidity signals that drive vector development (malaria), pathogen growth
(cholera) and droplet survival (TB/ARI).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.core.timeutils import epi_week_start, epi_week_end
from src.data_ingestion.base_adapter import BaseAdapter


class ERA5ClimateAdapter(BaseAdapter):
    source_name = "era5"
    variables = ("temperature_c", "humidity_pct", "wind_speed_ms")
    update_frequency_days = 7
    native_resolution = "0.25deg (~28km)"

    _CDS_VARIABLES = {
        "temperature_c": "2m_temperature",
        "humidity_pct": "2m_dewpoint_temperature",
        "wind_speed_ms": "10m_u_component_of_wind",
    }

    def is_configured(self) -> bool:
        try:
            import cdsapi  # noqa: F401
        except ImportError:
            return False
        return self.settings.has_cds()

    # -- live -------------------------------------------------------------
    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Request daily aggregates from the CDS API and reduce to weekly means."""
        import cdsapi
        import xarray as xr

        client = cdsapi.Client(url=self.settings.cds_api_url, key=self.settings.cds_api_key)
        bbox = self.region.bbox
        area = [bbox.north, bbox.west, bbox.south, bbox.east] if bbox else None

        target_dir = self.settings.raw_dir / "era5"
        target_dir.mkdir(parents=True, exist_ok=True)
        start, end = epi_week_start(weeks[0]), epi_week_end(weeks[-1])
        target = target_dir / f"era5_{start:%Y%m%d}_{end:%Y%m%d}.nc"

        if not target.exists():
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": list(self._CDS_VARIABLES.values())
                    + ["10m_v_component_of_wind"],
                    "date": f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    **({"area": area} if area else {}),
                },
                str(target),
            )

        dataset = xr.open_dataset(target)
        records = []
        for district in self.region.districts:
            point = dataset.sel(latitude=district.lat, longitude=district.lon, method="nearest")
            frame = point.to_dataframe().reset_index()
            frame["week"] = frame["time"].dt.date.map(
                lambda d: f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"
            )
            weekly = frame.groupby("week").mean(numeric_only=True)
            for week, row in weekly.iterrows():
                if week not in weeks:
                    continue
                temp_c = float(row["t2m"]) - 273.15
                dew_c = float(row["d2m"]) - 273.15
                humidity = _relative_humidity(temp_c, dew_c)
                wind = float(np.hypot(row.get("u10", 0.0), row.get("v10", 0.0)))
                for variable, value in (
                    ("temperature_c", temp_c),
                    ("humidity_pct", humidity),
                    ("wind_speed_ms", wind),
                ):
                    records.append(
                        {
                            "district": district.name,
                            "week": week,
                            "variable": variable,
                            "value": round(value, 3),
                            "quality": 1.0,
                        }
                    )
        return self.tidy(records)

    # -- synthetic --------------------------------------------------------
    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Altitude/latitude-aware temperature and humidity climatology."""
        records = []
        for district in self.region.districts:
            rng = self._rng(district.name)
            # Proxy elevation from the highland regions; keeps Moshi/Njombe cool.
            highland = district.region in {
                "Kilimanjaro", "Arusha", "Njombe", "Iringa", "Mbeya", "Manyara", "Songwe",
            }
            base_temp = 26.5 - (6.0 if highland else 0.0) + 0.35 * (district.lat + 6.0)
            base_humidity = 78.0 if abs(district.lon - 39.0) < 2.0 else 62.0  # coastal boost
            walk = np.cumsum(rng.normal(0, 0.08, size=len(weeks)))
            for i, week in enumerate(weeks):
                phase = self._season_phase(week)
                # Southern-hemisphere seasonality: warm Dec-Feb, cool Jun-Aug.
                temp = base_temp + 3.1 * np.cos(phase - np.radians(15)) + 0.6 * np.tanh(walk[i])
                temp += rng.normal(0, 0.45)
                humidity = base_humidity + 11.0 * np.cos(phase - np.radians(60)) + rng.normal(0, 2.2)
                wind = 3.2 + 1.4 * np.cos(phase - np.radians(200)) + rng.normal(0, 0.4)
                for variable, value, lo, hi in (
                    ("temperature_c", temp, 5.0, 42.0),
                    ("humidity_pct", humidity, 10.0, 100.0),
                    ("wind_speed_ms", wind, 0.0, 25.0),
                ):
                    records.append(
                        {
                            "district": district.name,
                            "week": week,
                            "variable": variable,
                            "value": round(float(np.clip(value, lo, hi)), 3),
                            "quality": 0.7,
                        }
                    )
        return self.tidy(records)


def _relative_humidity(temp_c: float, dewpoint_c: float) -> float:
    """Magnus-formula relative humidity (%) from temperature and dew point."""
    a, b = 17.625, 243.04
    numerator = np.exp(a * dewpoint_c / (b + dewpoint_c))
    denominator = np.exp(a * temp_c / (b + temp_c))
    return float(np.clip(100.0 * numerator / denominator, 0.0, 100.0))
