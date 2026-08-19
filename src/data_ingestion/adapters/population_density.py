"""WorldPop / census population density adapter.

Density is slow-moving: the adapter emits the same value for every week in the
range, interpolated across years by the national growth rate. It is a
contemporaneous (lag 0) proxy for every disease that scales with contact rate.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.core.timeutils import year_of_week
from src.data_ingestion.base_adapter import BaseAdapter

#: Tanzania's approximate annual population growth rate.
ANNUAL_GROWTH = 0.031
#: Census year the config table is anchored to.
BASE_YEAR = 2022


class PopulationDensityAdapter(BaseAdapter):
    source_name = "population_density"
    variables = ("population", "population_density_km2", "urban_share")
    update_frequency_days = 365
    native_resolution = "100m raster / council table"

    def is_configured(self) -> bool:
        """A WorldPop raster plus boundaries beats the config table when present."""
        try:
            import rasterio  # noqa: F401
        except ImportError:
            return False
        return bool(self.region.geojson_path)

    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Zonal-sum a WorldPop 100 m raster over the council boundaries."""
        from src.data_ingestion.raster_utils import load_boundaries, zonal_means

        raster = self.settings.raw_dir / "worldpop" / f"tza_ppp_{BASE_YEAR}.tif"
        if not raster.exists():
            raise FileNotFoundError(
                f"WorldPop raster not found at {raster}; download it or rely on the config table"
            )
        boundaries = load_boundaries(self.region)
        means = zonal_means(raster, boundaries)
        records = []
        for week in weeks:
            factor = (1 + ANNUAL_GROWTH) ** (year_of_week(week) - BASE_YEAR)
            for district_name, density in means.items():
                records.append(
                    {"district": district_name, "week": week, "variable": "population_density_km2",
                     "value": float(density) * factor, "quality": 1.0}
                )
        return self.tidy(records)

    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Project the census table forward/backward at the national growth rate."""
        records = []
        for district in self.region.districts:
            for week in weeks:
                factor = (1 + ANNUAL_GROWTH) ** (year_of_week(week) - BASE_YEAR)
                for variable, value in (
                    ("population", district.population * factor),
                    ("population_density_km2", district.density_km2 * factor),
                    ("urban_share", 1.0 if district.urban else 0.0),
                ):
                    records.append(
                        {
                            "district": district.name,
                            "week": week,
                            "variable": variable,
                            "value": round(float(value), 2),
                            # Census-anchored, so higher confidence than climatology.
                            "quality": 0.9,
                        }
                    )
        return self.tidy(records)
