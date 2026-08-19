"""CHIRPS precipitation adapter.

CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) is a
free, global, 0.05-degree (~5 km) rainfall product published pentadally by UCSB.
It is the platform's primary rainfall signal: 100% spatial coverage, no
dependence on Tanzania's ~50 automatic weather stations (shortcoming #7, #14).

Live path: download the pentadal/dekadal GeoTIFF for the requested window and
zonally average it over each council polygon. That needs `rasterio` + a boundary
GeoJSON, both optional extras — without them the adapter degrades to cache and
then to a bimodal/unimodal synthetic climatology keyed to district latitude.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.data_ingestion.base_adapter import BaseAdapter
from src.core.timeutils import epi_week_start


class CHIRPSRainfallAdapter(BaseAdapter):
    source_name = "chirps"
    variables = ("rainfall_mm",)
    update_frequency_days = 5
    native_resolution = "0.05deg (~5km)"

    def is_configured(self) -> bool:
        """CHIRPS is an open archive; a live pull only needs the geospatial extras."""
        try:
            import rasterio  # noqa: F401
        except ImportError:
            return False
        geojson = self.region.geojson_path
        return bool(geojson) and (self.settings.data_dir.parent / geojson).exists()

    # -- live -------------------------------------------------------------
    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Download CHIRPS rasters and zonally average per council."""
        import rasterio  # noqa: F401
        from rasterio.mask import mask  # noqa: F401

        from src.data_ingestion.raster_utils import (
            download_raster,
            zonal_means,
            load_boundaries,
        )

        boundaries = load_boundaries(self.region)
        records = []
        for week in weeks:
            start = epi_week_start(week)
            url = (
                f"{self.settings.chirps_base_url}/global_dekad/tifs/"
                f"chirps-v2.0.{start.year}.{start.month:02d}."
                f"{min(3, (start.day - 1) // 10 + 1)}.tif"
            )
            path = download_raster(url, self.settings.raw_dir / "chirps")
            means = zonal_means(path, boundaries)
            for district, value in means.items():
                # dekadal totals -> weekly equivalent
                records.append(
                    {
                        "district": district,
                        "week": week,
                        "variable": "rainfall_mm",
                        "value": float(value) * 7.0 / 10.0,
                        "quality": 1.0,
                    }
                )
        return self.tidy(records)

    # -- synthetic --------------------------------------------------------
    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Bimodal (north) / unimodal (south) Tanzanian rainfall climatology.

        Northern districts get the `vuli` (Oct-Dec) and `masika` (Mar-May)
        seasons; southern districts get a single Dec-Apr wet season. Inter-annual
        variability is a deterministic draw so backtests are reproducible.
        """
        records = []
        for district in self.region.districts:
            rng = self._rng(district.name)
            # Latitude drives the seasonal regime; -6 deg S is the rough divide.
            bimodal_weight = float(np.clip((district.lat + 9.0) / 8.0, 0.0, 1.0))
            base = 55.0 + 25.0 * np.cos(np.radians(district.lat * 3))
            anomaly_walk = np.cumsum(rng.normal(0, 0.14, size=len(weeks)))
            for i, week in enumerate(weeks):
                phase = self._season_phase(week)
                unimodal = np.exp(2.1 * np.cos(phase - np.radians(20)))
                vuli = np.exp(2.4 * np.cos(phase - np.radians(310)))
                masika = np.exp(2.6 * np.cos(phase - np.radians(95)))
                seasonal = (
                    bimodal_weight * 0.5 * (vuli + masika) + (1 - bimodal_weight) * unimodal
                ) / np.exp(2.3)
                enso = 0.22 * np.sin(2 * np.pi * i / (52 * 3.7))  # ~3.7y ENSO cycle
                value = base * seasonal * (1 + enso + 0.30 * np.tanh(anomaly_walk[i]))
                value = max(0.0, value + rng.normal(0, 3.0))
                records.append(
                    {
                        "district": district.name,
                        "week": week,
                        "variable": "rainfall_mm",
                        "value": round(float(value), 2),
                        "quality": 0.7,  # synthetic data is never full confidence
                    }
                )
        return self.tidy(records)
