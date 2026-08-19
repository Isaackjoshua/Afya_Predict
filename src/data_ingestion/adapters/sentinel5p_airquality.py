"""Sentinel-5P TROPOMI adapter — NO2 and aerosol-derived PM2.5 proxy.

Daily 7 km global coverage from the Copernicus Data Space Ecosystem. Drives the
respiratory module directly and the TB module through a long susceptibility lag.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.data_ingestion.base_adapter import BaseAdapter


class Sentinel5PAirQualityAdapter(BaseAdapter):
    source_name = "sentinel5p"
    variables = ("no2_mol_m2", "pm25_ug_m3", "aerosol_index")
    update_frequency_days = 1
    native_resolution = "7km"

    TOKEN_URL = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )
    STATISTICS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

    def is_configured(self) -> bool:
        return self.settings.has_cdse()

    # -- live -------------------------------------------------------------
    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Query the Sentinel Hub statistical API for weekly district means."""
        import requests

        from src.core.timeutils import epi_week_end, epi_week_start

        token_response = requests.post(
            self.TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.settings.cdse_client_id,
                "client_secret": self.settings.cdse_client_secret,
            },
            timeout=60,
        )
        token_response.raise_for_status()
        headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}

        evalscript = (
            "//VERSION=3\n"
            "function setup(){return{input:['NO2','dataMask'],"
            "output:[{id:'default',bands:1},{id:'dataMask',bands:1}]}}\n"
            "function evaluatePixel(s){return{default:[s.NO2],dataMask:[s.dataMask]}}"
        )
        records = []
        for district in self.region.districts:
            # ~0.1 degree box around the district centroid.
            box = [district.lon - 0.1, district.lat - 0.1, district.lon + 0.1, district.lat + 0.1]
            payload = {
                "input": {
                    "bounds": {"bbox": box, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                    "data": [{"type": "sentinel-5p-l2", "dataFilter": {}}],
                },
                "aggregation": {
                    "timeRange": {
                        "from": f"{epi_week_start(weeks[0])}T00:00:00Z",
                        "to": f"{epi_week_end(weeks[-1])}T23:59:59Z",
                    },
                    "aggregationInterval": {"of": "P7D"},
                    "evalscript": evalscript,
                    "resx": 0.05,
                    "resy": 0.05,
                },
            }
            response = requests.post(
                self.STATISTICS_URL, json=payload, headers=headers, timeout=180
            )
            response.raise_for_status()
            for interval in response.json().get("data", []):
                stamp = pd.Timestamp(interval["interval"]["from"])
                week = stamp.strftime("%G-W%V")
                if week not in set(weeks):
                    continue
                stats = interval["outputs"]["default"]["bands"]["B0"]["stats"]
                no2 = float(stats.get("mean", np.nan))
                if not np.isfinite(no2):
                    continue
                records.append(
                    {"district": district.name, "week": week, "variable": "no2_mol_m2",
                     "value": no2, "quality": 1.0}
                )
                records.append(
                    {"district": district.name, "week": week, "variable": "pm25_ug_m3",
                     "value": _no2_to_pm25(no2), "quality": 0.6}  # empirical proxy, not a measurement
                )
        return self.tidy(records)

    # -- synthetic --------------------------------------------------------
    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Urban-weighted pollution with a dry-season biomass-burning peak."""
        records = []
        for district in self.region.districts:
            rng = self._rng(district.name)
            urban_load = np.log1p(district.density_km2) / np.log1p(6000.0)
            base_no2 = 2.0e-5 + 9.0e-5 * urban_load
            base_pm25 = 12.0 + 34.0 * urban_load
            walk = np.cumsum(rng.normal(0, 0.07, size=len(weeks)))
            for i, week in enumerate(weeks):
                phase = self._season_phase(week)
                # Dry season (Jun-Oct) burning and dust raise particulates.
                burning = np.exp(1.1 * np.cos(phase - np.radians(230)))
                seasonal = burning / np.exp(1.1)
                no2 = base_no2 * (0.75 + 0.5 * seasonal) * (1 + 0.2 * np.tanh(walk[i]))
                pm25 = base_pm25 * (0.7 + 0.75 * seasonal) * (1 + 0.22 * np.tanh(walk[i]))
                aerosol = 0.3 + 1.5 * seasonal + rng.normal(0, 0.08)
                for variable, value, lo, hi in (
                    ("no2_mol_m2", no2 + rng.normal(0, 2e-6), 0.0, 1e-3),
                    ("pm25_ug_m3", pm25 + rng.normal(0, 2.0), 1.0, 300.0),
                    ("aerosol_index", aerosol, -1.0, 6.0),
                ):
                    records.append(
                        {
                            "district": district.name,
                            "week": week,
                            "variable": variable,
                            "value": float(np.clip(value, lo, hi)),
                            "quality": 0.65,
                        }
                    )
        return self.tidy(records)


def _no2_to_pm25(no2_mol_m2: float) -> float:
    """Crude NO2 -> PM2.5 surrogate; flagged at lower quality on purpose."""
    return float(np.clip(8.0 + 4.2e5 * no2_mol_m2, 1.0, 300.0))
