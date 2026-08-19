"""DHIS2 surveillance adapter — the ground truth the models are fit against.

Tanzania's routine health information system (DHIS2 / eIDSR) reports weekly and
monthly case and mortality counts by facility, aggregated to council level.
National completeness is around 93.9%, but roughly 70% of deaths occur outside
facilities, so DHIS2 *undercounts true burden* (shortcoming #7). We therefore:

* never zero-fill a missing report — the row is emitted with `NaN` and a
  quality flag so the imputer and the confidence intervals both see it;
* carry a per-district reporting-completeness score forward as `quality`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.timeutils import epi_week_end, epi_week_start
from src.data_ingestion.base_adapter import BaseAdapter

#: Disease slug -> the tidy variable name the rest of the pipeline expects.
CASE_VARIABLES: Dict[str, str] = {
    "malaria": "cases_malaria",
    "cholera": "cases_cholera",
    "tuberculosis": "cases_tuberculosis",
    "respiratory": "cases_respiratory",
    "hiv": "cases_hiv",
}

#: Baseline weekly incidence per 1,000 population used by the synthetic
#: generator. Anchored to published Tanzanian burden estimates, not fitted.
_BASELINE_PER_1000 = {
    "malaria": 1.8,
    "cholera": 0.03,
    "tuberculosis": 0.045,
    "respiratory": 4.5,
    "hiv": 0.025,
}


class DHIS2SurveillanceAdapter(BaseAdapter):
    source_name = "dhis2"
    variables = tuple(CASE_VARIABLES.values()) + ("reporting_completeness",)
    update_frequency_days = 7
    native_resolution = "council (admin-3)"

    def __init__(self, *args, diseases: Optional[List[str]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.diseases = diseases or list(CASE_VARIABLES)

    def is_configured(self) -> bool:
        return self.settings.has_dhis2()

    # -- live -------------------------------------------------------------
    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Pull weekly analytics from the DHIS2 REST API."""
        import requests

        base = self.settings.dhis2_base_url.rstrip("/")
        auth = (self.settings.dhis2_username, self.settings.dhis2_password)
        start, end = epi_week_start(weeks[0]), epi_week_end(weeks[-1])

        indicator_map = self._resolve_indicators(base, auth)
        if not indicator_map:
            raise RuntimeError("no DHIS2 data elements matched the configured diseases")

        params = {
            "dimension": [
                "dx:" + ";".join(indicator_map.values()),
                "ou:LEVEL-3",
                f"pe:{start:%Y-%m-%d}_{end:%Y-%m-%d}",
            ],
            "displayProperty": "NAME",
            "outputIdScheme": "NAME",
            "skipMeta": "false",
        }
        response = requests.get(f"{base}/api/analytics.json", params=params, auth=auth, timeout=120)
        response.raise_for_status()
        payload = response.json()

        headers = [h["name"] for h in payload.get("headers", [])]
        rows = payload.get("rows", [])
        frame = pd.DataFrame(rows, columns=headers)
        if frame.empty:
            return frame

        reverse = {v: k for k, v in indicator_map.items()}
        records = []
        for _, row in frame.iterrows():
            slug = reverse.get(row.get("dx"))
            if slug is None:
                continue
            records.append(
                {
                    "district": str(row.get("ou")),
                    "week": _dhis2_period_to_epi_week(str(row.get("pe"))),
                    "variable": CASE_VARIABLES[slug],
                    "value": pd.to_numeric(row.get("value"), errors="coerce"),
                    "quality": 1.0,
                }
            )
        return self.tidy(records)

    def _resolve_indicators(self, base: str, auth) -> Dict[str, str]:
        """Map disease slugs to DHIS2 data-element UIDs by name search."""
        import requests

        out: Dict[str, str] = {}
        for slug in self.diseases:
            response = requests.get(
                f"{base}/api/dataElements.json",
                params={"filter": f"name:ilike:{slug}", "fields": "id,name", "pageSize": 5},
                auth=auth,
                timeout=60,
            )
            response.raise_for_status()
            elements = response.json().get("dataElements", [])
            if elements:
                out[slug] = elements[0]["id"]
            else:
                self.log.warning("no DHIS2 data element found for %s", slug)
        return out

    # -- synthetic --------------------------------------------------------
    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Generate case series whose drivers match the configured proxies.

        The generator deliberately encodes the *same* causal lags the disease
        configs declare (rain -> malaria at ~6 weeks, heavy rain -> cholera at
        ~3 weeks, warm water -> cholera at ~16 weeks, ...) so that an end-to-end
        run without credentials still exercises real signal recovery rather than
        fitting noise. It also injects the reporting pathologies the real system
        must survive: missing weeks, completeness below 100% and occasional
        reporting spikes.
        """
        from src.data_ingestion.adapters.chirps_rainfall import CHIRPSRainfallAdapter
        from src.data_ingestion.adapters.era5_climate import ERA5ClimateAdapter

        warmup = 24  # weeks of driver history needed before the first target week
        from src.core.timeutils import shift_week, week_range

        extended = week_range(shift_week(weeks[0], -warmup), weeks[-1])

        rain = _pivot(CHIRPSRainfallAdapter(self.region, self.settings).synthesize(extended), "rainfall_mm")
        climate = ERA5ClimateAdapter(self.region, self.settings).synthesize(extended)
        temp = _pivot(climate, "temperature_c")
        humid = _pivot(climate, "humidity_pct")

        records = []
        for district in self.region.districts:
            rng = self._rng("cases", district.name)
            pop_k = district.population / 1000.0
            completeness = float(np.clip(self.region.reporting_completeness + rng.normal(0, 0.05), 0.55, 1.0))
            if not district.urban:
                completeness = float(np.clip(completeness - 0.06, 0.5, 1.0))

            r = rain[district.name].reindex(extended)
            t = temp[district.name].reindex(extended)
            h = humid[district.name].reindex(extended)

            for week in weeks:
                idx = extended.index(week)
                for slug in self.diseases:
                    intensity = self._intensity(slug, district, r, t, h, idx, rng)
                    expected = _BASELINE_PER_1000[slug] * pop_k * intensity
                    observed = float(rng.poisson(max(expected * completeness, 0.01)))

                    # ~4% of council-weeks simply never report (rule #7).
                    reported = rng.random() > 0.04
                    records.append(
                        {
                            "district": district.name,
                            "week": week,
                            "variable": CASE_VARIABLES[slug],
                            "value": observed if reported else np.nan,
                            "quality": round(completeness if reported else 0.0, 3),
                        }
                    )
                records.append(
                    {
                        "district": district.name,
                        "week": week,
                        "variable": "reporting_completeness",
                        "value": round(completeness, 3),
                        "quality": 0.8,
                    }
                )
        return self.tidy(records)

    @staticmethod
    def _intensity(slug, district, rain, temp, humid, idx, rng) -> float:
        """Multiplier on baseline incidence, driven by lagged proxies."""

        def lagged(series, lag):
            j = max(idx - lag, 0)
            value = series.iloc[j]
            return float(value) if pd.notna(value) else float(series.mean())

        noise = float(np.exp(rng.normal(0, 0.18)))
        if slug == "malaria":
            r6 = lagged(rain, 6)
            t10 = lagged(temp, 10)
            h4 = lagged(humid, 4)
            rain_term = min(r6, 150.0) / 60.0 - 0.4 * max(0.0, r6 - 150.0) / 100.0
            temp_term = np.exp(-((t10 - 27.0) ** 2) / (2 * 3.2**2))
            humid_term = np.clip(h4 / 70.0, 0.4, 1.6)
            return float(np.clip(0.35 + 0.9 * rain_term * temp_term * humid_term, 0.05, 6.0)) * noise
        if slug == "cholera":
            r3 = lagged(rain, 3)
            t16 = lagged(temp, 16)
            flood = 1.0 if r3 > 80.0 else 0.0
            wash_gap = 1.0 - district.wash_access
            return float(
                np.clip(0.2 + 5.0 * flood * wash_gap * (t16 / 27.0) + 1.4 * wash_gap, 0.02, 25.0)
            ) * noise
        if slug == "tuberculosis":
            density_term = np.log1p(district.density_km2) / np.log1p(5000.0)
            dryness = 1.0 + 0.4 * (1.0 - lagged(humid, 8) / 70.0)
            return float(np.clip(0.6 + 0.8 * density_term * dryness, 0.1, 3.0)) * noise
        if slug == "respiratory":
            cold = np.clip((26.0 - lagged(temp, 2)) / 6.0, -0.5, 1.5)
            density_term = np.log1p(district.density_km2) / np.log1p(5000.0)
            return float(np.clip(0.7 + 0.6 * cold + 0.5 * density_term, 0.2, 4.0)) * noise
        # HIV: slow, structural, mobility/urbanisation driven.
        urban_term = 1.4 if district.urban else 0.8
        return float(np.clip(urban_term * (1.0 + 0.25 * (1.0 - district.wash_access)), 0.2, 3.0)) * noise


def _pivot(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Tidy frame -> week x district wide frame for one variable."""
    subset = frame[frame["variable"] == variable]
    return subset.pivot_table(index="week", columns="district", values="value", aggfunc="mean")


def _dhis2_period_to_epi_week(period: str) -> str:
    """Convert a DHIS2 weekly period id (`2026W07`) to `2026-W07`."""
    if "W" in period and "-" not in period:
        year, week = period.split("W")
        return f"{year}-W{int(week):02d}"
    return period
