"""Google Trends adapter — OPTIONAL urban-boost layer only (shortcoming #14).

Google Flu Trends failed partly because search was its *only* signal. Tanzania's
smartphone penetration is around 35% against ~87% feature-phone ownership, so
search interest systematically misses the rural districts carrying most of the
burden.

This adapter is therefore marked `optional = True`: the feature pipeline may add
it as a secondary urban signal, and the model registry refuses to let it act as
a primary predictor.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_ingestion.base_adapter import BaseAdapter

#: Search terms per disease, Swahili first — the language actual users type.
SEARCH_TERMS: Dict[str, List[str]] = {
    "malaria": ["malaria", "homa ya malaria", "dawa ya malaria"],
    "cholera": ["kipindupindu", "cholera", "kuharisha"],
    "tuberculosis": ["kifua kikuu", "tb symptoms"],
    "respiratory": ["kikohozi", "homa", "pneumonia"],
    "hiv": ["ukimwi", "hiv test", "prep"],
}


class SearchTrendsAdapter(BaseAdapter):
    source_name = "search_trends"
    variables = tuple(f"search_{slug}" for slug in SEARCH_TERMS)
    update_frequency_days = 1
    optional = True  # never a primary predictor
    native_resolution = "region (sub-national Google Trends)"

    def is_configured(self) -> bool:
        try:
            import pytrends  # noqa: F401
        except ImportError:
            return False
        return True

    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Pull sub-national interest-over-time and map regions to councils."""
        from pytrends.request import TrendReq

        from src.core.timeutils import epi_week_end, epi_week_start

        pytrends = TrendReq(hl="sw-TZ", tz=180)
        timeframe = f"{epi_week_start(weeks[0])} {epi_week_end(weeks[-1])}"
        records = []
        for slug, terms in SEARCH_TERMS.items():
            pytrends.build_payload(terms[:5], timeframe=timeframe, geo="TZ")
            frame = pytrends.interest_over_time()
            if frame.empty:
                continue
            frame["week"] = frame.index.strftime("%G-W%V")
            frame["interest"] = frame[terms[:5]].mean(axis=1)
            weekly = frame.groupby("week")["interest"].mean()
            for week, value in weekly.items():
                if week not in set(weeks):
                    continue
                # National series; urban districts carry it, rural districts get
                # a heavily damped share to reflect the connectivity gap.
                for district in self.region.districts:
                    weight = 1.0 if district.urban else 0.15
                    records.append(
                        {
                            "district": district.name,
                            "week": str(week),
                            "variable": f"search_{slug}",
                            "value": float(value) * weight,
                            "quality": 0.7 if district.urban else 0.25,
                        }
                    )
        return self.tidy(records)

    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Urban-weighted search interest that mildly leads case presentation."""
        records = []
        for district in self.region.districts:
            rng = self._rng(district.name)
            connectivity = 0.9 if district.urban else 0.2
            for week in weeks:
                phase = self._season_phase(week)
                for slug in SEARCH_TERMS:
                    seasonal = 1.0 + 0.4 * np.cos(phase - np.radians(70 if slug == "malaria" else 200))
                    value = 40.0 * connectivity * seasonal * float(np.exp(rng.normal(0, 0.2)))
                    records.append(
                        {
                            "district": district.name,
                            "week": week,
                            "variable": f"search_{slug}",
                            "value": round(float(np.clip(value, 0, 100)), 2),
                            "quality": 0.5 if district.urban else 0.2,
                        }
                    )
        return self.tidy(records)
