"""Normalise every source onto one `district x epi-week` panel (shortcoming #6).

CHIRPS is pentadal GeoTIFF, ERA5 is hourly GRIB, DHIS2 is weekly JSON, CDR is
weekly Parquet, WorldPop is annual raster. They meet here, and only here.

Missing data is *modelled, not zero-filled* (shortcoming #7): the panel keeps
`NaN`s, records a `<variable>__quality` column alongside every value, and offers
neighbour-borrowing imputation that records exactly what it filled and how.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.config_loader import cached_region_config
from src.core.geo import neighbour_map
from src.core.logging import get_logger
from src.core.timeutils import sort_weeks, week_range
from src.core.types import QualityFlag, RegionConfig
from src.data_ingestion.base_adapter import AdapterResult

log = get_logger("ingest.normalizer")

QUALITY_SUFFIX = "__quality"
IMPUTED_SUFFIX = "__imputed"


@dataclass
class Panel:
    """The fused analysis panel handed to feature engineering."""

    frame: pd.DataFrame                     # MultiIndex (district, week) x variables
    sources: Dict[str, str] = field(default_factory=dict)   # variable -> source
    flags: List[QualityFlag] = field(default_factory=list)
    modes: Dict[str, str] = field(default_factory=dict)     # source -> fetch mode
    freshness: Dict[str, object] = field(default_factory=dict)

    @property
    def districts(self) -> List[str]:
        return sorted(self.frame.index.get_level_values("district").unique())

    @property
    def weeks(self) -> List[str]:
        return sort_weeks(self.frame.index.get_level_values("week").unique().tolist())

    @property
    def value_columns(self) -> List[str]:
        return [
            c
            for c in self.frame.columns
            if not c.endswith(QUALITY_SUFFIX) and not c.endswith(IMPUTED_SUFFIX)
        ]

    def values(self) -> pd.DataFrame:
        return self.frame[self.value_columns]

    def quality(self) -> pd.DataFrame:
        cols = [c for c in self.frame.columns if c.endswith(QUALITY_SUFFIX)]
        return self.frame[cols].rename(columns=lambda c: c[: -len(QUALITY_SUFFIX)])

    def mean_quality(self) -> float:
        quality = self.quality()
        return float(quality.to_numpy().mean()) if not quality.empty else 0.0

    def district_frame(self, district: str) -> pd.DataFrame:
        """Week-indexed frame for one district."""
        subset = self.frame.xs(district, level="district")
        return subset.reindex(sort_weeks(subset.index.tolist()))

    def source_count(self) -> int:
        return len(set(self.sources.values()))

    def summary(self) -> Dict[str, object]:
        return {
            "districts": len(self.districts),
            "weeks": len(self.weeks),
            "variables": len(self.value_columns),
            "sources": sorted(set(self.sources.values())),
            "mean_quality": round(self.mean_quality(), 3),
            "completeness": round(float(self.values().notna().to_numpy().mean()), 4),
            "modes": dict(self.modes),
        }


class Normalizer:
    """Fuse tidy adapter output into a single dense panel."""

    def __init__(self, region: Optional[RegionConfig] = None) -> None:
        self.region = region or cached_region_config()
        self.log = log

    # -- main entry point --------------------------------------------------
    def build_panel(
        self,
        results: Sequence[AdapterResult],
        start_week: str,
        end_week: str,
        districts: Optional[Iterable[str]] = None,
        impute: bool = True,
    ) -> Panel:
        """Combine adapter results into one aligned panel."""
        weeks = week_range(start_week, end_week)
        district_names = list(districts) if districts else self.region.district_names
        index = pd.MultiIndex.from_product([district_names, weeks], names=["district", "week"])

        wide_values: Dict[str, pd.Series] = {}
        wide_quality: Dict[str, pd.Series] = {}
        sources: Dict[str, str] = {}
        flags: List[QualityFlag] = []
        modes: Dict[str, str] = {}
        freshness: Dict[str, object] = {}

        for result in results:
            modes[result.source] = result.mode.value
            freshness[result.source] = result.latest_data_date
            flags.extend(result.flags)
            if result.is_empty:
                self.log.warning("source %s contributed no rows", result.source)
                continue
            frame = result.frame
            for variable, group in frame.groupby("variable"):
                pivot = group.set_index(["district", "week"])
                if pivot.index.has_duplicates:
                    pivot = group.groupby(["district", "week"]).agg(
                        value=("value", "mean"), quality=("quality", "min")
                    )
                wide_values[variable] = pivot["value"].reindex(index)
                wide_quality[variable] = pivot["quality"].reindex(index).fillna(0.0)
                sources[variable] = result.source

        if not wide_values:
            self.log.error("no source produced usable data for %s..%s", start_week, end_week)
            return Panel(frame=pd.DataFrame(index=index), flags=flags, modes=modes)

        panel = pd.DataFrame(index=index)
        for variable, series in wide_values.items():
            panel[variable] = series.astype(float)
            panel[f"{variable}{QUALITY_SUFFIX}"] = wide_quality[variable].astype(float)
            panel[f"{variable}{IMPUTED_SUFFIX}"] = 0.0

        result_panel = Panel(
            frame=panel, sources=sources, flags=flags, modes=modes, freshness=freshness
        )
        if impute:
            result_panel = self.impute(result_panel)
        result_panel.flags.extend(self._fusion_flags(result_panel))
        return result_panel

    # -- imputation --------------------------------------------------------
    def impute(self, panel: Panel, max_interpolate_weeks: int = 3) -> Panel:
        """Fill gaps in a way the model can see and account for.

        Order of preference, each recorded in `<variable>__imputed`:

        1. short within-district temporal interpolation (<= 3 weeks),
        2. neighbour-district borrowing for the same week (Bayesian-hierarchical
           in spirit: data-poor districts borrow strength from their neighbours),
        3. the district's own seasonal (week-of-year) climatology,
        4. the variable's national median.

        Case counts are deliberately excluded from steps 1-4: a fabricated case
        count would poison the target. They stay `NaN` and the trainer drops
        those rows explicitly.
        """
        frame = panel.frame.copy()
        neighbours = neighbour_map(self.region, k=5)
        value_cols = panel.value_columns

        for variable in value_cols:
            if variable.startswith("cases_"):
                continue
            series = frame[variable]
            if series.notna().all():
                continue
            imputed_mask = series.isna()

            wide = series.unstack("district")
            wide = wide.reindex(sort_weeks(wide.index.tolist()))

            # 1. short temporal interpolation
            wide = wide.interpolate(
                method="linear", limit=max_interpolate_weeks, limit_direction="both"
            )

            # 2. neighbour borrowing
            still_missing = wide.isna()
            if still_missing.to_numpy().any():
                for district in wide.columns:
                    gaps = wide[district].isna()
                    if not gaps.any():
                        continue
                    pool = [n for n in neighbours.get(district, []) if n in wide.columns]
                    if pool:
                        wide.loc[gaps, district] = wide.loc[gaps, pool].mean(axis=1)

            # 3. own seasonal climatology
            still_missing = wide.isna()
            if still_missing.to_numpy().any():
                week_of_year = pd.Index([int(w.split("-W")[1]) for w in wide.index], name="woy")
                climatology = wide.groupby(week_of_year).transform("mean")
                wide = wide.fillna(climatology)

            # 4. national median
            wide = wide.fillna(wide.median(axis=1).median())

            filled = wide.stack(future_stack=True).reorder_levels(["district", "week"])
            frame[variable] = filled.reindex(frame.index)
            frame.loc[imputed_mask, f"{variable}{IMPUTED_SUFFIX}"] = 1.0
            # Imputed rows can never claim full confidence.
            frame.loc[imputed_mask, f"{variable}{QUALITY_SUFFIX}"] = (
                frame.loc[imputed_mask, f"{variable}{QUALITY_SUFFIX}"].clip(upper=0.4)
            )

        panel.frame = frame
        return panel

    # -- diagnostics -------------------------------------------------------
    def _fusion_flags(self, panel: Panel) -> List[QualityFlag]:
        flags: List[QualityFlag] = []
        n_sources = panel.source_count()
        if n_sources < 3:
            flags.append(
                QualityFlag(
                    code="insufficient_fusion",
                    severity="error",
                    message=(
                        f"only {n_sources} source(s) contributed data; rule #1 requires >= 3. "
                        "Predictions will be produced with widened intervals and reduced confidence."
                    ),
                )
            )
        synthetic = [s for s, mode in panel.modes.items() if mode == "synthetic"]
        if synthetic:
            flags.append(
                QualityFlag(
                    code="synthetic_sources",
                    severity="warning",
                    message=(
                        f"{len(synthetic)} source(s) served synthetic data ({', '.join(sorted(synthetic))}); "
                        "results are demonstrative, not operational"
                    ),
                )
            )
        imputed_cols = [c for c in panel.frame.columns if c.endswith(IMPUTED_SUFFIX)]
        if imputed_cols:
            share = float(panel.frame[imputed_cols].to_numpy().mean())
            if share > 0.2:
                flags.append(
                    QualityFlag(
                        code="high_imputation",
                        severity="warning",
                        message=f"{share:.1%} of driver cells were imputed rather than observed",
                    )
                )
        return flags


def ingest(
    sources: Sequence[str],
    start_week: str,
    end_week: str,
    region: Optional[RegionConfig] = None,
    districts: Optional[Iterable[str]] = None,
    impute: bool = True,
    **adapter_kwargs,
) -> Panel:
    """Convenience: run every named adapter and fuse the results into a panel."""
    from src.data_ingestion.registry import get_adapter

    region = region or cached_region_config()
    results: List[AdapterResult] = []
    for source in sources:
        kwargs = adapter_kwargs.get(source, {})
        adapter = get_adapter(source, region=region, **kwargs)
        results.append(adapter.run(start_week, end_week))
    return Normalizer(region).build_panel(
        results, start_week, end_week, districts=districts, impute=impute
    )
