"""Data freshness and quality endpoints (shortcoming #7).

An operator needs to know *before* acting whether a forecast rests on live
DHIS2 data or on a synthetic fallback, and whether the fusion rule is actually
being met. Hiding that behind a clean-looking prediction is how a system loses
trust the first time it is wrong.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query

from src.api.dependencies import get_region, get_scheduler
from src.api.schemas import DataStatusResponse, SourceStatus
from src.core.logging import get_logger

router = APIRouter(prefix="/data", tags=["data"])
log = get_logger("api.data")


@router.get("/status", response_model=DataStatusResponse)
def data_status() -> DataStatusResponse:
    """Per-source freshness, mode and quality."""
    from src.data_ingestion.registry import get_adapter

    rows = get_scheduler().status()
    statuses: List[SourceStatus] = []
    for row in rows:
        try:
            configured = get_adapter(row["source"], region=get_region()).is_configured()
        except Exception:  # noqa: BLE001
            configured = False
        statuses.append(SourceStatus(**row, configured=configured))

    live = [s for s in statuses if s.in_use and s.mode == "live"]
    warnings: List[str] = []
    stale = [s.source for s in statuses if s.in_use and s.stale]
    if stale:
        warnings.append(
            f"{len(stale)} source(s) are stale or have never run: {', '.join(sorted(stale))}. "
            "Forecasts will fall back to cache or synthetic data with reduced confidence."
        )
    synthetic = [s.source for s in statuses if s.mode == "synthetic"]
    if synthetic:
        warnings.append(
            f"{len(synthetic)} source(s) last served synthetic data ({', '.join(sorted(synthetic))}). "
            "Results are demonstrative, not operational."
        )
    if len(live) < 3:
        warnings.append(
            f"only {len(live)} source(s) returned live data; critical rule #1 requires fusing "
            "at least 3. Predictions remain available with widened confidence intervals."
        )
    return DataStatusResponse(
        sources=statuses,
        fused_sources=len(live),
        meets_fusion_rule=len(live) >= 3,
        warnings=warnings,
    )


@router.get("/sources", response_model=list)
def list_sources() -> list:
    """Capability report for every registered adapter."""
    from src.data_ingestion.registry import describe_sources

    return describe_sources(region=get_region())


@router.get("/quality", response_model=dict)
def data_quality(
    disease: Optional[str] = Query(None),
    weeks: int = Query(12, ge=1, le=104),
) -> dict:
    """Recent input-quality summary for the data-quality dashboard page."""
    from datetime import date

    from src.core.config_loader import load_disease_config
    from src.core.timeutils import shift_week, to_epi_week
    from src.data_ingestion.normalizer import ingest
    from src.data_ingestion.quality_checks import quality_report

    end_week = to_epi_week(date.today())
    start_week = shift_week(end_week, -(weeks - 1))
    sources = (
        sorted(set(load_disease_config(disease).required_sources) | {"dhis2"})
        if disease
        else ["chirps", "era5", "dhis2"]
    )
    region = get_region()
    # Sample a handful of districts: a full national quality scan on a request
    # would blow the latency budget for no extra insight.
    sample = region.district_names[:8]
    panel = ingest(sources, start_week, end_week, region=region, districts=sample, impute=False)

    frames = {}
    values = panel.values()
    for variable in values.columns:
        source = panel.sources.get(variable, "unknown")
        frames.setdefault(source, []).append(values[variable])

    summary = []
    for source, series_list in frames.items():
        stacked = sum(s.notna().sum() for s in series_list)
        total = sum(len(s) for s in series_list)
        summary.append(
            {
                "source": source,
                "mode": panel.modes.get(source),
                "variables": len(series_list),
                "completeness": round(stacked / total, 4) if total else 0.0,
            }
        )
    return {
        "window": {"start_week": start_week, "end_week": end_week},
        "districts_sampled": sample,
        "sources": summary,
        "panel": panel.summary(),
        "flags": [f.model_dump() for f in panel.flags],
    }
