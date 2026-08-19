"""Intervention logging and impact endpoints (shortcoming #15)."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from src.api.dependencies import get_cache, get_region
from src.api.schemas import (
    ImpactResponse,
    InterventionListResponse,
    InterventionRequest,
    InterventionResponse,
)
from src.core.logging import get_logger
from src.core.types import Intervention

router = APIRouter(prefix="/interventions", tags=["interventions"])
log = get_logger("api.interventions")


@router.post("/log", response_model=InterventionResponse, status_code=status.HTTP_201_CREATED)
def log_intervention(request: InterventionRequest) -> InterventionResponse:
    """Record a response action so its effect can later be estimated."""
    from src.intervention_tracking.intervention_logger import InterventionLogger

    logger = InterventionLogger(cache=get_cache())
    intervention = logger.log(
        disease=request.disease,
        district=request.district,
        intervention_type=request.intervention_type,
        started_week=request.started_week,
        ended_week=request.ended_week,
        coverage=request.coverage,
        quantity=request.quantity,
        unit=request.unit,
        alert_id=request.alert_id,
        notes=request.notes,
        logged_by=request.logged_by,
    )
    return InterventionResponse(intervention=intervention)


@router.get("", response_model=InterventionListResponse)
def list_interventions(
    disease: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> InterventionListResponse:
    interventions = get_cache().get_interventions(disease=disease, district=district, limit=limit)
    return InterventionListResponse(count=len(interventions), interventions=interventions)


@router.get("/types", response_model=list)
def intervention_types() -> list:
    """Known response types and the lag before an effect is plausible."""
    from src.intervention_tracking.intervention_logger import INTERVENTION_TYPES

    return [
        {"type": key, **{k: v for k, v in value.items()}}
        for key, value in INTERVENTION_TYPES.items()
    ]


@router.post("/bulk", response_model=dict)
def bulk_upload(interventions: List[Intervention]) -> dict:
    """Receive interventions pushed by an offline district node."""
    cache = get_cache()
    for intervention in interventions:
        cache.save_intervention(intervention)
    cache.mark_synced("interventions", [i.intervention_id for i in interventions])
    return {"received": len(interventions)}


@router.get("/{intervention_id}/impact", response_model=ImpactResponse)
def estimate_impact(
    intervention_id: str,
    weeks_before: int = Query(12, ge=4, le=52),
    weeks_after: int = Query(12, ge=4, le=52),
) -> ImpactResponse:
    """Estimate what an intervention changed, against three counterfactuals."""
    import pandas as pd

    from src.intervention_tracking.impact_estimator import ImpactEstimator

    cache = get_cache()
    matches = [i for i in cache.get_interventions(limit=2000) if i.intervention_id == intervention_id]
    if not matches:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown intervention_id")
    intervention = matches[0]

    observations = cache.get_observations(intervention.disease, limit=5000)
    if not observations:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "No observed case history cached for this disease; impact cannot be estimated. "
                "Run the ingestion job to populate observations."
            ),
        )
    frame = pd.DataFrame(observations)
    wide = frame.pivot_table(index="week", columns="district", values="cases", aggfunc="mean")
    wide = wide.sort_index()

    if intervention.district not in wide.columns:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"No cached observations for {intervention.district}",
        )
    observed = wide[intervention.district]
    controls = wide.drop(columns=[intervention.district])

    forecasts = cache.latest_predictions(
        disease=intervention.disease, district=intervention.district, limit=200
    )
    forecast_series = None
    if forecasts:
        forecast_series = pd.Series(
            {p.target_week: p.predicted_cases for p in forecasts}
        ).sort_index()

    estimate = ImpactEstimator(region=get_region()).estimate(
        intervention, observed, forecast=forecast_series, control_series=controls
    )
    return ImpactResponse(
        intervention_id=intervention_id,
        estimates=estimate.estimates,
        confidence=estimate.confidence,
        narrative=estimate.narrative(),
        caveats=estimate.caveats,
    )


@router.get("/audit/responses", response_model=dict)
def response_audit(days: int = Query(90, ge=7, le=730)) -> dict:
    """Did alerts actually produce responses, and fast enough to matter?"""
    from datetime import datetime, timedelta

    from src.intervention_tracking.feedback_loop import FeedbackLoop

    cache = get_cache()
    alerts = cache.get_alerts(since=datetime.utcnow() - timedelta(days=days), limit=2000)
    return FeedbackLoop(cache=cache).audit_responses(alerts).to_dict()
