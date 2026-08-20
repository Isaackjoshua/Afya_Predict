"""Prediction endpoints.

Reads are served from the local cache, which the scheduler populates. That is
what keeps `GET /predictions/...` well inside the 2-second budget: a request
never trains a model or rebuilds a feature panel. `POST /predictions/run`
exists for an explicit, and explicitly slower, on-demand forecast.
"""

from __future__ import annotations

import time
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from src.api.dependencies import get_cache, get_module, get_region, validate_district
from src.api.schemas import (
    ForecastRequest,
    ForecastRunResponse,
    PredictionListResponse,
    PredictionResponse,
)
from src.core.logging import get_logger
from src.core.timeutils import shift_week, to_epi_week

router = APIRouter(prefix="/predictions", tags=["predictions"])
log = get_logger("api.predictions")


@router.get("", response_model=PredictionListResponse)
def list_predictions(
    disease: Optional[str] = Query(None, description="Disease slug, e.g. malaria"),
    district: Optional[str] = Query(None),
    target_week: Optional[str] = Query(None, description="ISO week, e.g. 2026-W12"),
    risk_level: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    limit: int = Query(100, ge=1, le=1000),
) -> PredictionListResponse:
    """Latest cached forecasts, filtered."""
    predictions = get_cache().latest_predictions(
        disease=disease, district=district, target_week=target_week,
        risk_level=risk_level, limit=limit,
    )
    warnings: List[str] = []
    if not predictions:
        warnings.append(
            "No cached forecasts match this query. Run the prediction job "
            "(POST /predictions/run or scripts/run_backtest.py) to populate the cache."
        )
    return PredictionListResponse(
        count=len(predictions), predictions=predictions, warnings=warnings
    )


# Declared before "/{disease}/{district}": Starlette matches routes in
# order, so the two-parameter route was swallowing every by-id lookup and
# rejecting the id as an unknown district.
@router.get("/id/{prediction_id}", response_model=PredictionResponse)
def prediction_by_id(prediction_id: str) -> PredictionResponse:
    prediction = get_cache().get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown prediction_id")
    return PredictionResponse(prediction=prediction)


@router.get("/{disease}/{district}", response_model=PredictionListResponse)
def district_predictions(
    disease: str,
    district: str,
    limit: int = Query(12, ge=1, le=104),
) -> PredictionListResponse:
    """Forecast history for one district, newest first."""
    validate_district(district)
    predictions = get_cache().latest_predictions(disease=disease, district=district, limit=limit)
    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No cached forecast for {disease} in {district}. "
                "Run POST /predictions/run to generate one."
            ),
        )
    return PredictionListResponse(count=len(predictions), predictions=predictions)


@router.post("/run", response_model=ForecastRunResponse)
def run_forecast(request: ForecastRequest) -> ForecastRunResponse:
    """Generate fresh forecasts on demand.

    This is the slow path — it ingests, rebuilds features and predicts — and is
    meant for manual refreshes and testing. Routine forecasting belongs to the
    scheduler.
    """
    from src.data_ingestion.normalizer import ingest

    started = time.perf_counter()
    module = get_module(request.disease)
    region = get_region()
    warnings: List[str] = []

    end_week = request.end_week or to_epi_week(date.today())
    start_week = shift_week(end_week, -208)
    sources = sorted(set(module.config.required_sources) | {"dhis2"})
    panel = ingest(sources, start_week, end_week, region=region, districts=request.districts)

    for flag in panel.flags:
        if flag.severity == "error":
            warnings.append(str(flag))

    matrix = module.build_feature_matrix(panel, horizon_weeks=request.horizon_weeks)
    if not module.is_trained():
        log.info("no persisted model for %s; training now", request.disease)
        module.train(matrix, districts=request.districts)
        module.save()

    districts = request.districts or matrix.districts
    predictions = module.predict_all(matrix, districts=districts, panel=panel)
    alerts = module.detect_outbreak(predictions)

    if request.persist:
        cache = get_cache()
        cache.save_predictions(predictions)
        cache.save_alerts(alerts)

    return ForecastRunResponse(
        disease=request.disease,
        districts=len(districts),
        predictions_generated=len(predictions),
        alerts_generated=len(alerts),
        duration_seconds=round(time.perf_counter() - started, 2),
        warnings=warnings,
    )


@router.get("/{disease}/map/national", tags=["predictions"])
def national_risk_map(
    disease: str, target_week: Optional[str] = Query(None)
) -> dict:
    """District-level risk scores for the national heatmap."""
    predictions = get_cache().latest_predictions(disease=disease, target_week=target_week, limit=1000)
    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No cached forecasts for {disease}"
        )
    latest: dict = {}
    for prediction in predictions:
        # `latest_predictions` is newest-first, so the first hit per district wins.
        latest.setdefault(prediction.district, prediction)
    region = get_region()
    return {
        "disease": disease,
        "target_week": target_week or max(p.target_week for p in predictions),
        "districts": [
            {
                "district": name,
                "region": prediction.region,
                "lat": region.get(name).lat if name in set(region.district_names) else None,
                "lon": region.get(name).lon if name in set(region.district_names) else None,
                "risk_level": prediction.risk_level,
                "risk_score": prediction.risk_score,
                "predicted_cases": prediction.predicted_cases,
                "importation_risk": prediction.importation_risk,
                "prediction_id": prediction.prediction_id,
            }
            for name, prediction in latest.items()
        ],
    }
