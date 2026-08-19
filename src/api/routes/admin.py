"""Administrative endpoints: retraining, ingestion, backtests, offline status.

These are the levers an operator needs and nothing more. They are also the
endpoints most worth protecting with `API_KEY` before exposing the service.
"""

from __future__ import annotations

import time
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from src.api.dependencies import get_cache, get_module, get_region, get_scheduler, reset_caches
from src.api.schemas import BacktestResponse, IngestRequest, RetrainRequest, RetrainResponse
from src.core.logging import get_logger
from src.core.timeutils import shift_week, to_epi_week

router = APIRouter(prefix="/admin", tags=["admin"])
log = get_logger("api.admin")


@router.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest) -> RetrainResponse:
    """Check for drift and refit where needed (shortcoming #3).

    A refit is promoted only if it beats the incumbent on a holdout window, so
    calling this repeatedly cannot degrade the deployed model.
    """
    from src.models.auto_retrain import run_retraining_cycle

    diseases = [request.disease] if request.disease else None
    results = run_retraining_cycle(diseases=diseases, region=get_region(),
                                   history_weeks=request.history_weeks)
    reset_caches()
    return RetrainResponse(results=results)


@router.get("/drift/{disease}", response_model=dict)
def drift_status(disease: str) -> dict:
    """Current drift verdict from the monitored residual stream."""
    from src.models.auto_retrain import AutoRetrainer

    module = get_module(disease)
    retrainer = AutoRetrainer(module)
    residuals = retrainer.monitored_residuals()
    decision = retrainer.should_retrain(residuals=residuals)
    return {
        "disease": disease,
        "monitored_residuals": len(residuals),
        "should_retrain": decision.should_retrain,
        "reasons": decision.reasons,
        "drift_events": decision.drift_events,
        "last_retrain": retrainer.load_state().get("last_retrain"),
    }


@router.post("/ingest", response_model=dict)
def trigger_ingest(request: IngestRequest, background: BackgroundTasks) -> dict:
    """Kick off a data refresh in the background."""
    scheduler = get_scheduler()
    sources = request.sources or scheduler.sources
    end_week = request.end_week or to_epi_week(date.today())

    def _run():
        for source in sources:
            scheduler.run_source(source, end_week=end_week)

    background.add_task(_run)
    return {
        "status": "started",
        "sources": sources,
        "end_week": end_week,
        "note": "Running in the background; poll GET /data/status for progress.",
    }


@router.post("/backtest/{disease}", response_model=BacktestResponse)
def backtest(
    disease: str,
    history_weeks: int = Query(260, ge=104, le=1040),
    max_folds: int = Query(4, ge=1, le=20),
    districts: Optional[List[str]] = Query(None),
) -> BacktestResponse:
    """Run walk-forward validation and return the acceptance report.

    Slow by nature — it refits the model once per fold. Intended for a release
    check, not a routine call.
    """
    from src.data_ingestion.normalizer import ingest
    from src.evaluation.walk_forward_cv import WalkForwardCV

    module = get_module(disease)
    region = get_region()
    end_week = to_epi_week(date.today())
    start_week = shift_week(end_week, -history_weeks)
    sources = sorted(set(module.config.required_sources) | {"dhis2"})

    panel = ingest(sources, start_week, end_week, region=region, districts=districts)
    matrix = module.build_feature_matrix(panel)
    cv = WalkForwardCV(module, initial_train_weeks=156, step_weeks=26, test_weeks=26,
                       max_folds=max_folds)
    result = cv.run(matrix, districts=districts)
    return BacktestResponse(disease=disease, report=result.report())


@router.get("/offline/status", response_model=dict)
def offline_status() -> dict:
    """Can this node keep working through a connectivity outage?"""
    from offline.sync_manager import SyncManager

    return SyncManager(cache=get_cache()).offline_readiness()


@router.post("/offline/sync", response_model=dict)
def trigger_sync(central_url: Optional[str] = Query(None)) -> dict:
    """Push local records upstream and pull fresh forecasts down."""
    from offline.sync_manager import SyncManager

    return SyncManager(cache=get_cache(), central_url=central_url).sync().to_dict()


@router.post("/cache/prune", response_model=dict)
def prune_cache(keep_days: int = Query(120, ge=14, le=1095)) -> dict:
    """Drop stale rows so the local store stays small on a low-spec device."""
    return get_cache().prune(keep_days=keep_days)


@router.get("/registry/validate", response_model=dict)
def validate_registry_endpoint() -> dict:
    """Config and interface check for every registered disease."""
    from src.models.registry import validate_registry

    problems = validate_registry()
    return {
        "valid": not problems,
        "problems": problems,
        "note": "Empty problems means every disease config and module satisfies the contract.",
    }
