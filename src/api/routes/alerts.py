"""Alert endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from src.api.dependencies import get_cache
from src.api.schemas import (
    AlertAcknowledgeRequest,
    AlertAcknowledgeResponse,
    AlertListResponse,
)
from src.core.logging import get_logger
from src.core.types import Alert

router = APIRouter(prefix="/alerts", tags=["alerts"])
log = get_logger("api.alerts")


@router.get("", response_model=AlertListResponse)
def list_alerts(
    disease: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    acknowledged: Optional[bool] = Query(None),
    days: int = Query(30, ge=1, le=365, description="Look-back window"),
    limit: int = Query(100, ge=1, le=500),
) -> AlertListResponse:
    """Recent alerts, newest first."""
    alerts = get_cache().get_alerts(
        disease=disease,
        district=district,
        risk_level=risk_level,
        acknowledged=acknowledged,
        since=datetime.utcnow() - timedelta(days=days),
        limit=limit,
    )
    return AlertListResponse(count=len(alerts), alerts=alerts)


@router.get("/active", response_model=AlertListResponse)
def active_alerts(days: int = Query(14, ge=1, le=90)) -> AlertListResponse:
    """Unacknowledged alerts at medium severity or above — the working queue."""
    alerts = [
        a
        for a in get_cache().get_alerts(
            acknowledged=False, since=datetime.utcnow() - timedelta(days=days), limit=500
        )
        if a.risk_level in ("medium", "high", "critical")
    ]
    alerts.sort(key=lambda a: (("low", "medium", "high", "critical").index(a.risk_level), a.issued_at),
                reverse=True)
    return AlertListResponse(count=len(alerts), alerts=alerts)


@router.get("/{alert_id}", response_model=Alert)
def get_alert(alert_id: str) -> Alert:
    matches = [a for a in get_cache().get_alerts(limit=1000) if a.alert_id == alert_id]
    if not matches:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown alert_id")
    return matches[0]


@router.post("/{alert_id}/acknowledge", response_model=AlertAcknowledgeResponse)
def acknowledge_alert(alert_id: str, request: AlertAcknowledgeRequest) -> AlertAcknowledgeResponse:
    """Acknowledge an alert.

    Acknowledgement is part of the feedback loop, not bookkeeping: an alert that
    is never acknowledged is evidence the warning did not reach a decision-maker.
    """
    if not get_cache().acknowledge_alert(alert_id, by=request.acknowledged_by):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown alert_id")
    log.info("alert %s acknowledged by %s", alert_id, request.acknowledged_by)
    return AlertAcknowledgeResponse(
        alert_id=alert_id, acknowledged=True, acknowledged_by=request.acknowledged_by
    )


@router.post("/bulk", response_model=dict)
def bulk_upload(alerts: List[Alert]) -> dict:
    """Receive alerts pushed by an offline district node on reconnect."""
    saved = get_cache().save_alerts(alerts)
    get_cache().mark_synced("alerts", [a.alert_id for a in alerts])
    return {"received": len(alerts), "saved": saved}


@router.get("/summary/by-district", response_model=dict)
def summary_by_district(days: int = Query(30, ge=1, le=365)) -> dict:
    """Alert counts per district and level — the dashboard's overview table."""
    alerts = get_cache().get_alerts(since=datetime.utcnow() - timedelta(days=days), limit=2000)
    summary: dict = {}
    for alert in alerts:
        bucket = summary.setdefault(
            alert.district,
            {"district": alert.district, "region": alert.region,
             "low": 0, "medium": 0, "high": 0, "critical": 0,
             "unacknowledged": 0, "diseases": set()},
        )
        bucket[alert.risk_level] += 1
        bucket["diseases"].add(alert.disease)
        if not alert.acknowledged:
            bucket["unacknowledged"] += 1
    return {
        "window_days": days,
        "districts": [
            {**bucket, "diseases": sorted(bucket["diseases"])} for bucket in summary.values()
        ],
    }
