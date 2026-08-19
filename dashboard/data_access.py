"""Data access for the dashboard.

The dashboard reads through the API when one is reachable and falls back to the
local SQLite cache otherwise. That is not a convenience — it is the whole point
of shortcoming #14: a district officer on an intermittent link must still be
able to open the dashboard and see the last forecasts that arrived.

Every function here returns plain pandas objects, so the page modules contain
only presentation logic.
"""

from __future__ import annotations

import functools
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

API_URL = os.environ.get("API_URL", "").rstrip("/")
REQUEST_TIMEOUT = 5


# ---------------------------------------------------------------- connection
@functools.lru_cache(maxsize=1)
def _cache():
    from offline.local_cache import LocalCache

    return LocalCache()


def api_available() -> bool:
    """Is the API reachable right now?"""
    if not API_URL:
        return False
    try:
        import requests

        return requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT).status_code < 500
    except Exception:  # noqa: BLE001
        return False


def _get(path: str, **params) -> Optional[Any]:
    if not API_URL:
        return None
    try:
        import requests

        response = requests.get(f"{API_URL}{path}", params=params, timeout=REQUEST_TIMEOUT * 4)
        response.raise_for_status()
        return response.json()
    except Exception:  # noqa: BLE001 - the local cache is always the fallback
        return None


def data_source_label() -> str:
    """What the current page is actually reading from — shown in the sidebar."""
    if api_available():
        return f"live API ({API_URL})"
    if API_URL:
        return "local cache (API unreachable — showing the last data that arrived)"
    return "local cache (no API configured)"


# ------------------------------------------------------------------- region
@functools.lru_cache(maxsize=1)
def region():
    from src.core.config_loader import cached_region_config

    return cached_region_config()


@functools.lru_cache(maxsize=1)
def district_frame() -> pd.DataFrame:
    from src.core.geo import district_frame as build

    return build(region()).reset_index()


@functools.lru_cache(maxsize=1)
def diseases() -> List[str]:
    payload = _get("/diseases")
    if payload:
        return [d["slug"] for d in payload.get("diseases", [])]
    from src.models.registry import list_modules

    return list_modules()


@functools.lru_cache(maxsize=1)
def disease_details() -> pd.DataFrame:
    payload = _get("/diseases")
    if payload:
        return pd.DataFrame(payload.get("diseases", []))
    from src.models.registry import describe_all

    return pd.DataFrame(describe_all(region=region()))


# -------------------------------------------------------------- predictions
def predictions(
    disease: Optional[str] = None,
    district: Optional[str] = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Latest cached forecasts as a flat frame."""
    payload = _get("/predictions", disease=disease, district=district, limit=limit)
    if payload is not None:
        records = payload.get("predictions", [])
    else:
        records = [
            p.model_dump(mode="json")
            for p in _cache().latest_predictions(disease=disease, district=district, limit=limit)
        ]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    for column in ("forecast_date",):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def prediction_detail(prediction_id: str) -> Optional[dict]:
    payload = _get(f"/predictions/id/{prediction_id}")
    if payload:
        return payload.get("prediction")
    result = _cache().get_prediction(prediction_id)
    return result.model_dump(mode="json") if result else None


def latest_per_district(disease: str) -> pd.DataFrame:
    """One row per district: the most recent forecast, with coordinates joined."""
    frame = predictions(disease=disease, limit=2000)
    if frame.empty:
        return frame
    frame = frame.sort_values("target_week").groupby("district", as_index=False).last()
    return frame.merge(
        district_frame()[["name", "lat", "lon", "population", "region"]],
        left_on="district", right_on="name", how="left", suffixes=("", "_geo"),
    )


# ------------------------------------------------------------------- alerts
def alerts(
    disease: Optional[str] = None,
    district: Optional[str] = None,
    days: int = 60,
    limit: int = 500,
) -> pd.DataFrame:
    payload = _get("/alerts", disease=disease, district=district, days=days, limit=limit)
    if payload is not None:
        records = payload.get("alerts", [])
    else:
        records = [
            a.model_dump(mode="json")
            for a in _cache().get_alerts(
                disease=disease, district=district,
                since=datetime.utcnow() - timedelta(days=days), limit=limit,
            )
        ]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    if "issued_at" in frame:
        frame["issued_at"] = pd.to_datetime(frame["issued_at"], errors="coerce")
    return frame


def acknowledge(alert_id: str, by: str) -> bool:
    if API_URL:
        try:
            import requests

            response = requests.post(
                f"{API_URL}/alerts/{alert_id}/acknowledge",
                json={"acknowledged_by": by}, timeout=REQUEST_TIMEOUT * 2,
            )
            if response.ok:
                return True
        except Exception:  # noqa: BLE001 - fall through to the local write
            pass
    # Written locally and pushed upstream on the next sync (rule #6).
    return _cache().acknowledge_alert(alert_id, by=by)


# ------------------------------------------------------------- observations
def observations(disease: str, district: Optional[str] = None, limit: int = 2000) -> pd.DataFrame:
    rows = _cache().get_observations(disease, district=district, limit=limit)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("week")


# ------------------------------------------------------------ data quality
def data_status() -> pd.DataFrame:
    payload = _get("/data/status")
    if payload is not None:
        return pd.DataFrame(payload.get("sources", []))
    from src.data_ingestion.scheduler import IngestionScheduler

    return pd.DataFrame(IngestionScheduler(region=region()).status())


def data_status_warnings() -> List[str]:
    payload = _get("/data/status")
    if payload is not None:
        return payload.get("warnings", [])
    frame = data_status()
    if frame.empty:
        return ["No ingestion has run yet on this node."]
    stale = frame[frame.get("stale", False) & frame.get("in_use", False)]
    warnings = []
    if len(stale):
        warnings.append(
            f"{len(stale)} source(s) are stale or have never run: "
            f"{', '.join(sorted(stale['source']))}."
        )
    return warnings


# ------------------------------------------------------------ interventions
def interventions(disease: Optional[str] = None, district: Optional[str] = None) -> pd.DataFrame:
    payload = _get("/interventions", disease=disease, district=district)
    if payload is not None:
        records = payload.get("interventions", [])
    else:
        records = [
            i.model_dump(mode="json")
            for i in _cache().get_interventions(disease=disease, district=district)
        ]
    return pd.DataFrame(records) if records else pd.DataFrame()


def log_intervention(payload: Dict[str, Any]) -> Optional[dict]:
    if API_URL:
        try:
            import requests

            response = requests.post(
                f"{API_URL}/interventions/log", json=payload, timeout=REQUEST_TIMEOUT * 2
            )
            if response.ok:
                return response.json().get("intervention")
        except Exception:  # noqa: BLE001
            pass
    from src.intervention_tracking.intervention_logger import InterventionLogger

    logger = InterventionLogger(cache=_cache())
    return logger.log(**payload).model_dump(mode="json")


def intervention_types() -> pd.DataFrame:
    payload = _get("/interventions/types")
    if payload is not None:
        return pd.DataFrame(payload)
    from src.intervention_tracking.intervention_logger import INTERVENTION_TYPES

    return pd.DataFrame([{"type": k, **v} for k, v in INTERVENTION_TYPES.items()])


# ------------------------------------------------------------------ offline
def offline_status() -> dict:
    payload = _get("/admin/offline/status")
    if payload is not None:
        return payload
    from offline.sync_manager import SyncManager

    return SyncManager(cache=_cache()).offline_readiness()


def cache_status() -> dict:
    return _cache().status()


def response_audit(days: int = 90) -> dict:
    payload = _get("/interventions/audit/responses", days=days)
    if payload is not None:
        return payload
    from src.intervention_tracking.feedback_loop import FeedbackLoop

    cache = _cache()
    recent = cache.get_alerts(since=datetime.utcnow() - timedelta(days=days), limit=2000)
    return FeedbackLoop(cache=cache).audit_responses(recent).to_dict()
