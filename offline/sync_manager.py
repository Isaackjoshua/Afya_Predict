"""Reconcile an offline district node with the central instance.

The sync contract is deliberately one-way-safe:

* **push** — locally raised alerts, acknowledgements and logged interventions go
  up. These are the records a district creates and the centre cannot reconstruct.
* **pull** — new forecasts, refreshed model artefacts and updated configuration
  come down.
* **queue flush** — any notification that failed to send while offline is
  retried on reconnect, so an alert raised during an outage still reaches its
  recipients (rule #6).

Nothing is deleted on either side by a sync; conflicts resolve in favour of the
district's own record, because that is where the ground truth is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from src.core.logging import get_logger
from offline.local_cache import LocalCache

log = get_logger("offline.sync")


@dataclass
class SyncReport:
    """Outcome of one synchronisation attempt."""

    started_at: datetime = field(default_factory=datetime.utcnow)
    online: bool = False
    alerts_pushed: int = 0
    interventions_pushed: int = 0
    predictions_pulled: int = 0
    notifications_flushed: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.online and not self.errors

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "online": self.online,
            "alerts_pushed": self.alerts_pushed,
            "interventions_pushed": self.interventions_pushed,
            "predictions_pulled": self.predictions_pulled,
            "notifications_flushed": self.notifications_flushed,
            "errors": self.errors,
            "ok": self.ok,
        }


class SyncManager:
    """Push local records upstream and pull fresh forecasts down."""

    def __init__(
        self,
        cache: Optional[LocalCache] = None,
        central_url: Optional[str] = None,
        settings=None,
        timeout: int = 30,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or LocalCache(settings=self.settings)
        self.central_url = (central_url or "").rstrip("/")
        self.timeout = timeout

    # -- connectivity ------------------------------------------------------
    def is_online(self) -> bool:
        """Cheap reachability probe against the central instance."""
        if self.settings.offline_mode or not self.central_url:
            return False
        try:
            import requests

            response = requests.get(f"{self.central_url}/health", timeout=self.timeout)
            return response.status_code < 500
        except Exception as exc:  # noqa: BLE001
            log.debug("central instance unreachable: %s", exc)
            return False

    # -- sync --------------------------------------------------------------
    def sync(self) -> SyncReport:
        report = SyncReport()
        report.online = self.is_online()

        # The notification queue is retried whether or not the central instance
        # is up: SMS and email gateways are independent of it.
        try:
            from src.alerting.notification_service import NotificationService

            flushed = NotificationService(settings=self.settings).flush_queue()
            report.notifications_flushed = flushed.get("sent", 0)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"notification flush failed: {exc}")

        if not report.online:
            log.info("offline: %d queued notification(s) retried, sync deferred",
                     report.notifications_flushed)
            self.cache.set_meta("last_sync_attempt", report.to_dict())
            return report

        try:
            report.alerts_pushed = self._push_alerts()
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"alert push failed: {exc}")
        try:
            report.interventions_pushed = self._push_interventions()
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"intervention push failed: {exc}")
        try:
            report.predictions_pulled = self._pull_predictions()
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"prediction pull failed: {exc}")

        self.cache.set_meta("last_sync", report.to_dict())
        log.info("sync complete: %s", report.to_dict())
        return report

    # -- push --------------------------------------------------------------
    def _push_alerts(self) -> int:
        import requests

        alerts = self.cache.unsynced_alerts()
        if not alerts:
            return 0
        response = requests.post(
            f"{self.central_url}/alerts/bulk",
            json=[a.model_dump(mode="json") for a in alerts],
            timeout=self.timeout,
        )
        response.raise_for_status()
        self.cache.mark_synced("alerts", [a.alert_id for a in alerts])
        return len(alerts)

    def _push_interventions(self) -> int:
        import requests

        interventions = self.cache.unsynced_interventions()
        if not interventions:
            return 0
        response = requests.post(
            f"{self.central_url}/interventions/bulk",
            json=[i.model_dump(mode="json") for i in interventions],
            timeout=self.timeout,
        )
        response.raise_for_status()
        self.cache.mark_synced("interventions", [i.intervention_id for i in interventions])
        return len(interventions)

    # -- pull --------------------------------------------------------------
    def _pull_predictions(self, limit: int = 500) -> int:
        import requests

        from src.core.types import PredictionResult

        response = requests.get(
            f"{self.central_url}/predictions", params={"limit": limit}, timeout=self.timeout
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("predictions", payload if isinstance(payload, list) else [])
        predictions = []
        for item in items:
            try:
                predictions.append(PredictionResult.model_validate(item))
            except Exception:  # noqa: BLE001 - skip a malformed record, keep the rest
                continue
        return self.cache.save_predictions(predictions)

    # -- readiness ---------------------------------------------------------
    def offline_readiness(self, required_weeks: int = 2) -> Dict[str, Any]:
        """Can this node survive a disconnection? (acceptance criterion #12)"""
        status = self.cache.status()
        weeks = status["weeks_cached"]
        return {
            "ready": weeks >= required_weeks,
            "weeks_cached": weeks,
            "required_weeks": required_weeks,
            "predictions_cached": status["predictions"],
            "latest_target_week": status["latest_target_week"],
            "unsynced_alerts": status["unsynced_alerts"],
            "unsynced_interventions": status["unsynced_interventions"],
            "last_sync": self.cache.get_meta("last_sync"),
            "message": (
                f"{weeks} week(s) of forecasts cached locally; node can operate offline"
                if weeks >= required_weeks
                else f"only {weeks} week(s) cached — run the prediction job before disconnecting"
            ),
        }
