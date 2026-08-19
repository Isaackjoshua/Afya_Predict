"""SQLite cache backing offline operation and fast API responses.

Two jobs in one store:

* **offline** — a district node keeps at least two weeks of forecasts, alerts
  and explanations locally, so the dashboard and API keep working through an
  outage (acceptance criterion #12);
* **latency** — predictions are precomputed by the scheduler and read back by
  key, which is what keeps `GET /predictions/...` well under the 2-second
  requirement instead of retraining on the request path.

SQLite specifically: no server to run, no ops burden, and the file can be
copied onto a USB stick and carried to a district office.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from config.settings import get_settings
from src.core.logging import get_logger
from src.core.types import Alert, Intervention, PredictionResult

log = get_logger("offline.cache")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    disease       TEXT NOT NULL,
    district      TEXT NOT NULL,
    region        TEXT,
    forecast_date TEXT NOT NULL,
    target_week   TEXT NOT NULL,
    predicted_cases REAL,
    risk_level    TEXT,
    risk_score    REAL,
    model_version TEXT,
    payload       TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_lookup ON predictions(disease, district, target_week);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id      TEXT PRIMARY KEY,
    disease       TEXT NOT NULL,
    district      TEXT NOT NULL,
    region        TEXT,
    issued_at     TEXT NOT NULL,
    target_week   TEXT NOT NULL,
    risk_level    TEXT,
    acknowledged  INTEGER DEFAULT 0,
    synced        INTEGER DEFAULT 0,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alert_lookup ON alerts(disease, district, issued_at);
CREATE INDEX IF NOT EXISTS idx_alert_sync ON alerts(synced);

CREATE TABLE IF NOT EXISTS interventions (
    intervention_id TEXT PRIMARY KEY,
    disease       TEXT NOT NULL,
    district      TEXT NOT NULL,
    alert_id      TEXT,
    started_week  TEXT NOT NULL,
    synced        INTEGER DEFAULT 0,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intervention_sync ON interventions(synced);

CREATE TABLE IF NOT EXISTS observations (
    disease       TEXT NOT NULL,
    district      TEXT NOT NULL,
    week          TEXT NOT NULL,
    cases         REAL,
    incidence_per_1000 REAL,
    quality       REAL,
    PRIMARY KEY (disease, district, week)
);

CREATE TABLE IF NOT EXISTS meta (
    key           TEXT PRIMARY KEY,
    value         TEXT,
    updated_at    TEXT
);
"""


class LocalCache:
    """Thread-safe SQLite store for the offline node."""

    def __init__(self, path: Optional[Path] = None, settings=None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path or self.settings.sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._lock, self.connect() as connection:
            connection.executescript(SCHEMA)

    # -- predictions -------------------------------------------------------
    def save_predictions(self, predictions: Iterable[PredictionResult]) -> int:
        rows = [
            (
                p.prediction_id, p.disease, p.district, p.region, str(p.forecast_date),
                p.target_week, p.predicted_cases, p.risk_level, p.risk_score,
                p.model_version, json.dumps(p.model_dump(), default=str),
                datetime.utcnow().isoformat(timespec="seconds"),
            )
            for p in predictions
        ]
        if not rows:
            return 0
        with self._lock, self.connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
        return len(rows)

    def get_prediction(self, prediction_id: str) -> Optional[PredictionResult]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM predictions WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
        return PredictionResult.model_validate(json.loads(row["payload"])) if row else None

    def latest_predictions(
        self,
        disease: Optional[str] = None,
        district: Optional[str] = None,
        target_week: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 200,
    ) -> List[PredictionResult]:
        """Most recent forecast per district, filtered — the API's hot path."""
        clauses, params = [], []
        for column, value in (
            ("disease", disease), ("district", district),
            ("target_week", target_week), ("risk_level", risk_level),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT payload FROM predictions {where}
            ORDER BY created_at DESC, target_week DESC LIMIT ?
        """
        params.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [PredictionResult.model_validate(json.loads(r["payload"])) for r in rows]

    def prediction_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"])

    def weeks_cached(self, disease: Optional[str] = None) -> int:
        """Distinct target weeks held locally — acceptance criterion #12."""
        query = "SELECT COUNT(DISTINCT target_week) c FROM predictions"
        params: List[Any] = []
        if disease:
            query += " WHERE disease = ?"
            params.append(disease)
        with self.connect() as connection:
            return int(connection.execute(query, params).fetchone()["c"])

    # -- alerts ------------------------------------------------------------
    def save_alerts(self, alerts: Iterable[Alert]) -> int:
        rows = [
            (
                a.alert_id, a.disease, a.district, a.region, a.issued_at.isoformat(),
                a.target_week, a.risk_level, int(a.acknowledged), 0,
                json.dumps(a.model_dump(), default=str),
            )
            for a in alerts
        ]
        if not rows:
            return 0
        with self._lock, self.connect() as connection:
            connection.executemany("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def get_alerts(
        self,
        disease: Optional[str] = None,
        district: Optional[str] = None,
        risk_level: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Alert]:
        clauses, params = [], []
        for column, value in (("disease", disease), ("district", district), ("risk_level", risk_level)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if acknowledged is not None:
            clauses.append("acknowledged = ?")
            params.append(int(acknowledged))
        if since is not None:
            clauses.append("issued_at >= ?")
            params.append(since.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT payload FROM alerts {where} ORDER BY issued_at DESC LIMIT ?", params
            ).fetchall()
        return [Alert.model_validate(json.loads(r["payload"])) for r in rows]

    def acknowledge_alert(self, alert_id: str, by: str) -> bool:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            if not row:
                return False
            payload = json.loads(row["payload"])
            payload["acknowledged"] = True
            payload["acknowledged_by"] = by
            connection.execute(
                "UPDATE alerts SET acknowledged = 1, synced = 0, payload = ? WHERE alert_id = ?",
                (json.dumps(payload, default=str), alert_id),
            )
        return True

    def unsynced_alerts(self) -> List[Alert]:
        with self.connect() as connection:
            rows = connection.execute("SELECT payload FROM alerts WHERE synced = 0").fetchall()
        return [Alert.model_validate(json.loads(r["payload"])) for r in rows]

    def mark_synced(self, table: str, ids: Sequence[str]) -> int:
        if not ids or table not in ("alerts", "interventions"):
            return 0
        key = "alert_id" if table == "alerts" else "intervention_id"
        placeholders = ",".join("?" * len(ids))
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE {table} SET synced = 1 WHERE {key} IN ({placeholders})", list(ids)
            )
        return cursor.rowcount

    # -- interventions -----------------------------------------------------
    def save_intervention(self, intervention: Intervention) -> str:
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO interventions VALUES (?,?,?,?,?,?,?)",
                (
                    intervention.intervention_id, intervention.disease, intervention.district,
                    intervention.alert_id, intervention.started_week, 0,
                    json.dumps(intervention.model_dump(), default=str),
                ),
            )
        return intervention.intervention_id

    def get_interventions(
        self, disease: Optional[str] = None, district: Optional[str] = None, limit: int = 200
    ) -> List[Intervention]:
        clauses, params = [], []
        for column, value in (("disease", disease), ("district", district)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT payload FROM interventions {where} ORDER BY started_week DESC LIMIT ?",
                params,
            ).fetchall()
        return [Intervention.model_validate(json.loads(r["payload"])) for r in rows]

    def unsynced_interventions(self) -> List[Intervention]:
        with self.connect() as connection:
            rows = connection.execute("SELECT payload FROM interventions WHERE synced = 0").fetchall()
        return [Intervention.model_validate(json.loads(r["payload"])) for r in rows]

    # -- observations ------------------------------------------------------
    def save_observations(self, rows: Sequence[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        payload = [
            (r["disease"], r["district"], r["week"], r.get("cases"),
             r.get("incidence_per_1000"), r.get("quality", 1.0))
            for r in rows
        ]
        with self._lock, self.connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?)", payload
            )
        return len(payload)

    def get_observations(
        self, disease: str, district: Optional[str] = None, limit: int = 520
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM observations WHERE disease = ?"
        params: List[Any] = [disease]
        if district:
            query += " AND district = ?"
            params.append(district)
        query += " ORDER BY week DESC LIMIT ?"
        params.append(limit)
        with self.connect() as connection:
            return [dict(r) for r in connection.execute(query, params).fetchall()]

    # -- metadata ----------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?,?)",
                (key, json.dumps(value, default=str), datetime.utcnow().isoformat(timespec="seconds")),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # -- housekeeping ------------------------------------------------------
    def prune(self, keep_days: int = 120) -> Dict[str, int]:
        """Drop stale rows so the cache stays small on a low-spec device."""
        cutoff = (datetime.utcnow() - timedelta(days=keep_days)).isoformat()
        with self._lock, self.connect() as connection:
            predictions = connection.execute(
                "DELETE FROM predictions WHERE created_at < ?", (cutoff,)
            ).rowcount
            alerts = connection.execute(
                "DELETE FROM alerts WHERE issued_at < ? AND synced = 1", (cutoff,)
            ).rowcount
        return {"predictions_removed": predictions, "alerts_removed": alerts}

    def status(self) -> Dict[str, Any]:
        with self.connect() as connection:
            counts = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
                )
                for table in ("predictions", "alerts", "interventions", "observations")
            }
            latest = connection.execute(
                "SELECT MAX(target_week) w FROM predictions"
            ).fetchone()["w"]
        weeks = self.weeks_cached()
        return {
            "path": str(self.path),
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
            **counts,
            "weeks_cached": weeks,
            "latest_target_week": latest,
            # Acceptance criterion #12: at least two weeks held locally.
            "offline_ready": weeks >= 2,
            "unsynced_alerts": len(self.unsynced_alerts()),
            "unsynced_interventions": len(self.unsynced_interventions()),
        }
