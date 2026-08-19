"""Automated data-fetch scheduling.

Each source declares its own `update_frequency_days`, so the scheduler polls
CHIRPS every 5 days, DHIS2 weekly and WorldPop annually rather than hammering
everything on one cron line. Runs are recorded so `/data/status` can report
freshness, and failures never stop the loop — the affected source simply serves
cache or synthetic data on the next pipeline run (rule #1).

Run standalone with:

    python -m src.data_ingestion.scheduler
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import Settings, get_settings
from src.core.config_loader import cached_region_config, load_all_disease_configs
from src.core.logging import get_logger
from src.core.timeutils import shift_week, to_epi_week
from src.core.types import RegionConfig
from src.data_ingestion.registry import ADAPTER_REGISTRY, available_sources, get_adapter

log = get_logger("ingest.scheduler")


@dataclass
class RunRecord:
    """Outcome of one adapter run, persisted for the data-status endpoint."""

    source: str
    started_at: str
    finished_at: str
    mode: str
    rows: int
    mean_quality: float
    latest_data_date: Optional[str]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class IngestionScheduler:
    """Poll every configured source on its own cadence."""

    def __init__(
        self,
        region: Optional[RegionConfig] = None,
        settings: Optional[Settings] = None,
        lookback_weeks: int = 8,
        sources: Optional[List[str]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.region = region or cached_region_config(self.settings.default_region)
        self.lookback_weeks = lookback_weeks
        self.sources = sources or self._sources_in_use()
        self.state_path = Path(self.settings.data_dir) / "ingestion_state.json"
        self._stop = False

    # -- source selection --------------------------------------------------
    def _sources_in_use(self) -> List[str]:
        """Every source referenced by at least one disease config, plus DHIS2."""
        wanted = {"dhis2"}
        for config in load_all_disease_configs().values():
            wanted.update(config.all_sources)
        known = set(ADAPTER_REGISTRY)
        unknown = wanted - known
        if unknown:
            log.warning("disease configs reference unknown sources: %s", ", ".join(sorted(unknown)))
        return sorted(wanted & known)

    # -- state -------------------------------------------------------------
    def load_state(self) -> Dict[str, dict]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read ingestion state: %s", exc)
            return {}

    def save_state(self, state: Dict[str, dict]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str))

    def due_sources(self, now: Optional[datetime] = None) -> List[str]:
        """Sources whose update interval has elapsed since their last success."""
        now = now or datetime.utcnow()
        state = self.load_state()
        due = []
        for source in self.sources:
            record = state.get(source)
            if not record or record.get("error"):
                due.append(source)
                continue
            try:
                last = datetime.fromisoformat(record["finished_at"])
            except (KeyError, ValueError):
                due.append(source)
                continue
            interval = timedelta(days=ADAPTER_REGISTRY[source].update_frequency_days)
            if now - last >= interval:
                due.append(source)
        return due

    # -- running -----------------------------------------------------------
    def run_source(self, source: str, end_week: Optional[str] = None) -> RunRecord:
        """Fetch the trailing window for one source and cache the result."""
        end_week = end_week or to_epi_week(date.today())
        start_week = shift_week(end_week, -(self.lookback_weeks - 1))
        started = datetime.utcnow()
        try:
            adapter = get_adapter(source, region=self.region, settings=self.settings)
            result = adapter.run(start_week, end_week)
            # Persist even synthetic/cached output so offline nodes stay warm.
            adapter.write_cache(result.frame)
            record = RunRecord(
                source=source,
                started_at=started.isoformat(timespec="seconds"),
                finished_at=datetime.utcnow().isoformat(timespec="seconds"),
                mode=result.mode.value,
                rows=0 if result.is_empty else len(result.frame),
                mean_quality=round(result.mean_quality, 3),
                latest_data_date=str(result.latest_data_date) if result.latest_data_date else None,
            )
        except Exception as exc:  # noqa: BLE001 - a broken source must not stop the loop
            log.exception("ingestion failed for %s", source)
            record = RunRecord(
                source=source,
                started_at=started.isoformat(timespec="seconds"),
                finished_at=datetime.utcnow().isoformat(timespec="seconds"),
                mode="error",
                rows=0,
                mean_quality=0.0,
                latest_data_date=None,
                error=str(exc),
            )
        state = self.load_state()
        state[source] = asdict(record)
        self.save_state(state)
        return record

    def run_due(self, end_week: Optional[str] = None) -> List[RunRecord]:
        due = self.due_sources()
        if not due:
            log.info("no sources due")
            return []
        log.info("running %d due source(s): %s", len(due), ", ".join(due))
        return [self.run_source(source, end_week) for source in due]

    def run_all(self, end_week: Optional[str] = None) -> List[RunRecord]:
        return [self.run_source(source, end_week) for source in self.sources]

    # -- status ------------------------------------------------------------
    def status(self) -> List[dict]:
        """Freshness/quality report used by `GET /data/status`."""
        state = self.load_state()
        now = datetime.utcnow()
        rows = []
        for source in available_sources():
            cls = ADAPTER_REGISTRY[source]
            record = state.get(source, {})
            finished = record.get("finished_at")
            age_hours = None
            if finished:
                try:
                    age_hours = round((now - datetime.fromisoformat(finished)).total_seconds() / 3600, 1)
                except ValueError:
                    age_hours = None
            interval_hours = cls.update_frequency_days * 24
            rows.append(
                {
                    "source": source,
                    "in_use": source in self.sources,
                    "optional": cls.optional,
                    "update_frequency_days": cls.update_frequency_days,
                    "last_run": finished,
                    "age_hours": age_hours,
                    "stale": age_hours is None or age_hours > interval_hours * 2,
                    "mode": record.get("mode"),
                    "rows": record.get("rows", 0),
                    "mean_quality": record.get("mean_quality", 0.0),
                    "latest_data_date": record.get("latest_data_date"),
                    "error": record.get("error"),
                }
            )
        return rows

    # -- daemon ------------------------------------------------------------
    def start(self, poll_seconds: int = 3600, use_apscheduler: bool = True) -> None:
        """Block and poll. Uses APScheduler when installed, else a sleep loop."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        if use_apscheduler:
            try:
                from apscheduler.schedulers.background import BackgroundScheduler
            except ImportError:
                log.info("APScheduler not installed; falling back to a simple sleep loop")
            else:
                scheduler = BackgroundScheduler(timezone="UTC")
                for source in self.sources:
                    hours = max(1, ADAPTER_REGISTRY[source].update_frequency_days * 24 // 2)
                    scheduler.add_job(
                        self.run_source,
                        "interval",
                        hours=hours,
                        args=[source],
                        id=f"ingest-{source}",
                        max_instances=1,
                        coalesce=True,
                        next_run_time=datetime.utcnow() + timedelta(seconds=5),
                    )
                scheduler.start()
                log.info("scheduler started for %d source(s)", len(self.sources))
                try:
                    while not self._stop:
                        time.sleep(1)
                finally:
                    scheduler.shutdown(wait=False)
                return

        log.info("polling every %ds for %d source(s)", poll_seconds, len(self.sources))
        while not self._stop:
            self.run_due()
            for _ in range(poll_seconds):
                if self._stop:
                    break
                time.sleep(1)

    def _handle_signal(self, signum, _frame) -> None:  # pragma: no cover - signal path
        log.info("received signal %s; shutting down", signum)
        self._stop = True


def main() -> None:  # pragma: no cover - process entry point
    import argparse

    parser = argparse.ArgumentParser(description="AFYA-PREDICT ingestion scheduler")
    parser.add_argument("--once", action="store_true", help="run all due sources once and exit")
    parser.add_argument("--all", action="store_true", help="force-run every source once and exit")
    parser.add_argument("--status", action="store_true", help="print source freshness and exit")
    parser.add_argument("--lookback-weeks", type=int, default=8)
    args = parser.parse_args()

    scheduler = IngestionScheduler(lookback_weeks=args.lookback_weeks)
    if args.status:
        print(json.dumps(scheduler.status(), indent=2, default=str))
        return
    if args.all:
        records = scheduler.run_all()
    elif args.once:
        records = scheduler.run_due()
    else:
        scheduler.start()
        return
    print(json.dumps([asdict(r) for r in records], indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
