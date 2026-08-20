#!/usr/bin/env python3
"""Initialise storage for a deployment.

    python scripts/setup_database.py                 # SQLite (default, no server)
    python scripts/setup_database.py --check         # report on existing storage
    python scripts/setup_database.py --postgres      # also set up TimescaleDB
    python scripts/setup_database.py --reset --yes   # wipe and recreate

SQLite is the default and is sufficient for a single district node: no server to
run, no ops burden, and the file can be copied onto a USB stick and carried to a
district office. PostgreSQL + TimescaleDB is for the central instance holding
national history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.settings import get_settings  # noqa: E402
from src.core.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("scripts.setup_db")

TIMESCALE_SQL = """
-- Central-instance schema. TimescaleDB turns the two high-volume tables into
-- hypertables partitioned by time, which is what keeps national-scale queries
-- fast once several years of district x week history accumulate.
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- `week_start` must be part of the primary key: TimescaleDB refuses to create a
-- hypertable when a unique index omits the partitioning column. Without it,
-- create_hypertable() fails and the whole schema silently falls back to plain
-- tables - which is exactly what happened, unnoticed, until a real deployment.
-- It does not weaken the constraint: `week_start` is functionally determined by
-- `week`, so (disease, district, week) is still effectively unique.
CREATE TABLE IF NOT EXISTS observations (
    disease            TEXT        NOT NULL,
    district           TEXT        NOT NULL,
    week               TEXT        NOT NULL,
    week_start         DATE        NOT NULL,
    cases              DOUBLE PRECISION,
    incidence_per_1000 DOUBLE PRECISION,
    quality            DOUBLE PRECISION DEFAULT 1.0,
    source             TEXT,
    PRIMARY KEY (disease, district, week, week_start)
);

CREATE TABLE IF NOT EXISTS driver_values (
    source     TEXT             NOT NULL,
    variable   TEXT             NOT NULL,
    district   TEXT             NOT NULL,
    week       TEXT             NOT NULL,
    week_start DATE             NOT NULL,
    value      DOUBLE PRECISION,
    quality    DOUBLE PRECISION DEFAULT 1.0,
    imputed    BOOLEAN          DEFAULT FALSE,
    PRIMARY KEY (source, variable, district, week, week_start)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   TEXT PRIMARY KEY,
    disease         TEXT NOT NULL,
    district        TEXT NOT NULL,
    region          TEXT,
    forecast_date   DATE NOT NULL,
    target_week     TEXT NOT NULL,
    predicted_cases DOUBLE PRECISION,
    ci_lower        DOUBLE PRECISION,
    ci_upper        DOUBLE PRECISION,
    risk_level      TEXT,
    risk_score      DOUBLE PRECISION,
    model_version   TEXT,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id      TEXT PRIMARY KEY,
    disease       TEXT NOT NULL,
    district      TEXT NOT NULL,
    region        TEXT,
    issued_at     TIMESTAMPTZ NOT NULL,
    target_week   TEXT NOT NULL,
    risk_level    TEXT,
    acknowledged  BOOLEAN DEFAULT FALSE,
    payload       JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS interventions (
    intervention_id   TEXT PRIMARY KEY,
    disease           TEXT NOT NULL,
    district          TEXT NOT NULL,
    alert_id          TEXT,
    intervention_type TEXT NOT NULL,
    started_week      TEXT NOT NULL,
    coverage          DOUBLE PRECISION,
    payload           JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id        TEXT PRIMARY KEY,
    disease       TEXT NOT NULL,
    scope         TEXT NOT NULL,
    trained_at    TIMESTAMPTZ NOT NULL,
    backend       TEXT,
    n_rows        INTEGER,
    holdout_mae   DOUBLE PRECISION,
    promoted      BOOLEAN,
    metadata      JSONB
);

SELECT create_hypertable('observations',  'week_start', if_not_exists => TRUE);
SELECT create_hypertable('driver_values', 'week_start', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_obs_lookup   ON observations (disease, district, week_start DESC);
CREATE INDEX IF NOT EXISTS idx_driver_lookup ON driver_values (variable, district, week_start DESC);
CREATE INDEX IF NOT EXISTS idx_pred_lookup  ON predictions (disease, district, target_week);
CREATE INDEX IF NOT EXISTS idx_alert_lookup ON alerts (disease, district, issued_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_open   ON alerts (acknowledged, issued_at DESC);
"""


def setup_sqlite(reset: bool) -> int:
    from offline.local_cache import LocalCache

    settings = get_settings()
    path = settings.sqlite_path
    if reset and path.exists():
        path.unlink()
        print(f"  removed {path}")

    cache = LocalCache(path)
    status = cache.status()
    print(f"  SQLite store ready at {status['path']}")
    print(f"  tables: predictions, alerts, interventions, observations, meta")
    print(f"  size  : {status['size_bytes'] / 1024:.1f} KiB")
    return 0


def setup_postgres(reset: bool) -> int:
    settings = get_settings()
    url = settings.database_url
    if not url:
        print("  DATABASE_URL is not set - nothing to do.")
        print("  Set it in .env, e.g.")
        print("    DATABASE_URL=postgresql://afya:secret@localhost:5432/afya_predict")
        return 1
    try:
        import psycopg
    except ImportError:
        print('  psycopg is not installed. Install the extra:  pip install -e ".[db]"')
        return 1

    print(f"  connecting to {url.split('@')[-1]}")
    with psycopg.connect(url) as connection:
        with connection.cursor() as cursor:
            if reset:
                for table in ("model_runs", "interventions", "alerts", "predictions",
                              "driver_values", "observations"):
                    cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print("  dropped existing tables")
            hypertables = True
            try:
                cursor.execute(TIMESCALE_SQL)
            except Exception as exc:  # noqa: BLE001
                # TimescaleDB is an optimisation, not a requirement - but say so
                # loudly rather than reporting success for a degraded schema.
                print(f"  TimescaleDB unavailable ({exc});")
                print("  creating plain tables instead - queries will still work, but")
                print("  time-partitioning is lost. Check that the extension is installed.")
                hypertables = False
                connection.rollback()
                plain = "\n".join(
                    line for line in TIMESCALE_SQL.splitlines()
                    if "create_hypertable" not in line and "CREATE EXTENSION" not in line
                )
                cursor.execute(plain)
        connection.commit()

    if hypertables:
        with psycopg.connect(url) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT hypertable_name FROM timescaledb_information.hypertables"
            )
            names = sorted(row[0] for row in cursor.fetchall())
        print(f"  schema created with {len(names)} hypertable(s): {', '.join(names) or 'none'}")
        if not names:
            print("  WARNING: no hypertables exist despite the extension loading.")
            return 1
    else:
        print("  schema created (plain tables)")
    return 0


def check() -> int:
    from offline.local_cache import LocalCache

    settings = get_settings()
    print("Storage configuration")
    print(f"  data dir        : {settings.data_dir}")
    print(f"  artifact dir    : {settings.artifact_dir}")
    print(f"  effective DB URL: {settings.effective_database_url}")
    print(f"  redis           : {settings.redis_url or 'not configured (optional)'}")

    path = settings.sqlite_path
    if not path.exists():
        print(f"\n  SQLite store not created yet - run this script without --check")
        return 1

    status = LocalCache(path).status()
    print(f"\nLocal SQLite store")
    for key in ("predictions", "alerts", "interventions", "observations",
                "weeks_cached", "latest_target_week", "offline_ready",
                "unsynced_alerts", "unsynced_interventions"):
        print(f"  {key:24}: {status[key]}")
    print(f"  {'size_kib':24}: {status['size_bytes'] / 1024:.1f}")

    if not status["offline_ready"]:
        print("\n  NOTE: fewer than 2 weeks of forecasts cached, so this node is not")
        print("        yet ready to run offline (acceptance criterion #12).")
        print("        Run: python scripts/seed_historical_data.py --predict")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialise AFYA-PREDICT storage")
    parser.add_argument("--check", action="store_true", help="Report on storage and exit")
    parser.add_argument("--postgres", action="store_true",
                        help="Also create the PostgreSQL/TimescaleDB schema")
    parser.add_argument("--sqlite-only", action="store_true", help="Only set up SQLite")
    parser.add_argument("--reset", action="store_true", help="Drop existing data first")
    parser.add_argument("--yes", action="store_true", help="Skip the --reset confirmation")
    args = parser.parse_args()

    configure_logging()

    if args.check:
        return check()

    if args.reset and not args.yes:
        settings = get_settings()
        print("This will DELETE all cached predictions, alerts and interventions at")
        print(f"  {settings.sqlite_path}")
        if args.postgres:
            print("and DROP every table in the configured PostgreSQL database.")
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            print("aborted")
            return 1

    print("SQLite (offline node + API cache)")
    code = setup_sqlite(args.reset)

    if args.postgres and not args.sqlite_only:
        print("\nPostgreSQL / TimescaleDB (central instance)")
        code |= setup_postgres(args.reset)

    print("\nNext: python scripts/seed_historical_data.py")
    return code


if __name__ == "__main__":
    sys.exit(main())
