#!/usr/bin/env python3
"""Populate a fresh install with history, trained models and forecasts.

    python scripts/seed_historical_data.py                    # everything, all diseases
    python scripts/seed_historical_data.py --disease malaria --districts Kinondoni Ilala
    python scripts/seed_historical_data.py --no-train         # ingest and cache only

This is what turns a clean checkout into a working node. It:

1. ingests the history each disease needs and caches it to disk,
2. writes observed case counts into the local store so impact estimation has a
   baseline to compare against,
3. trains and saves a model per disease,
4. generates forecasts and alerts and caches them, so the API serves reads
   without touching a model, and so the node can survive a disconnection
   (acceptance criterion #12).

With no credentials configured every source falls back to synthetic data. The
run says so, loudly, at the end.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.config_loader import cached_region_config  # noqa: E402
from src.core.geo import subset_region  # noqa: E402
from src.core.logging import configure_logging, get_logger  # noqa: E402
from src.core.timeutils import shift_week, to_epi_week  # noqa: E402

log = get_logger("scripts.seed")


def seed_disease(
    slug: str,
    region,
    cache,
    history_weeks: int,
    end_week: str,
    train: bool,
    predict: bool,
    forecast_weeks: int = 4,
) -> dict:
    from src.data_ingestion.normalizer import ingest
    from src.models.registry import build_module

    started = time.perf_counter()
    module = build_module(slug, region=region)
    sources = sorted(set(module.config.required_sources) | {"dhis2"})
    start_week = shift_week(end_week, -history_weeks)

    print(f"\n  {module.config.name} ({module.config.code})")
    print(f"    ingesting {len(sources)} source(s) over {history_weeks} weeks "
          f"({start_week} .. {end_week})")
    panel = ingest(sources, start_week, end_week, region=region)

    synthetic = sorted(s for s, mode in panel.modes.items() if mode == "synthetic")
    live = sorted(s for s, mode in panel.modes.items() if mode == "live")
    print(f"    live: {len(live)}  cached/synthetic: {len(synthetic)}  "
          f"mean quality: {panel.mean_quality():.2f}")

    # Store observed cases so the impact estimator has a real baseline.
    target = module.target_column
    observations: List[dict] = []
    if target in panel.value_columns:
        populations = {d.name: float(d.population) for d in region.districts}
        series = panel.values()[target]
        for (district, week), value in series.items():
            if value != value:      # NaN: never fabricate a case count
                continue
            population = populations.get(district, 1.0)
            observations.append({
                "disease": slug, "district": district, "week": week,
                "cases": float(value),
                "incidence_per_1000": float(value) / max(population, 1) * 1000.0,
                "quality": 1.0,
            })
        cache.save_observations(observations)
        print(f"    cached {len(observations)} observed district-weeks")

    result = {
        "disease": slug, "sources": len(sources), "live_sources": len(live),
        "synthetic_sources": synthetic, "observations": len(observations),
        "trained": False, "predictions": 0, "alerts": 0,
    }

    if not train:
        result["duration_seconds"] = round(time.perf_counter() - started, 1)
        return result

    matrix = module.build_feature_matrix(panel)
    print(f"    built {matrix.X.shape[1]} features over {matrix.X.shape[0]} district-weeks")
    module.train(matrix)
    path = module.save()
    district_models = max(len(module.models) - 1, 0)
    print(f"    trained pooled + {district_models} district model(s) -> "
          f"{path.relative_to(REPO_ROOT)}")
    result["trained"] = True
    result["district_models"] = district_models

    if predict:
        # Forecast several consecutive weeks, not just the latest one: an offline
        # node needs a rolling horizon to keep serving through a disconnection
        # (acceptance criterion #12 requires at least two weeks cached).
        predictions = module.predict_all(matrix, panel=panel, n_weeks=forecast_weeks)
        alerts = module.detect_outbreak(predictions)
        cache.save_predictions(predictions)
        cache.save_alerts(alerts)
        result["predictions"] = len(predictions)
        result["alerts"] = len(alerts)
        by_level: dict = {}
        for prediction in predictions:
            by_level[prediction.risk_level] = by_level.get(prediction.risk_level, 0) + 1
        print(f"    forecast {len(predictions)} district-week(s): "
              + ", ".join(f"{v} {k}" for k, v in sorted(by_level.items())))
        print(f"    raised {len(alerts)} alert(s)")

    result["duration_seconds"] = round(time.perf_counter() - started, 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed a fresh AFYA-PREDICT install")
    parser.add_argument("--disease", help="Seed only this disease")
    parser.add_argument("--districts", nargs="*", help="Restrict to these districts")
    parser.add_argument("--history-weeks", type=int, default=260,
                        help="Weeks of history to ingest (default 260 = 5 years)")
    parser.add_argument("--end-week", help="Anchor week, e.g. 2024-W52")
    parser.add_argument("--no-train", action="store_true", help="Ingest and cache only")
    parser.add_argument("--no-predict", action="store_true", help="Train but do not forecast")
    parser.add_argument("--forecast-weeks", type=int, default=4,
                        help="Consecutive weeks to forecast per district (default 4). "
                             "At least 2 are needed for offline readiness.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    configure_logging("WARNING" if args.quiet else None)

    from offline.local_cache import LocalCache
    from src.models.registry import list_modules

    region = cached_region_config()
    if args.districts:
        region = subset_region(region, args.districts)

    end_week = args.end_week or to_epi_week(date.today())
    diseases = [args.disease] if args.disease else list_modules()
    cache = LocalCache()

    print("Seeding AFYA-PREDICT")
    print(f"  region    : {region.name} ({len(region.districts)} districts)")
    print(f"  diseases  : {', '.join(diseases)}")
    print(f"  anchor    : {end_week}")
    print(f"  store     : {cache.path}")

    started = time.perf_counter()
    results = []
    for slug in diseases:
        try:
            results.append(seed_disease(
                slug, region, cache, args.history_weeks, end_week,
                train=not args.no_train, predict=not (args.no_train or args.no_predict),
                forecast_weeks=args.forecast_weeks,
            ))
        except Exception as exc:  # noqa: BLE001 - one disease must not stop the seed
            log.exception("seeding failed for %s", slug)
            results.append({"disease": slug, "error": str(exc)})

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for result in results:
        if "error" in result:
            print(f"  FAILED  {result['disease']:16} {result['error']}")
            continue
        print(f"  ok      {result['disease']:16} "
              f"{result['observations']:>6} obs  "
              f"{result['predictions']:>4} forecasts  "
              f"{result['alerts']:>3} alerts  "
              f"({result['duration_seconds']}s)")

    status = cache.status()
    print(f"\nLocal store: {status['predictions']} predictions, {status['alerts']} alerts, "
          f"{status['observations']} observations")
    print(f"Offline ready ({status['weeks_cached']} week(s) cached): {status['offline_ready']}")

    all_synthetic = {s for r in results for s in r.get("synthetic_sources", [])}
    if all_synthetic:
        print(f"\n  NOTE: {len(all_synthetic)} source(s) served synthetic data:")
        print(f"        {', '.join(sorted(all_synthetic))}")
        print("        Everything above is demonstrative, not operational. Configure")
        print("        credentials in .env and re-run before using these forecasts.")

    print(f"\nDone in {time.perf_counter() - started:.1f}s. Start the API with:")
    print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
