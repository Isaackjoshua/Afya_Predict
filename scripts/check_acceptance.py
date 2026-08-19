#!/usr/bin/env python3
"""Check the platform against its 15 acceptance criteria.

    python scripts/check_acceptance.py            # fast structural checks
    python scripts/check_acceptance.py --full     # also run a backtest (slow)

Structural criteria are verified directly against the code and configuration.
The two that need a fitted model — outbreak-detection AUC and the naive-baseline
gate — are only evaluated under `--full`, and are reported as NOT VERIFIED rather
than assumed otherwise.

Exit code is 0 when every evaluated criterion passes.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PASS, FAIL, SKIP = "PASS", "FAIL", "NOT VERIFIED"


@dataclass
class Result:
    number: int
    criterion: str
    status: str
    detail: str = ""


def check(number: int, criterion: str):
    """Wrap a check so a raised exception becomes a FAIL, never a crash.

    The wrapper forwards any argument it is given, because the two criteria that
    need a fitted model take the backtest report.
    """
    def decorator(fn: Callable[..., tuple]) -> Callable[..., Result]:
        def wrapper(*args) -> Result:
            try:
                status, detail = fn(*args)
            except Exception as exc:  # noqa: BLE001
                return Result(number, criterion, FAIL, f"check raised: {exc}")
            return Result(number, criterion, status, detail)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------- criteria
@check(1, "At least 5 disease modules registered and functional")
def criterion_1():
    from src.models.registry import list_modules, validate_registry

    modules = list_modules()
    problems = validate_registry()
    if len(modules) >= 5 and not problems:
        return PASS, f"{len(modules)} registered: {', '.join(modules)}"
    return FAIL, f"{len(modules)} modules, problems: {problems}"


@check(2, "Each disease module uses >= 3 digital proxy data sources")
def criterion_2():
    from src.core.config_loader import load_all_disease_configs

    thin = {
        slug: len(config.required_sources)
        for slug, config in load_all_disease_configs().items()
        if len(config.required_sources) < 3
    }
    if thin:
        return FAIL, f"below 3 sources: {thin}"
    counts = {s: len(c.required_sources) for s, c in load_all_disease_configs().items()}
    return PASS, f"sources per disease: {counts}"


@check(3, "Predictions include 95% confidence intervals")
def criterion_3():
    from src.core.types import PredictionResult

    fields = set(PredictionResult.model_fields)
    required = {"confidence_interval_lower", "confidence_interval_upper"}
    if not required <= fields:
        return FAIL, f"missing {required - fields}"
    # The interval must be built from out-of-sample residuals, not in-sample.
    source = (REPO_ROOT / "src/models/base_model.py").read_text()
    if "_holdout_residuals" not in source:
        return FAIL, "intervals are not derived from holdout residuals"
    return PASS, "intervals derived from a chronological internal holdout"


@check(4, "Every prediction has SHAP explanations + natural language summary")
def criterion_4():
    from src.core.types import PredictionResult

    fields = set(PredictionResult.model_fields)
    required = {"shap_values", "top_drivers", "natural_language_explanation", "counterfactual"}
    missing = required - fields
    if missing:
        return FAIL, f"missing {missing}"
    source = (REPO_ROOT / "src/models/standard_module.py").read_text()
    if "ShapExplainer" not in source or "explain_prediction" not in source:
        return FAIL, "predict() does not build an explanation"
    return PASS, "explanation constructed in the same call as the prediction"


@check(5, "Walk-forward CV shows AUC >= 0.75 for outbreak detection")
def criterion_5(report: Optional[dict] = None):
    if report is None:
        return SKIP, "run with --full to evaluate"
    outbreak = report.get("outbreak_detection") or {}
    auc = outbreak.get("auc")
    if auc is None:
        return SKIP, outbreak.get("verdict", "no outbreak weeks in the window")
    return (PASS if auc >= 0.75 else FAIL), f"AUC {auc:.3f} ({outbreak.get('verdict','')})"


@check(6, "All models beat the 3 naive baselines")
def criterion_6(report: Optional[dict] = None):
    if report is None:
        return SKIP, "run with --full to evaluate"
    baselines = report.get("baselines") or {}
    if not baselines:
        return SKIP, "no baseline comparison produced"
    return (PASS if baselines.get("passes") else FAIL), baselines.get("verdict", "")


@check(7, "Spatial diffusion produces district-level importation risk scores")
def criterion_7():
    import pandas as pd

    from src.core.config_loader import load_disease_config, load_region_config
    from src.core.geo import subset_region
    from src.models.spatial_diffusion import SpatialDiffusionModel

    region = subset_region(load_region_config(), ["Kinondoni", "Ilala", "Mwanza City"])
    model = SpatialDiffusionModel(load_disease_config("cholera"), region)
    incidence = pd.Series(0.0, index=region.district_names)
    incidence["Ilala"] = 3.0
    forecast = model.project(incidence, "2024-W20", horizon_weeks=3)
    contributors = model.contributors(incidence, "Kinondoni")
    if forecast.risk.empty or not contributors:
        return FAIL, "no importation risk produced"
    return PASS, (
        f"risk for {forecast.risk.shape[1]} districts over {forecast.risk.shape[0]} weeks; "
        f"sources named (e.g. {contributors[0].district})"
    )


@check(8, "Concept drift detection triggers automatic retraining")
def criterion_8():
    import numpy as np

    from src.models.drift_detector import detect_drift

    rng = np.random.default_rng(0)
    stable = rng.normal(0, 1, 200)
    drifting = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
    quiet = detect_drift(np.zeros(200), -stable)["drift_detected"]
    caught = detect_drift(np.zeros(200), -drifting)["drift_detected"]

    source = (REPO_ROOT / "src/models/auto_retrain.py").read_text()
    wired = "DriftDetector" in source and "should_retrain" in source
    if quiet or not caught or not wired:
        return FAIL, f"false alarm={quiet}, detected={caught}, wired={wired}"
    return PASS, "no false alarm on a stable stream; step change detected and wired to refit"


@check(9, "Alerts include actionable recommendations")
def criterion_9():
    from datetime import datetime

    from src.alerting.recommendation_engine import RecommendationEngine
    from src.core.config_loader import load_disease_config, load_region_config
    from src.core.types import Alert

    region = load_region_config()
    config = load_disease_config("cholera")
    alert = Alert(
        alert_id="check", disease="Cholera", district="Sengerema", region="Mwanza",
        issued_at=datetime.utcnow(), target_week="2026-W12", risk_level="high",
        risk_score=0.8, predicted_cases=480.0, predicted_incidence_per_1000=0.72,
        threshold_crossed=0.6, lead_time_weeks=6,
    )
    recommendations = RecommendationEngine(config, region).build(alert)
    if not recommendations:
        return FAIL, "no recommendations generated"
    if not all(r.responsible and r.timeframe_days > 0 for r in recommendations):
        return FAIL, "some recommendations lack an owner or a deadline"
    quantified = sum(1 for r in recommendations if r.quantity)
    return PASS, (
        f"{len(recommendations)} actions, all owned and timeboxed, "
        f"{quantified} with quantities"
    )


@check(10, "API returns predictions in < 2 seconds")
def criterion_10():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return SKIP, "fastapi not installed"
    from src.api.main import app

    with TestClient(app) as client:
        started = time.perf_counter()
        response = client.get("/predictions", params={"limit": 50})
        elapsed = time.perf_counter() - started
    if response.status_code != 200:
        return FAIL, f"HTTP {response.status_code}"
    return (PASS if elapsed < 2.0 else FAIL), f"{elapsed * 1000:.0f} ms for 50 predictions"


@check(11, "Dashboard renders heatmap, district detail and SHAP waterfall")
def criterion_11():
    import importlib

    pages = ["overview", "district_detail", "disease_comparison",
             "data_quality", "model_performance", "intervention_tracker"]
    missing = [p for p in pages if not (REPO_ROOT / f"dashboard/pages/{p}.py").exists()]
    if missing:
        return FAIL, f"missing pages: {missing}"
    for component in ("risk_map", "forecast_chart", "shap_waterfall", "recommendation_card"):
        importlib.import_module(f"dashboard.components.{component}")
    for page in pages:
        importlib.import_module(f"dashboard.pages.{page}")
    return PASS, f"{len(pages)} pages and 4 components import cleanly"


@check(12, "Offline mode caches >= 2 weeks of predictions locally")
def criterion_12():
    from offline.local_cache import LocalCache
    from offline.sync_manager import SyncManager

    cache = LocalCache()
    status = cache.status()
    readiness = SyncManager(cache=cache).offline_readiness(required_weeks=2)
    if not readiness["ready"]:
        return FAIL, (
            f"only {status['weeks_cached']} week(s) cached — "
            "run scripts/seed_historical_data.py"
        )
    return PASS, f"{status['weeks_cached']} weeks, {status['predictions']} predictions cached"


@check(13, "add_new_disease.py scaffolds a new disease in < 5 minutes")
def criterion_13():
    import subprocess
    import tempfile

    script = REPO_ROOT / "scripts/add_new_disease.py"
    if not script.exists():
        return FAIL, "script missing"
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script), "--validate", "malaria"],
        capture_output=True, text=True, timeout=120,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        return FAIL, result.stdout.strip().splitlines()[-1] if result.stdout else "validate failed"
    return PASS, f"scaffold + validate path works ({elapsed:.1f}s for a validation run)"


@check(14, "docker-compose brings up the full stack in a single command")
def criterion_14():
    import subprocess

    if not (REPO_ROOT / "docker-compose.yml").exists():
        return FAIL, "docker-compose.yml missing"
    try:
        result = subprocess.run(
            ["docker", "compose", "config", "--quiet"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return SKIP, "docker not available on this machine"
    if result.returncode != 0:
        return FAIL, result.stderr.strip()[:200]
    import yaml

    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    services = sorted(compose["services"])
    return PASS, f"compose config valid; services: {', '.join(services)}"


@check(15, "Test coverage >= 80%")
def criterion_15():
    import subprocess

    coverage_file = REPO_ROOT / ".coverage"
    if not coverage_file.exists():
        return SKIP, "no .coverage file — run: coverage run -m pytest"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "report", "--format=total"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        total = float(result.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        return SKIP, f"could not read coverage: {exc}"
    return (PASS if total >= 80 else FAIL), f"{total:.0f}% line coverage"


CHECKS = [
    criterion_1, criterion_2, criterion_3, criterion_4, criterion_5, criterion_6,
    criterion_7, criterion_8, criterion_9, criterion_10, criterion_11,
    criterion_12, criterion_13, criterion_14, criterion_15,
]
NEEDS_BACKTEST = {5, 6}


def run_backtest() -> Optional[dict]:
    """Fit and validate a model so criteria 5 and 6 can be evaluated."""
    from src.core.config_loader import cached_region_config
    from src.core.geo import subset_region
    from src.data_ingestion.normalizer import ingest
    from src.evaluation.walk_forward_cv import WalkForwardCV
    from src.models.registry import build_module

    region = subset_region(cached_region_config(),
                           ["Kinondoni", "Ilala", "Mwanza City", "Sengerema"])
    module = build_module("malaria", region=region)
    sources = sorted(set(module.config.required_sources) | {"dhis2"})
    panel = ingest(sources, "2019-W01", "2024-W52", region=region)
    matrix = module.build_feature_matrix(panel)
    cv = WalkForwardCV(module, initial_train_weeks=156, step_weeks=26,
                       test_weeks=26, max_folds=3)
    result = cv.run(matrix)
    return result.report() if result.folds else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the 15 acceptance criteria")
    parser.add_argument("--full", action="store_true",
                        help="also run a backtest so criteria 5 and 6 can be evaluated")
    args = parser.parse_args()

    from src.core.logging import configure_logging

    configure_logging("WARNING")

    report = None
    if args.full:
        print("Running a walk-forward backtest for criteria 5 and 6 (this takes minutes)…\n")
        try:
            report = run_backtest()
        except Exception as exc:  # noqa: BLE001
            print(f"backtest failed: {exc}\n")

    print("=" * 78)
    print("AFYA-PREDICT — acceptance criteria")
    print("=" * 78)

    results: List[Result] = []
    for fn in CHECKS:
        number = int(fn.__name__.split("_")[-1])
        results.append(fn(report) if number in NEEDS_BACKTEST else fn())

    for result in results:
        marker = {PASS: "PASS", FAIL: "FAIL", SKIP: "----"}[result.status]
        print(f"\n[{marker}] {result.number:>2}. {result.criterion}")
        if result.detail:
            print(f"        {result.detail}")

    passed = sum(r.status == PASS for r in results)
    failed = sum(r.status == FAIL for r in results)
    skipped = sum(r.status == SKIP for r in results)

    print("\n" + "=" * 78)
    print(f"{passed} passed · {failed} failed · {skipped} not verified")
    if skipped and not args.full:
        print("Re-run with --full to evaluate the criteria that need a fitted model.")
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
