#!/usr/bin/env python3
"""Walk-forward backtest and deployment gate (shortcoming #13, critical rule #10).

    python scripts/run_backtest.py --disease malaria
    python scripts/run_backtest.py --all --districts Kinondoni "Mwanza City"
    python scripts/run_backtest.py --disease cholera --json artifacts/cholera_backtest.json

Exit codes are meaningful, so this can gate a release in CI:

    0  every evaluated disease passed
    1  at least one disease failed the acceptance criteria
    2  the backtest could not run (no data, no usable folds)
"""

from __future__ import annotations

import argparse
import json
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

log = get_logger("scripts.backtest")


def run_one(
    slug: str,
    history_weeks: int,
    districts: Optional[List[str]],
    initial_train_weeks: int,
    step_weeks: int,
    test_weeks: int,
    max_folds: int,
    end_week: Optional[str],
) -> dict:
    from src.data_ingestion.normalizer import ingest
    from src.evaluation.walk_forward_cv import WalkForwardCV
    from src.models.registry import build_module

    region = cached_region_config()
    if districts:
        region = subset_region(region, districts)

    module = build_module(slug, region=region)
    end = end_week or to_epi_week(date.today())
    start = shift_week(end, -history_weeks)
    sources = sorted(set(module.config.required_sources) | {"dhis2"})

    print(f"\n{'=' * 74}")
    print(f"{module.config.name} ({module.config.code})")
    print(f"{'=' * 74}")
    print(f"  window     : {start} .. {end} ({history_weeks} weeks)")
    print(f"  districts  : {len(region.districts)}")
    print(f"  sources    : {', '.join(sources)}")
    print(f"  horizon    : {module.horizon} weeks")

    started = time.perf_counter()
    panel = ingest(sources, start, end, region=region)
    synthetic = [s for s, mode in panel.modes.items() if mode == "synthetic"]
    if synthetic:
        print(f"\n  WARNING: {len(synthetic)} source(s) served synthetic data "
              f"({', '.join(sorted(synthetic))}).")
        print("           Results below are demonstrative, not operational.")

    matrix = module.build_feature_matrix(panel)
    cv = WalkForwardCV(module, initial_train_weeks=initial_train_weeks,
                       step_weeks=step_weeks, test_weeks=test_weeks, max_folds=max_folds)
    result = cv.run(matrix)
    elapsed = time.perf_counter() - started

    if not result.folds:
        print(f"\n  ERROR: no usable folds. Need at least "
              f"{initial_train_weeks + module.horizon + test_weeks} weeks of history.")
        return {"disease": slug, "error": "no usable folds", "passes_acceptance": False}

    report = result.report()
    accuracy = report["accuracy"]
    print(f"\n  ACCURACY (out of sample, {report['folds']} folds, "
          f"{report['n_predictions']} predictions)")
    print(f"    MAE   {accuracy['mae']:>10}    RMSE  {accuracy['rmse']:>10}")
    print(f"    R2    {accuracy['r2']:>10}    bias  {accuracy['bias']:>10}")
    print(f"    MASE  {accuracy['mase']:>10}  (< 1 beats a naive one-step forecast)")

    intervals = report["intervals"]
    print(f"\n  INTERVALS")
    print(f"    coverage {intervals['coverage']} vs nominal {intervals['nominal']} "
          f"(gap {intervals['coverage_gap']})")
    print(f"    mean width {intervals['mean_width']}")

    outbreak = report["outbreak_detection"]
    if outbreak:
        print(f"\n  OUTBREAK DETECTION (threshold {outbreak['threshold_per_1000']} per 1,000)")
        print(f"    AUC {outbreak['auc']}   sensitivity {outbreak['sensitivity']}   "
              f"precision {outbreak['precision']}")
        print(f"    {outbreak['verdict']}")

    timeliness = report["timeliness"]
    if timeliness.get("mean_lead_time_weeks"):
        print(f"\n  TIMELINESS")
        print(f"    {timeliness['detected']}/{timeliness['onsets']} onsets detected, "
              f"mean lead time {timeliness['mean_lead_time_weeks']} weeks")

    baselines = report["baselines"]
    if baselines:
        print(f"\n  BASELINE GATE (critical rule #10)")
        for name, skill in baselines["skill"].items():
            marker = "PASS" if skill and skill > 0 else "FAIL"
            print(f"    vs {name:20} skill {skill:+.3f}  [{marker}]")
        print(f"    {baselines['verdict']}")

    print(f"\n  NOTES")
    for note in report["acceptance_notes"]:
        print(f"    - {note}")

    verdict = "PASS" if report["passes_acceptance"] else "FAIL"
    print(f"\n  VERDICT: {verdict}   ({elapsed:.1f}s)")
    report["duration_seconds"] = round(elapsed, 1)
    report["synthetic_sources"] = sorted(synthetic)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest against the acceptance criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("exit codes: 0 = all passed, 1 = a disease failed, "
                "2 = the backtest could not run\n"),
    )
    parser.add_argument("--disease", help="Disease slug (default: all registered)")
    parser.add_argument("--all", action="store_true", help="Backtest every registered disease")
    parser.add_argument("--districts", nargs="*", help="Restrict to these districts")
    parser.add_argument("--history-weeks", type=int, default=312,
                        help="Weeks of history to ingest (default 312 = 6 years)")
    parser.add_argument("--initial-train-weeks", type=int, default=156)
    parser.add_argument("--step-weeks", type=int, default=26)
    parser.add_argument("--test-weeks", type=int, default=26)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--end-week", help="Anchor week, e.g. 2024-W52")
    parser.add_argument("--json", metavar="PATH", help="Write the full report to a JSON file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    configure_logging("WARNING" if args.quiet else None)

    from src.models.registry import list_modules

    if args.disease:
        diseases = [args.disease]
    elif args.all:
        diseases = list_modules()
    else:
        diseases = ["malaria"]
        print("No --disease or --all given; backtesting malaria. "
              "Use --all to evaluate every disease.")

    reports = []
    for slug in diseases:
        try:
            reports.append(run_one(
                slug, args.history_weeks, args.districts, args.initial_train_weeks,
                args.step_weeks, args.test_weeks, args.max_folds, args.end_week,
            ))
        except Exception as exc:  # noqa: BLE001 - one disease failing must not stop the sweep
            log.exception("backtest failed for %s", slug)
            reports.append({"disease": slug, "error": str(exc), "passes_acceptance": False})

    print(f"\n{'=' * 74}")
    print("SUMMARY")
    print(f"{'=' * 74}")
    for report in reports:
        status = "PASS" if report.get("passes_acceptance") else "FAIL"
        detail = report.get("error") or (
            f"MAE {report.get('accuracy', {}).get('mae')}, "
            f"AUC {(report.get('outbreak_detection') or {}).get('auc')}"
        )
        print(f"  {status}  {report['disease']:16} {detail}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports, indent=2, default=str))
        print(f"\nwrote {path}")

    if any("error" in r and "no usable folds" in str(r.get("error", "")) for r in reports):
        return 2
    return 0 if all(r.get("passes_acceptance") for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
