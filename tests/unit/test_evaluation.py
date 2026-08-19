"""Evaluation: metrics, baselines, detection, calibration and walk-forward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmark import (
    autoregressive_baseline,
    benchmark_against_naive,
    rolling_mean_baseline,
    seasonal_naive,
)
from src.evaluation.calibration import (
    brier_score,
    interval_calibration,
    recalibration_factor,
    risk_score_reliability,
)
from src.evaluation.metrics import (
    average_precision,
    classification_metrics,
    interval_metrics,
    regression_metrics,
    roc_auc,
    skill_score,
)
from src.evaluation.outbreak_detection import (
    evaluate_detection,
    evaluate_timeliness,
    label_outbreaks,
    threshold_sweep,
)
from src.evaluation.spatial_accuracy import (
    detect_new_onsets,
    evaluate_spatial,
    hotspot_hit_rate,
    importation_accuracy,
)
from src.evaluation.walk_forward_cv import WalkForwardCV

WEEKS = [f"2023-W{w:02d}" for w in range(1, 53)] + [f"2024-W{w:02d}" for w in range(1, 53)]


# ------------------------------------------------------------------ metrics
def test_regression_metrics_on_a_perfect_forecast():
    values = [1.0, 2.0, 3.0, 4.0]
    metrics = regression_metrics(values, values)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_regression_metrics_ignore_missing_pairs():
    metrics = regression_metrics([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    assert metrics["n"] == 2
    assert metrics["mae"] == 0.0


def test_roc_auc_matches_known_values():
    assert roc_auc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([False, False, True, True], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([False, True], [0.5, 0.5]) == 0.5   # ties -> chance


def test_roc_auc_is_undefined_without_both_classes():
    assert np.isnan(roc_auc([True, True], [0.1, 0.9]))


def test_average_precision_rewards_ranking_positives_first():
    assert average_precision([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert average_precision([False, False, True, True], [0.9, 0.8, 0.2, 0.1]) < 0.6


def test_classification_metrics_confusion_matrix():
    metrics = classification_metrics([True, True, False, False], [True, False, True, False])
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["precision"] == 0.5 and metrics["recall"] == 0.5


def test_interval_metrics_measure_coverage():
    actual = np.array([1.0, 2.0, 3.0, 10.0])
    metrics = interval_metrics(actual, actual - 1, actual + 1)
    assert metrics["coverage"] == 1.0
    tight = interval_metrics(actual, actual + 5, actual + 6)
    assert tight["coverage"] == 0.0


def test_skill_score_sign_convention():
    assert skill_score(5.0, 10.0) == 0.5      # half the baseline error
    assert skill_score(20.0, 10.0) == -1.0    # twice the baseline error


# ---------------------------------------------------------------- baselines
@pytest.fixture
def seasonal_series():
    rng = np.random.default_rng(0)
    signal = 60 + 30 * np.sin(np.arange(len(WEEKS)) / 52 * 2 * np.pi)
    return pd.Series(signal + rng.normal(0, 4, len(WEEKS)), index=WEEKS)


def test_all_three_baselines_produce_forecasts(seasonal_series):
    assert seasonal_naive(seasonal_series, 4).notna().any()
    assert rolling_mean_baseline(seasonal_series).notna().any()
    assert autoregressive_baseline(seasonal_series).notna().any()


def test_a_good_model_passes_the_baseline_gate(seasonal_series):
    """Rule #10: beat all three or do not deploy."""
    rng = np.random.default_rng(1)
    good = seasonal_series + rng.normal(0, 1, len(seasonal_series))
    comparison = benchmark_against_naive(seasonal_series, good, history=seasonal_series, horizon=4)
    assert comparison.passes
    assert "PASS" in comparison.verdict()
    assert not comparison.to_frame().empty


def test_a_useless_model_fails_the_baseline_gate(seasonal_series):
    rng = np.random.default_rng(2)
    useless = pd.Series(rng.normal(60, 40, len(seasonal_series)), index=seasonal_series.index)
    comparison = benchmark_against_naive(seasonal_series, useless, history=seasonal_series, horizon=4)
    assert not comparison.passes
    assert "FAIL" in comparison.verdict()
    assert comparison.failed_against


# ------------------------------------------------------------ outbreak detection
def test_outbreak_labelling_requires_persistence():
    incidence = pd.Series([0.1, 5.0, 0.1, 5.0, 5.0, 5.0], index=WEEKS[:6])
    single = label_outbreaks(incidence, 1.0, min_consecutive_weeks=1)
    sustained = label_outbreaks(incidence, 1.0, min_consecutive_weeks=3)
    assert single.sum() == 4
    assert sustained.sum() < single.sum()   # the isolated spikes are filtered


def test_detection_scores_a_skilful_forecast():
    rng = np.random.default_rng(3)
    actual = pd.Series(np.abs(rng.normal(1.0, 1.0, 200)), index=range(200))
    predicted = actual + rng.normal(0, 0.15, 200)
    evaluation = evaluate_detection(actual, predicted, threshold_per_1000=2.0)
    assert evaluation.evaluable
    assert evaluation.auc > 0.75
    assert evaluation.passes_acceptance
    assert "PASS" in evaluation.verdict


def test_detection_reports_not_evaluable_when_nothing_happened():
    """A quiet window is not the same finding as a missed outbreak."""
    quiet = pd.Series(np.full(50, 0.01), index=range(50))
    evaluation = evaluate_detection(quiet, quiet, threshold_per_1000=5.0)
    assert not evaluation.evaluable
    assert not evaluation.passes_acceptance
    assert "NOT EVALUABLE" in evaluation.verdict


def test_detection_suggests_an_operating_point():
    rng = np.random.default_rng(4)
    actual = pd.Series(np.abs(rng.normal(1.5, 1.2, 300)), index=range(300))
    # A conservative forecast: correct ranking, compressed magnitude.
    predicted = actual * 0.6 + rng.normal(0, 0.1, 300)
    evaluation = evaluate_detection(actual, predicted, threshold_per_1000=3.0)
    assert evaluation.optimal_threshold is not None
    assert evaluation.optimal_threshold < 3.0   # a lower trigger recovers sensitivity


def test_timeliness_measures_real_lead_time():
    """The number that justifies forecasting over surveillance."""
    incidence = pd.Series([0.1] * 10 + [5.0] * 5, index=WEEKS[:15])
    early = pd.Series([0.1] * 7 + [5.0] * 8, index=WEEKS[:15])
    result = evaluate_timeliness(incidence, early, threshold_per_1000=1.0, forecast_horizon_weeks=4)
    assert result["onsets"] == 1
    assert result["detected"] == 1
    assert result["mean_lead_time_weeks"] >= 4


def test_threshold_sweep_covers_every_candidate():
    rng = np.random.default_rng(5)
    actual = pd.Series(np.abs(rng.normal(1, 1, 120)), index=range(120))
    sweep = threshold_sweep(actual, actual * 0.9, [0.5, 1.0, 2.0])
    assert len(sweep) == 3
    assert "auc" in sweep.columns


# --------------------------------------------------------------- calibration
def test_calibration_detects_overconfident_intervals():
    rng = np.random.default_rng(6)
    actual = rng.normal(100, 20, 400)
    report = interval_calibration(actual, actual * 0 + 99, actual * 0 + 101)
    assert report.coverage < 0.2
    assert not report.is_calibrated


def test_calibration_accepts_honest_intervals():
    rng = np.random.default_rng(7)
    predicted = rng.normal(100, 20, 500)
    actual = predicted + rng.normal(0, 5, 500)
    report = interval_calibration(actual, predicted - 9.8, predicted + 9.8)
    assert report.is_calibrated
    assert not report.reliability.empty


def test_recalibration_factor_widens_undercovering_intervals():
    rng = np.random.default_rng(8)
    predicted = rng.normal(100, 20, 300)
    actual = predicted + rng.normal(0, 10, 300)
    factor = recalibration_factor(actual, predicted - 2, predicted + 2)
    assert factor > 1.0


def test_risk_score_reliability_and_brier():
    rng = np.random.default_rng(9)
    scores = rng.uniform(0, 1, 500)
    outcomes = rng.uniform(0, 1, 500) < scores      # perfectly calibrated by construction
    reliability = risk_score_reliability(scores, outcomes)
    assert not reliability.empty
    assert abs(reliability["gap"].mean()) < 0.15
    assert 0 <= brier_score(scores, outcomes) <= 1


# ------------------------------------------------------------------- spatial
def test_hotspot_hit_rate_rewards_naming_the_right_districts():
    predicted = pd.Series({"A": 9, "B": 8, "C": 1, "D": 0})
    actual = pd.Series({"A": 10, "B": 7, "C": 0, "D": 0})
    assert hotspot_hit_rate(predicted, actual, k=2) == 1.0
    assert hotspot_hit_rate(pd.Series({"A": 0, "B": 0, "C": 9, "D": 8}), actual, k=2) == 0.0


def test_spatial_scorecard_includes_distance_error(small_region):
    predicted = pd.Series({d: float(i) for i, d in enumerate(small_region.district_names)})
    actual = predicted.iloc[::-1]
    scorecard = evaluate_spatial(predicted, actual, small_region, week="2024-W10", k=2)
    assert scorecard.week == "2024-W10"
    assert np.isfinite(scorecard.mean_distance_error_km)
    assert scorecard.predicted_hotspots


def test_importation_accuracy_scores_named_onsets():
    predicted = pd.Series({"A": 0.9, "B": 0.5, "C": 0.1})
    result = importation_accuracy(predicted, ["A"], k=2)
    assert result["hits"] == 1
    assert result["recall_at_k"] == 1.0


def test_new_onsets_are_districts_that_were_previously_quiet():
    frame = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0, 5.0], "B": [5.0, 5.0, 5.0, 5.0]}, index=WEEKS[:4]
    )
    onsets = detect_new_onsets(frame, threshold=1.0, week=WEEKS[3])
    assert onsets == ["A"]      # B was never quiet


# -------------------------------------------------------------- walk-forward
def test_splits_are_chronological_and_purged(small_region, malaria_config):
    from src.models.registry import build_module

    module = build_module("malaria", region=small_region)
    cv = WalkForwardCV(module, initial_train_weeks=20, step_weeks=10, test_weeks=10)
    splits = list(cv.splits(WEEKS, horizon=8))
    assert splits
    for train, test in splits:
        assert max(train) < min(test)          # never trains on the future
        gap_weeks = WEEKS.index(test[0]) - WEEKS.index(train[-1])
        assert gap_weeks > 8                   # purge gap covers the horizon


def test_walk_forward_produces_out_of_sample_scores(panel, small_region):
    from src.models.registry import build_module

    module = build_module("malaria", region=small_region)
    matrix = module.build_feature_matrix(panel)
    cv = WalkForwardCV(module, initial_train_weeks=60, step_weeks=20, test_weeks=20, max_folds=2)
    result = cv.run(matrix)
    assert result.folds
    assert not result.predictions.empty
    assert np.isfinite(result.metrics["mae"])
    assert result.baselines is not None
    assert result.acceptance_notes
    report = result.report()
    assert report["disease"] == "malaria"
    assert "passes_acceptance" in report
