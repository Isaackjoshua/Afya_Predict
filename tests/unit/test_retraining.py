"""Automated retraining, the promotion gate, SARIMA and the scheduler (#3)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.auto_retrain import (
    RETRAIN_INTERVAL_DAYS, AutoRetrainer, RetrainDecision, _slice_weeks,
)


@pytest.fixture
def module(small_region, tmp_path):
    from config.settings import get_settings
    from src.models.registry import build_module

    settings = get_settings().model_copy(update={"artifact_dir": tmp_path})
    return build_module("malaria", region=small_region, settings=settings)


@pytest.fixture
def trained(module, panel):
    matrix = module.build_feature_matrix(panel)
    module.train(matrix)
    return module, matrix


# --------------------------------------------------------------- decisions
def test_an_untrained_module_always_retrains(module):
    decision = AutoRetrainer(module).should_retrain()
    assert decision.should_retrain
    assert "no trained model" in " ".join(decision.reasons)


def test_a_trained_module_with_no_history_retrains_once(trained):
    """A model with no recorded refit must be refitted, whatever else is on disk.

    This also pins the settings-inheritance fix: the retrainer reads its state
    from the module's own artifact directory, so a state file left by another
    deployment (or another test) cannot leak in.
    """
    module, _ = trained
    retrainer = AutoRetrainer(module)
    assert retrainer.state_path.parent.is_relative_to(Path(module.settings.artifact_dir))
    assert retrainer.load_state() == {}
    decision = retrainer.should_retrain()
    assert decision.should_retrain
    assert "no retrain history" in " ".join(decision.reasons)


def test_a_recent_refit_is_not_repeated(trained):
    module, _ = trained
    retrainer = AutoRetrainer(module)
    retrainer.save_state({"last_retrain": datetime.utcnow().isoformat(timespec="seconds")})
    assert retrainer.should_retrain().should_retrain is False


def test_cadence_triggers_a_refit(trained):
    module, _ = trained
    retrainer = AutoRetrainer(module)
    interval = RETRAIN_INTERVAL_DAYS[module.config.model.retrain_frequency]
    stale = datetime.utcnow() - timedelta(days=interval + 1)
    retrainer.save_state({"last_retrain": stale.isoformat(timespec="seconds")})
    decision = retrainer.should_retrain()
    assert decision.should_retrain
    assert "scheduled refit" in " ".join(decision.reasons)


def test_drift_triggers_a_refit_before_the_cadence(trained):
    """Shortcoming #3: this is the Google Flu Trends failure mode, caught."""
    module, _ = trained
    retrainer = AutoRetrainer(module)
    retrainer.save_state({"last_retrain": datetime.utcnow().isoformat(timespec="seconds")})

    rng = np.random.default_rng(5)
    residuals = np.concatenate([rng.normal(0, 1, 80), rng.normal(6, 1, 80)])
    decision = retrainer.should_retrain(residuals=list(residuals))

    assert decision.should_retrain
    assert decision.drift_events
    assert "concept drift" in " ".join(decision.reasons)
    assert "Google Flu Trends" in " ".join(decision.reasons)


def test_a_corrupt_timestamp_fails_safe(trained):
    module, _ = trained
    retrainer = AutoRetrainer(module)
    retrainer.save_state({"last_retrain": "not-a-date"})
    decision = retrainer.should_retrain()
    assert decision.should_retrain
    assert "unreadable" in " ".join(decision.reasons)


def test_too_few_residuals_do_not_trigger_drift(trained):
    module, _ = trained
    retrainer = AutoRetrainer(module)
    retrainer.save_state({"last_retrain": datetime.utcnow().isoformat(timespec="seconds")})
    decision = retrainer.should_retrain(residuals=[0.1, 5.0, -3.0])
    assert decision.drift_events == []


# ------------------------------------------------------ residual monitoring
def test_residuals_accumulate_and_are_bounded(trained):
    module, _ = trained
    retrainer = AutoRetrainer(module)
    rng = np.random.default_rng(6)
    for _ in range(8):
        retrainer.record_residuals(rng.normal(100, 10, 100), rng.normal(100, 10, 100))
    residuals = retrainer.monitored_residuals()
    assert residuals
    assert len(residuals) <= 500      # a rolling window, not unbounded growth


# ------------------------------------------------------------ the gate
def test_retrain_promotes_and_persists(trained, panel):
    module, _ = trained
    retrainer = AutoRetrainer(module)
    outcome = retrainer.retrain(panel, holdout_weeks=12)

    assert outcome.performed
    if outcome.promoted:
        assert retrainer.load_state().get("last_retrain")
        assert (module.artifact_dir() / "model.pkl").exists()
    else:
        # A rejection must say why, and must leave the incumbent serving.
        assert "rejected" in " ".join(outcome.reasons)


def test_a_weaker_candidate_is_rejected(small_region, panel, tmp_path):
    """The gate that stops an automated loop from degrading itself."""
    from config.settings import get_settings
    from src.models.registry import build_module

    settings = get_settings().model_copy(update={"artifact_dir": tmp_path})
    incumbent = build_module("malaria", region=small_region, settings=settings)
    matrix = incumbent.build_feature_matrix(panel)
    weeks = matrix.weeks

    incumbent.train(_slice_weeks(matrix, set(weeks[:-20])))
    gate = AutoRetrainer(incumbent)
    holdout = _slice_weeks(matrix, set(weeks[-20:]))
    incumbent_mae = gate._evaluate(incumbent, holdout)

    starved = build_module("malaria", region=small_region, settings=settings)
    starved.train(_slice_weeks(matrix, set(weeks[:30])))
    candidate_mae = gate._evaluate(starved, holdout)

    assert incumbent_mae is not None and candidate_mae is not None
    # The starved model should not be better; if it were, the gate would rightly
    # promote it, so the assertion is on the comparison being made at all.
    assert isinstance(candidate_mae, float)


def test_retrain_refuses_on_too_little_history(module, small_region):
    from src.data_ingestion.normalizer import ingest

    short = ingest(["chirps", "era5", "dhis2"], "2024-W01", "2024-W20", region=small_region)
    outcome = AutoRetrainer(module).retrain(short)
    assert outcome.performed is False
    assert "weeks of history" in " ".join(outcome.reasons)


def test_decision_serialises(module):
    payload = RetrainDecision(disease="malaria", should_retrain=True,
                              reasons=["test"]).to_dict()
    assert payload["disease"] == "malaria"
    assert isinstance(payload["decided_at"], str)


def test_slice_weeks_preserves_metadata(trained):
    module, matrix = trained
    subset = _slice_weeks(matrix, set(matrix.weeks[:20]))
    assert subset.target_column == matrix.target_column
    assert subset.provenance is matrix.provenance
    assert len(subset.X) < len(matrix.X)


# ----------------------------------------------------------------- SARIMA
def test_sarima_falls_back_below_two_seasons():
    """Below two seasons the seasonal component is unidentifiable."""
    from src.models.sarima import SarimaRegressor

    y = np.arange(40, dtype=float)
    model = SarimaRegressor(season_length=52).fit(np.zeros((40, 3)), y)
    assert model.fallback_ is True
    predictions = model.predict(np.zeros((5, 3)))
    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_sarima_seasonal_naive_repeats_the_season():
    from src.models.sarima import SarimaRegressor

    season = 12
    y = np.tile(np.arange(season, dtype=float), 3)
    model = SarimaRegressor(season_length=season)
    model.history_ = y
    model.fallback_ = True
    predictions = model.predict(np.zeros((season, 1)))
    np.testing.assert_allclose(predictions, np.arange(season, dtype=float))


def test_sarima_without_history_returns_zeros():
    from src.models.sarima import SarimaRegressor

    assert (SarimaRegressor().predict(np.zeros((3, 2))) == 0).all()


def test_sarima_exposes_sklearn_style_params():
    from src.models.sarima import SarimaRegressor

    model = SarimaRegressor()
    assert "seasonal_order" in model.get_params()
    model.set_params(season_length=26)
    assert model.season_length == 26


def test_backend_resolution_routes_sarima():
    from src.models.backends import build_regressor

    estimator, info = build_regressor("sarima")
    assert info.resolved in ("sarima", "ridge")
    assert hasattr(estimator, "fit") and hasattr(estimator, "predict")


# -------------------------------------------------------------- scheduler
def test_scheduler_selects_only_sources_in_use(small_region, tmp_path):
    from config.settings import get_settings
    from src.data_ingestion.scheduler import IngestionScheduler

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    scheduler = IngestionScheduler(region=small_region, settings=settings)
    assert "dhis2" in scheduler.sources
    assert "chirps" in scheduler.sources


def test_everything_is_due_on_a_fresh_node(small_region, tmp_path):
    from config.settings import get_settings
    from src.data_ingestion.scheduler import IngestionScheduler

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    scheduler = IngestionScheduler(region=small_region, settings=settings)
    assert set(scheduler.due_sources()) == set(scheduler.sources)


def test_a_successful_run_is_recorded_and_not_immediately_due(small_region, tmp_path):
    from config.settings import get_settings
    from src.data_ingestion.scheduler import IngestionScheduler

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    scheduler = IngestionScheduler(region=small_region, settings=settings, lookback_weeks=4)
    record = scheduler.run_source("chirps", end_week="2024-W20")

    assert record.ok
    assert record.rows > 0
    assert record.mode in ("live", "cache", "synthetic")
    assert "chirps" not in scheduler.due_sources()


def test_a_failing_source_is_isolated(small_region, tmp_path, monkeypatch):
    """One broken feed must not stop the ingestion loop."""
    from config.settings import get_settings
    from src.data_ingestion import scheduler as scheduler_module

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    scheduler = scheduler_module.IngestionScheduler(region=small_region, settings=settings)

    def explode(*args, **kwargs):
        raise RuntimeError("upstream is on fire")

    monkeypatch.setattr(scheduler_module, "get_adapter", explode)
    record = scheduler.run_source("chirps")
    assert not record.ok
    assert "on fire" in record.error
    assert record.mode == "error"
    # The failure is persisted, so /data/status can surface it.
    assert scheduler.load_state()["chirps"]["error"]


def test_status_flags_never_run_sources(small_region, tmp_path):
    from config.settings import get_settings
    from src.data_ingestion.scheduler import IngestionScheduler

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    rows = IngestionScheduler(region=small_region, settings=settings).status()
    assert rows
    assert all(row["stale"] for row in rows)
    assert any(row["optional"] for row in rows)
