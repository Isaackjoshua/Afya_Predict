"""Offline node: local cache, distilled edge model and sync (#14, rule #6)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.core.types import Alert, Intervention, PredictionResult


@pytest.fixture
def cache(tmp_path):
    from offline.local_cache import LocalCache

    return LocalCache(tmp_path / "node.sqlite")


def _prediction(district="Kinondoni", week="2026-W10", **overrides) -> PredictionResult:
    base = dict(
        prediction_id=f"p-{district}-{week}", disease="Malaria", district=district,
        region="Dar es Salaam", forecast_date=date(2026, 1, 5), target_week=week,
        predicted_cases=1200.0, confidence_interval_lower=900.0,
        confidence_interval_upper=1600.0, risk_level="high", risk_score=0.8,
        model_version="test-1",
    )
    base.update(overrides)
    return PredictionResult(**base)


def _alert(alert_id="a1", **overrides) -> Alert:
    base = dict(
        alert_id=alert_id, disease="Malaria", district="Kinondoni",
        region="Dar es Salaam", issued_at=datetime.utcnow(), target_week="2026-W10",
        risk_level="high", risk_score=0.8, predicted_cases=1200.0,
        predicted_incidence_per_1000=1.0, threshold_crossed=5.0,
    )
    base.update(overrides)
    return Alert(**base)


# ------------------------------------------------------------------- cache
def test_fresh_cache_is_empty_but_valid(cache):
    status = cache.status()
    assert status["predictions"] == 0
    assert status["offline_ready"] is False
    assert status["size_bytes"] > 0


def test_predictions_round_trip(cache):
    assert cache.save_predictions([_prediction()]) == 1
    stored = cache.get_prediction("p-Kinondoni-2026-W10")
    assert stored is not None
    assert stored.predicted_cases == 1200.0
    assert stored.risk_level == "high"


def test_prediction_filters(cache):
    cache.save_predictions([
        _prediction(district="Kinondoni", week="2026-W10"),
        _prediction(district="Ilala", week="2026-W11", risk_level="low"),
    ])
    assert len(cache.latest_predictions(district="Ilala")) == 1
    assert len(cache.latest_predictions(risk_level="high")) == 1
    assert len(cache.latest_predictions(target_week="2026-W11")) == 1
    assert len(cache.latest_predictions()) == 2


def test_saving_the_same_prediction_twice_replaces_it(cache):
    cache.save_predictions([_prediction()])
    cache.save_predictions([_prediction(predicted_cases=99.0)])
    assert cache.prediction_count() == 1
    assert cache.get_prediction("p-Kinondoni-2026-W10").predicted_cases == 99.0


def test_offline_readiness_needs_two_weeks(cache):
    """Acceptance criterion #12."""
    cache.save_predictions([_prediction(week="2026-W10")])
    assert cache.status()["offline_ready"] is False
    cache.save_predictions([_prediction(week="2026-W11")])
    status = cache.status()
    assert status["weeks_cached"] == 2
    assert status["offline_ready"] is True


def test_alerts_round_trip_and_acknowledge(cache):
    cache.save_alerts([_alert()])
    assert len(cache.get_alerts()) == 1
    assert cache.acknowledge_alert("a1", by="dho@example.org") is True
    acknowledged = cache.get_alerts(acknowledged=True)
    assert acknowledged and acknowledged[0].acknowledged_by == "dho@example.org"
    assert cache.acknowledge_alert("missing", by="x") is False


def test_alert_time_filter(cache):
    old = _alert("old", issued_at=datetime.utcnow() - timedelta(days=60))
    new = _alert("new")
    cache.save_alerts([old, new])
    recent = cache.get_alerts(since=datetime.utcnow() - timedelta(days=7))
    assert [a.alert_id for a in recent] == ["new"]


def test_sync_marking(cache):
    cache.save_alerts([_alert()])
    assert len(cache.unsynced_alerts()) == 1
    assert cache.mark_synced("alerts", ["a1"]) == 1
    assert cache.unsynced_alerts() == []
    assert cache.mark_synced("nonsense_table", ["a1"]) == 0


def test_acknowledging_marks_the_alert_for_resync(cache):
    """A local acknowledgement is a record the centre cannot reconstruct."""
    cache.save_alerts([_alert()])
    cache.mark_synced("alerts", ["a1"])
    cache.acknowledge_alert("a1", by="dho")
    assert len(cache.unsynced_alerts()) == 1


def test_interventions_and_observations_round_trip(cache):
    intervention = Intervention(
        intervention_id="i1", disease="malaria", district="Kinondoni",
        intervention_type="llin_distribution", started_week="2026-W08", coverage=0.6,
    )
    cache.save_intervention(intervention)
    assert len(cache.get_interventions(disease="malaria")) == 1
    assert len(cache.unsynced_interventions()) == 1

    assert cache.save_observations([
        {"disease": "malaria", "district": "Kinondoni", "week": "2026-W08",
         "cases": 900.0, "incidence_per_1000": 0.74, "quality": 0.9},
    ]) == 1
    assert cache.get_observations("malaria", district="Kinondoni")[0]["cases"] == 900.0


def test_metadata_round_trip(cache):
    cache.set_meta("last_sync", {"ok": True, "pushed": 3})
    assert cache.get_meta("last_sync")["pushed"] == 3
    assert cache.get_meta("never_set", default="fallback") == "fallback"


def test_prune_removes_stale_rows(cache):
    cache.save_predictions([_prediction()])
    result = cache.prune(keep_days=0)
    assert result["predictions_removed"] >= 1


# --------------------------------------------------------- lightweight model
@pytest.fixture(scope="module")
def teacher():
    from src.models.base_model import TrainedModel
    from src.models.backends import BackendInfo, build_regressor

    rng = np.random.default_rng(3)
    columns = [f"f{i}" for i in range(12)]
    X = pd.DataFrame(rng.normal(size=(400, 12)), columns=columns)
    y = 5 * X["f0"] + 3 * X["f1"] - 2 * X["f2"] + rng.normal(0, 0.4, 400)

    estimator, info = build_regressor("numpy_gbm", random_state=1)
    estimator.fit(X.to_numpy(), y.to_numpy())
    model = TrainedModel(estimator=estimator, backend=info, feature_names=columns,
                         scope="pooled", n_rows=len(X))
    return model, X


def test_distillation_reproduces_the_teacher(teacher):
    from offline.lightweight_model import LightweightModel

    model, X = teacher
    light = LightweightModel.distil(model, X, n_features=6)
    assert len(light.features) == 6
    assert light.fidelity_r2 > 0.5
    assert light.describe()["usable_offline"] in (True, False)


def test_distilled_model_is_small_enough_to_sync(teacher):
    """It has to fit through a bad link, which is the entire point."""
    from offline.lightweight_model import LightweightModel

    model, X = teacher
    light = LightweightModel.distil(model, X, n_features=8)
    assert light.size_bytes < 12_000


def test_distilled_model_predicts_and_explains(teacher):
    from offline.lightweight_model import LightweightModel

    model, X = teacher
    light = LightweightModel.distil(model, X, n_features=6)

    predictions = light.predict(X.head(20))
    assert len(predictions) == 20
    assert (predictions >= 0).all()

    point, lower, upper = light.predict_with_interval(X.head(5))
    assert (lower <= point).all() and (point <= upper).all()

    drivers = light.explain(X.iloc[0])
    assert len(drivers) == 6
    assert abs(sum(d["share"] for d in drivers) - 1.0) < 1e-6
    assert drivers[0]["direction"] in ("increases", "decreases")


def test_distilled_model_handles_missing_inputs(teacher):
    from offline.lightweight_model import LightweightModel

    model, X = teacher
    light = LightweightModel.distil(model, X, n_features=5)
    incomplete = X.head(3).copy()
    incomplete.iloc[:, 0] = np.nan
    assert np.isfinite(light.predict(incomplete)).all()


def test_distilled_model_round_trips_through_json(teacher, tmp_path):
    from offline.lightweight_model import LightweightModel

    model, X = teacher
    light = LightweightModel.distil(model, X, n_features=5)
    path = light.save(tmp_path / "edge.json")
    reloaded = LightweightModel.load(path)
    assert reloaded.features == light.features
    np.testing.assert_allclose(reloaded.predict(X.head(5)), light.predict(X.head(5)))


def test_distilled_model_states_its_own_fidelity(teacher):
    """It reports the accuracy cost rather than degrading silently."""
    from offline.lightweight_model import LightweightModel

    model, X = teacher
    description = LightweightModel.distil(model, X, n_features=4).describe()
    assert "reproduces" in description["note"]
    assert 0 <= description["fidelity_r2"] <= 1


# ------------------------------------------------------------ sync manager
def test_sync_is_offline_without_a_central_url(cache):
    from offline.sync_manager import SyncManager

    manager = SyncManager(cache=cache, central_url="")
    assert manager.is_online() is False
    report = manager.sync()
    assert report.online is False
    assert report.ok is False
    assert cache.get_meta("last_sync_attempt") is not None


def test_sync_flushes_the_notification_queue_even_when_offline(cache, tmp_path):
    """Rule #6: SMS and email gateways are independent of the central instance."""
    from config.settings import get_settings
    from offline.sync_manager import SyncManager

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    report = SyncManager(cache=cache, central_url="", settings=settings).sync()
    assert report.notifications_flushed == 0     # nothing queued, but it tried
    assert "notification" not in " ".join(report.errors)


def test_offline_readiness_message_is_actionable(cache):
    from offline.sync_manager import SyncManager

    manager = SyncManager(cache=cache)
    not_ready = manager.offline_readiness()
    assert not_ready["ready"] is False
    assert "run the prediction job" in not_ready["message"]

    cache.save_predictions([_prediction(week="2026-W10"), _prediction(week="2026-W11")])
    ready = manager.offline_readiness()
    assert ready["ready"] is True
    assert "can operate offline" in ready["message"]


def test_sync_report_serialises(cache):
    from offline.sync_manager import SyncReport

    report = SyncReport(online=True, alerts_pushed=2)
    payload = report.to_dict()
    assert payload["alerts_pushed"] == 2
    assert payload["ok"] is True
