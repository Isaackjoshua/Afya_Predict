"""Intervention logging, impact estimation and the feedback loops (#15)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.core.types import Alert, Intervention
from src.intervention_tracking.feedback_loop import (
    CONTAMINATION_COVERAGE, FeedbackLoop, ResponseAudit,
)
from src.intervention_tracking.impact_estimator import ImpactEstimator
from src.intervention_tracking.intervention_logger import (
    INTERVENTION_TYPES, InterventionLogger,
)

WEEKS = [f"2024-W{w:02d}" for w in range(1, 41)]


@pytest.fixture
def cache(tmp_path):
    from offline.local_cache import LocalCache

    return LocalCache(tmp_path / "test.sqlite")


@pytest.fixture
def logger(cache):
    return InterventionLogger(cache=cache)


def _alert(**overrides) -> Alert:
    base = dict(
        alert_id="alert-1", disease="Cholera", district="Sengerema", region="Mwanza",
        issued_at=datetime.utcnow() - timedelta(weeks=4), target_week="2024-W20",
        risk_level="high", risk_score=0.8, predicted_cases=400.0,
        predicted_incidence_per_1000=0.6, threshold_crossed=0.6, lead_time_weeks=6,
    )
    base.update(overrides)
    return Alert(**base)


# ------------------------------------------------------------------ logging
def test_logging_captures_coverage_and_timing(logger):
    intervention = logger.log(
        disease="cholera", district="Sengerema",
        intervention_type="water_chlorination", started_week="2024-W18",
        coverage=0.62, quantity=48000, unit="households",
    )
    assert intervention.intervention_id
    assert intervention.coverage == 0.62
    assert intervention.started_week == "2024-W18"
    assert logger.list(disease="cholera")


def test_coverage_is_clamped_to_a_share(logger):
    over = logger.log(disease="malaria", district="Ilala",
                      intervention_type="llin_distribution", coverage=3.5)
    under = logger.log(disease="malaria", district="Ilala",
                       intervention_type="llin_distribution", coverage=-1.0)
    assert over.coverage == 1.0
    assert under.coverage == 0.0


def test_an_unknown_type_is_accepted_but_uses_the_default_lag(logger):
    intervention = logger.log(disease="malaria", district="Ilala",
                              intervention_type="drone_delivery", coverage=0.4)
    assert intervention.intervention_type == "drone_delivery"
    assert InterventionLogger.effect_lag("drone_delivery") == \
        INTERVENTION_TYPES["other"]["effect_lag_weeks"]


def test_logging_from_an_alert_keeps_the_link(logger):
    alert = _alert()
    intervention = logger.log_from_alert(alert, "ors_distribution", coverage=0.5)
    assert intervention.alert_id == alert.alert_id
    assert intervention.disease == alert.disease
    assert logger.for_alert(alert.alert_id)


def test_response_time_is_measured_in_weeks(logger):
    alert = _alert()
    intervention = logger.log_from_alert(alert, "ors_distribution", coverage=0.5)
    intervention.started_week = "2024-W22"
    assert isinstance(logger.response_time(alert, intervention), int)


def test_every_known_type_declares_an_effect_lag():
    """The lag places the evaluation window; without it impact is unmeasurable."""
    for name, record in INTERVENTION_TYPES.items():
        assert record["effect_lag_weeks"] >= 0, name
        assert record["label"], name


def test_coverage_summary_aggregates_by_type(logger):
    logger.log(disease="cholera", district="A", intervention_type="ors_distribution",
               coverage=0.4)
    logger.log(disease="cholera", district="B", intervention_type="ors_distribution",
               coverage=0.8)
    summary = logger.coverage_summary("cholera")
    assert summary["n"] == 2
    assert summary["mean_coverage"] == pytest.approx(0.6)
    assert summary["by_type"]["ors_distribution"]["n"] == 2


# ------------------------------------------------------------------- impact
def _series(values) -> pd.Series:
    return pd.Series(values, index=WEEKS[:len(values)])


def test_impact_needs_enough_weeks_either_side():
    intervention = Intervention(
        intervention_id="i1", disease="cholera", district="Sengerema",
        intervention_type="water_chlorination", started_week="2024-W03",
    )
    estimate = ImpactEstimator().estimate(intervention, _series([10, 12, 11, 9, 10]))
    assert estimate.confidence == "insufficient_data"
    assert "at least" in " ".join(estimate.caveats)
    assert "Not enough observation weeks" in estimate.narrative()


def test_forecast_counterfactual_detects_a_reduction():
    """The estimate the platform is uniquely able to make."""
    observed = _series([100] * 12 + [60] * 16)      # a real drop after week 12
    forecast = _series([100] * 28)                  # the model expected no drop
    intervention = Intervention(
        intervention_id="i2", disease="cholera", district="Sengerema",
        intervention_type="water_chlorination", started_week=WEEKS[12], coverage=0.7,
    )
    estimate = ImpactEstimator().estimate(intervention, observed, forecast=forecast)
    assert estimate.forecast_counterfactual_effect is not None
    assert estimate.forecast_counterfactual_effect < 0        # fewer cases than expected
    assert estimate.relative_effect < 0
    assert "fewer than expected" in estimate.narrative()


def test_difference_in_differences_nets_out_the_season():
    """A drop that every district shared is not the intervention's doing."""
    observed = _series([100] * 12 + [60] * 16)
    controls = pd.DataFrame(
        {"Other1": [100] * 12 + [60] * 16, "Other2": [100] * 12 + [60] * 16},
        index=WEEKS[:28],
    )
    intervention = Intervention(
        intervention_id="i3", disease="cholera", district="Sengerema",
        intervention_type="water_chlorination", started_week=WEEKS[12], coverage=0.7,
    )
    estimate = ImpactEstimator().estimate(intervention, observed, control_series=controls)
    assert estimate.did_effect == pytest.approx(0.0, abs=1e-6)
    assert estimate.control_districts


def test_conflicting_estimates_are_reported_as_unresolved():
    observed = _series([100] * 12 + [60] * 16)
    forecast = _series([100] * 28)                       # says the drop is real
    controls = pd.DataFrame(                             # says everyone dropped further
        {"Other": [100] * 12 + [20] * 16}, index=WEEKS[:28]
    )
    intervention = Intervention(
        intervention_id="i4", disease="cholera", district="Sengerema",
        intervention_type="water_chlorination", started_week=WEEKS[12], coverage=0.7,
    )
    estimate = ImpactEstimator().estimate(
        intervention, observed, forecast=forecast, control_series=controls
    )
    assert estimate.agreement == "conflicting"
    assert estimate.confidence == "unresolved"
    assert "disagree" in estimate.narrative()


def test_confidence_never_reaches_high():
    """Observational data cannot support a causal claim, however many signals agree."""
    observed = _series([100] * 12 + [50] * 20)
    forecast = _series([100] * 32)
    controls = pd.DataFrame({"Other": [100] * 32}, index=WEEKS[:32])
    intervention = Intervention(
        intervention_id="i5", disease="cholera", district="Sengerema",
        intervention_type="water_chlorination", started_week=WEEKS[12], coverage=0.95,
    )
    estimate = ImpactEstimator().estimate(
        intervention, observed, forecast=forecast, control_series=controls
    )
    assert estimate.confidence in ("moderate", "low", "very_low", "unresolved")
    assert estimate.confidence != "high"


def test_pre_post_is_always_labelled_as_unreliable():
    observed = _series([100] * 12 + [60] * 16)
    intervention = Intervention(
        intervention_id="i6", disease="cholera", district="Sengerema",
        intervention_type="water_chlorination", started_week=WEEKS[12], coverage=0.7,
    )
    estimate = ImpactEstimator().estimate(intervention, observed)
    assert estimate.pre_post_effect is not None
    assert any("reference only" in c for c in estimate.caveats)
    assert "estimates" in estimate.to_dict()


def test_summarise_marks_results_as_associational():
    observed = _series([100] * 12 + [60] * 16)
    forecast = _series([100] * 28)
    intervention = Intervention(
        intervention_id="i7", disease="cholera", district="Sengerema",
        intervention_type="water_chlorination", started_week=WEEKS[12], coverage=0.7,
    )
    estimate = ImpactEstimator().estimate(intervention, observed, forecast=forecast)
    summary = ImpactEstimator().summarise([estimate])
    assert not summary.empty
    assert "associational" in summary.loc[0, "note"]


def test_summarise_handles_no_usable_estimates():
    assert ImpactEstimator().summarise([]).empty


# ------------------------------------------------------------- feedback loop
def test_response_audit_reports_no_responses_honestly(cache):
    loop = FeedbackLoop(cache=cache)
    audit = loop.audit_responses([_alert()])
    assert audit.total_alerts == 1
    assert audit.alerts_with_response == 0
    assert "none recorded a response" in audit.to_dict()["interpretation"]


def test_response_audit_flags_latency_exceeding_lead_time(cache, logger):
    """A system with 6 weeks of warning and a 9-week response has delivered nothing."""
    alert = _alert(lead_time_weeks=6)
    cache.save_alerts([alert])
    intervention = logger.log_from_alert(alert, "ors_distribution", coverage=0.5)
    # Respond long after the alert was issued.
    from src.core.timeutils import shift_week, to_epi_week

    intervention.started_week = shift_week(to_epi_week(alert.issued_at.date()), 9)
    cache.save_intervention(intervention)

    audit = FeedbackLoop(cache=cache).audit_responses([alert])
    assert audit.alerts_with_response == 1
    assert audit.lead_time_used is not None and audit.lead_time_used < 0
    assert "operational, not predictive" in audit.to_dict()["interpretation"]


def test_response_audit_handles_an_empty_window(cache):
    audit = FeedbackLoop(cache=cache).audit_responses([])
    assert audit.total_alerts == 0
    assert audit.response_rate == 0.0
    assert "No alerts issued" in audit.to_dict()["interpretation"]


def test_contaminated_weeks_are_down_weighted(cache, logger):
    """The model must not be punished for an outbreak its own alert helped avert."""
    logger.log(disease="malaria", district="Kinondoni",
               intervention_type="llin_distribution",
               started_week="2024-W10", coverage=0.9)
    index = pd.MultiIndex.from_product(
        [["Kinondoni", "Ilala"], WEEKS], names=["district", "week"]
    )
    weights = FeedbackLoop(cache=cache).contamination_weights(index)

    affected = weights.xs("Kinondoni", level="district")
    untouched = weights.xs("Ilala", level="district")
    assert (affected < 1.0).any()
    assert (untouched == 1.0).all()      # only the treated district is discounted


def test_low_coverage_interventions_do_not_contaminate(cache, logger):
    logger.log(disease="malaria", district="Kinondoni",
               intervention_type="llin_distribution",
               started_week="2024-W10", coverage=CONTAMINATION_COVERAGE / 2)
    index = pd.MultiIndex.from_product([["Kinondoni"], WEEKS], names=["district", "week"])
    weights = FeedbackLoop(cache=cache).contamination_weights(index)
    assert (weights == 1.0).all()


def test_weight_scales_with_coverage(cache, logger):
    logger.log(disease="malaria", district="A", intervention_type="llin_distribution",
               started_week="2024-W10", coverage=0.3)
    logger.log(disease="malaria", district="B", intervention_type="llin_distribution",
               started_week="2024-W10", coverage=0.95)
    index = pd.MultiIndex.from_product([["A", "B"], WEEKS], names=["district", "week"])
    weights = FeedbackLoop(cache=cache).contamination_weights(index)
    assert weights.xs("A", level="district").min() > weights.xs("B", level="district").min()


def test_contaminated_weeks_are_listed_for_audit(cache, logger):
    logger.log(disease="malaria", district="Kinondoni",
               intervention_type="irs_spraying", started_week="2024-W10", coverage=0.8)
    frame = FeedbackLoop(cache=cache).contaminated_weeks()
    assert not frame.empty
    assert frame.loc[0, "district"] == "Kinondoni"
    assert 0 < frame.loc[0, "weight"] < 1


def test_preferred_actions_are_labelled_non_causal(cache):
    observed = _series([100] * 12 + [60] * 16)
    forecast = _series([100] * 28)
    intervention = Intervention(
        intervention_id="i8", disease="cholera", district="Sengerema",
        intervention_type="water_chlorination", started_week=WEEKS[12], coverage=0.7,
    )
    estimate = ImpactEstimator().estimate(intervention, observed, forecast=forecast)
    actions = FeedbackLoop(cache=cache).preferred_actions("cholera", estimates=[estimate])
    assert actions
    assert "not a causal estimate" in actions[0]["note"]


def test_preferred_actions_without_estimates_is_empty(cache):
    assert FeedbackLoop(cache=cache).preferred_actions("cholera") == []
