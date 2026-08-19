"""Alerting: classification, escalation, recommendations and delivery."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.alerting.alert_generator import AlertGenerator, build_alert
from src.alerting.escalation_rules import EscalationEngine
from src.alerting.notification_service import NotificationService
from src.alerting.recommendation_engine import RecommendationEngine
from src.alerting.risk_classifier import RiskClassifier
from src.core.types import Alert, DriverExplanation, PredictionResult, SourceRisk


@pytest.fixture
def classifier(cholera_config):
    return RiskClassifier(cholera_config.alerts)


@pytest.fixture
def sample_prediction():
    return PredictionResult(
        prediction_id="p1",
        disease="Cholera",
        district="Sengerema",
        region="Mwanza",
        forecast_date=datetime.utcnow().date(),
        target_week="2026-W12",
        predicted_cases=480.0,
        confidence_interval_lower=380.0,
        confidence_interval_upper=610.0,
        risk_level="high",
        risk_score=0.8,
        top_drivers=[
            DriverExplanation(feature="rainfall_mm_lag3", proxy="rainfall", lag_weeks=3,
                              shap_value=2.0, contribution_share=0.4, direction="increases",
                              mechanism="heavy rain floods latrines"),
        ],
        natural_language_explanation="Risk is high for cholera in Sengerema.",
        importation_risk=0.35,
        source_districts=[SourceRisk(district="Mwanza City", flow_weight=0.3,
                                     active_cases=120.0, contributed_risk=0.6)],
    )


# ------------------------------------------------------------- classification
def test_thresholds_map_to_levels(classifier, cholera_config):
    assert classifier.base_level(0.0) == "low"
    assert classifier.base_level(cholera_config.alerts.medium) == "medium"
    assert classifier.base_level(cholera_config.alerts.high) == "high"
    assert classifier.base_level(cholera_config.alerts.critical * 2) == "critical"


def test_risk_score_is_monotonic(classifier):
    scores = [classifier.score(x) for x in (0.0, 0.05, 0.2, 0.6, 1.5, 5.0)]
    assert scores == sorted(scores)
    assert scores[-1] == 1.0


def test_rapid_growth_escalates_a_district(classifier, cholera_config):
    steady = classifier.classify(cholera_config.alerts.medium, recent_incidence=[0.2, 0.2, 0.2, 0.2])
    surging = classifier.classify(cholera_config.alerts.medium, recent_incidence=[0.02, 0.03, 0.04, 0.05])
    assert surging.level != "low"
    assert any("rising faster" in a for a in surging.adjustments)
    assert steady.level == "medium"


def test_importation_pressure_escalates_a_quiet_district(classifier):
    """Shortcoming #10: spread must be actionable before local counts rise."""
    result = classifier.classify(0.25, importation_risk=0.85)
    assert any("importation risk" in a for a in result.adjustments)
    assert result.level in ("high", "critical")


def test_low_data_confidence_pulls_the_level_back(classifier, cholera_config):
    """Rule #7: state the uncertainty rather than over-claiming."""
    result = classifier.classify(cholera_config.alerts.high, ci_width_ratio=3.0, input_quality=0.3)
    assert result.low_confidence
    assert result.level == "medium"
    assert any("low data confidence" in a for a in result.adjustments)


def test_notification_floors_follow_severity(classifier):
    assert classifier.should_notify_national("high")
    assert not classifier.should_notify_national("medium")
    assert classifier.should_notify_who("critical")


# ------------------------------------------------------------ recommendations
def test_recommendations_are_specific_and_owned(cholera_config, small_region, sample_prediction):
    alert = build_alert(sample_prediction, 0.72, 0.6, 6)
    recommendations = RecommendationEngine(cholera_config, small_region).build(alert)
    assert recommendations
    assert all(r.timeframe_days > 0 and r.responsible for r in recommendations)
    assert any("ORS" in r.action for r in recommendations)
    assert any(r.quantity for r in recommendations)


def test_higher_levels_inherit_lower_level_actions(cholera_config, small_region, sample_prediction):
    engine = RecommendationEngine(cholera_config, small_region)
    medium = sample_prediction.model_copy(update={"risk_level": "medium"})
    critical = sample_prediction.model_copy(update={"risk_level": "critical"})
    medium_actions = {r.action for r in engine.build(build_alert(medium, 0.3, 0.2, 6))}
    critical_actions = {r.action for r in engine.build(build_alert(critical, 3.0, 1.5, 6))}
    assert len(critical_actions) > len(medium_actions)


def test_critical_alerts_get_the_tightest_deadline(cholera_config, small_region, sample_prediction):
    engine = RecommendationEngine(cholera_config, small_region)
    critical = sample_prediction.model_copy(update={"risk_level": "critical"})
    recommendations = engine.build(build_alert(critical, 3.0, 1.5, 6))
    assert min(r.timeframe_days for r in recommendations) <= 3


def test_importation_produces_a_corridor_recommendation(cholera_config, small_region, sample_prediction):
    alert = build_alert(sample_prediction, 0.72, 0.6, 6)
    recommendations = RecommendationEngine(cholera_config, small_region).build(alert)
    assert any("Mwanza City" in r.action for r in recommendations)


def test_low_confidence_puts_verification_first(cholera_config, small_region, sample_prediction):
    flagged = sample_prediction.model_copy(
        update={"data_quality_flags": ["LOW DATA CONFIDENCE — verify with field reports."]}
    )
    alert = build_alert(flagged, 0.72, 0.6, 6)
    recommendations = RecommendationEngine(cholera_config, small_region).build(alert)
    assert "Verify this signal" in recommendations[0].action


def test_cholera_module_adds_a_water_point_action(small_region, sample_prediction):
    from src.models.registry import build_module

    module = build_module("cholera", region=small_region)
    alert = build_alert(sample_prediction, 0.72, 0.6, 6)
    actions = [r.action for r in module.generate_recommendations(alert)]
    assert any("water points" in a for a in actions)


# ----------------------------------------------------------------- escalation
def test_sustained_risk_escalates(cholera_config):
    engine = EscalationEngine()
    now = datetime.utcnow()
    history = [
        Alert(alert_id=str(i), disease="Cholera", district="X", region="Y", issued_at=now - timedelta(weeks=i),
              target_week=f"2026-W{i:02d}", risk_level="high", risk_score=0.8, predicted_cases=100,
              predicted_incidence_per_1000=0.7, threshold_crossed=0.6)
        for i in range(1, 4)
    ]
    alert = Alert(alert_id="new", disease="Cholera", district="X", region="Y", issued_at=now,
                  target_week="2026-W05", risk_level="high", risk_score=0.8, predicted_cases=100,
                  predicted_incidence_per_1000=0.7, threshold_crossed=0.6)
    escalated = engine.apply(alert, history)
    assert escalated.risk_level == "critical"
    assert escalated.escalated and "consecutive" in escalated.escalation_reason


def test_notify_targets_widen_with_severity():
    engine = EscalationEngine()
    base = dict(disease="D", district="X", region="Y", issued_at=datetime.utcnow(),
                target_week="2026-W01", risk_score=0.5, predicted_cases=1.0,
                predicted_incidence_per_1000=0.1, threshold_crossed=0.1)
    low = engine.notify_targets(Alert(alert_id="1", risk_level="low", **base))
    critical = engine.notify_targets(Alert(alert_id="2", risk_level="critical", **base))
    assert low == ["district_health_officer"]
    assert "national_eoc" in critical and "who_country_office" in critical


# ------------------------------------------------------------------ generation
def test_generator_emits_and_deduplicates(cholera_config, small_region, sample_prediction, tmp_path):
    from config.settings import get_settings

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    generator = AlertGenerator(cholera_config, small_region, settings=settings)
    alerts = generator.generate([sample_prediction], min_level="medium")
    assert alerts
    assert generator.persist(alerts) == len(alerts)
    # The same prediction inside the dedup window produces nothing new.
    assert generator.generate([sample_prediction], min_level="medium") == []


def test_acknowledgement_round_trips(cholera_config, small_region, sample_prediction, tmp_path):
    from config.settings import get_settings

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    generator = AlertGenerator(cholera_config, small_region, settings=settings)
    alerts = generator.generate([sample_prediction], min_level="medium")
    generator.persist(alerts)
    assert generator.acknowledge(alerts[0].alert_id, by="dho@example.org")
    reloaded = generator.load_all()
    assert any(a.acknowledged and a.acknowledged_by == "dho@example.org" for a in reloaded)


# ---------------------------------------------------------------- notification
def test_unreachable_channels_queue_instead_of_dropping(sample_prediction, tmp_path):
    """Rule #6: an alert is never lost because the link was down."""
    from config.settings import get_settings

    settings = get_settings().model_copy(update={"data_dir": tmp_path, "smtp_host": None})
    service = NotificationService(settings=settings)
    alert = build_alert(sample_prediction, 0.72, 0.6, 6)
    report = service.send(alert)
    assert "log" in report.delivered
    assert "email" in report.queued
    assert service.pending()


def test_email_body_contains_the_full_decision_package(sample_prediction, small_region, cholera_config):
    alert = build_alert(sample_prediction, 0.72, 0.6, 6)
    alert.recommendations = RecommendationEngine(cholera_config, small_region).build(alert)
    body = NotificationService().render_email(alert)
    for section in ("WHY THIS ALERT", "TOP DRIVERS", "RECOMMENDED ACTIONS", "IMPORTATION SOURCES"):
        assert section in body
