"""API contract tests.

Two things these assert that unit tests cannot:

* the service **boots on a fresh install** — no credentials, no cache, no trained
  models — and reports honestly what is missing rather than failing at import;
* a prediction served over HTTP still carries its explanation and its
  recommendations. Critical rule #2 has to hold at the API boundary too, or the
  transparency guarantee is only skin deep.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi is required for the API tests")
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# ------------------------------------------------------------------- meta
def test_root_advertises_the_docs(client):
    payload = client.get("/").json()
    assert payload["name"] == "AFYA-PREDICT"
    assert payload["license"] == "MIT"
    assert payload["docs"] == "/docs"


def test_health_reports_capability_honestly(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] in ("ok", "degraded")
    assert set(payload["diseases"]) >= {"malaria", "cholera", "tuberculosis",
                                        "respiratory", "hiv"}
    assert payload["districts"] > 50
    # A degraded instance must say *why*, not just flag itself degraded.
    if payload["status"] == "degraded":
        assert payload["warnings"]
    assert payload["backends"]["default_resolved"]


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "AFYA-PREDICT API"
    for path in ("/predictions", "/alerts", "/data/status", "/diseases",
                 "/interventions/log", "/admin/retrain"):
        assert path in schema["paths"], f"{path} missing from the OpenAPI schema"


def test_timing_header_is_attached(client):
    response = client.get("/health")
    assert "X-Response-Time-Ms" in response.headers
    assert float(response.headers["X-Response-Time-Ms"]) >= 0


# --------------------------------------------------------------- diseases
def test_disease_listing(client):
    payload = client.get("/diseases").json()
    assert payload["count"] >= 5
    slugs = {d["slug"] for d in payload["diseases"]}
    assert {"malaria", "cholera"} <= slugs
    for disease in payload["diseases"]:
        # Acceptance criterion #2: every disease fuses at least three sources.
        assert len(disease["sources"]) >= 3
        assert disease["horizon_weeks"] >= 1


def test_disease_config_publishes_every_mechanism(client):
    """Shortcoming #4: an agency should not need to read code to see the reasoning."""
    payload = client.get("/diseases/malaria/config").json()
    assert payload["validation_problems"] == []
    mechanisms = payload["proxy_mechanisms"]
    assert len(mechanisms) >= 3
    for proxy in mechanisms:
        assert proxy["mechanism"], f"{proxy['proxy']} has no stated mechanism"
        assert len(proxy["lag_search_range_weeks"]) == 2


def test_unknown_disease_is_a_clean_404(client):
    response = client.get("/diseases/smallpox")
    assert response.status_code == 404
    assert "Available" in response.json()["detail"]


# ------------------------------------------------------------ data status
def test_data_status_reports_the_fusion_rule(client):
    payload = client.get("/data/status").json()
    assert payload["sources"]
    assert isinstance(payload["meets_fusion_rule"], bool)
    if not payload["meets_fusion_rule"]:
        # Rule #1: falling short must be stated, not silently tolerated.
        assert any("rule #1" in w or "at least 3" in w for w in payload["warnings"])


def test_source_listing_marks_optional_sources(client):
    sources = client.get("/data/sources").json()
    by_name = {s["source"]: s for s in sources}
    # Rule #14: search/social can never be a primary predictor.
    assert by_name["search_trends"]["optional"] is True
    assert by_name["chirps"]["optional"] is False


# ------------------------------------------------------------- predictions
def test_prediction_listing_is_shaped_correctly(client):
    payload = client.get("/predictions", params={"limit": 5}).json()
    assert "count" in payload and "predictions" in payload
    if payload["count"] == 0:
        # An empty cache must explain how to populate it, not just return nothing.
        assert payload["warnings"]


def test_served_predictions_keep_their_explanation(client):
    """Critical rule #2 at the HTTP boundary."""
    payload = client.get("/predictions", params={"limit": 5}).json()
    if not payload["predictions"]:
        pytest.skip("no cached predictions on this node")
    for prediction in payload["predictions"]:
        assert prediction["natural_language_explanation"]
        assert prediction["top_drivers"]
        assert prediction["shap_values"]
        assert prediction["confidence_interval_lower"] <= prediction["predicted_cases"]
        assert prediction["predicted_cases"] <= prediction["confidence_interval_upper"]
        for driver in prediction["top_drivers"]:
            assert driver["mechanism"], "a driver without a mechanism is not an explanation"


def test_unknown_district_is_rejected(client):
    response = client.get("/predictions/malaria/Atlantis")
    assert response.status_code == 404


def test_prediction_read_is_fast(client):
    """Acceptance criterion #10: predictions in under 2 seconds."""
    import time

    started = time.perf_counter()
    response = client.get("/predictions", params={"limit": 50})
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < 2.0, f"prediction read took {elapsed:.2f}s (budget 2s)"


# ------------------------------------------------------------------ alerts
def test_alert_listing_and_summary(client):
    payload = client.get("/alerts", params={"days": 365}).json()
    assert "count" in payload and "alerts" in payload
    summary = client.get("/alerts/summary/by-district", params={"days": 365}).json()
    assert "districts" in summary


def test_active_alerts_are_ordered_by_severity(client):
    payload = client.get("/alerts/active").json()
    levels = [a["risk_level"] for a in payload["alerts"]]
    order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    assert levels == sorted(levels, key=lambda l: order[l], reverse=True)


def test_acknowledging_an_unknown_alert_is_a_404(client):
    response = client.post(
        "/alerts/does-not-exist/acknowledge", json={"acknowledged_by": "tester"}
    )
    assert response.status_code == 404


# ----------------------------------------------------------- interventions
def test_intervention_round_trip(client):
    payload = {
        "disease": "malaria", "district": "Kinondoni",
        "intervention_type": "llin_distribution",
        "coverage": 0.6, "quantity": 25000, "unit": "nets",
        "logged_by": "pytest", "notes": "API contract test",
    }
    created = client.post("/interventions/log", json=payload)
    assert created.status_code == 201
    intervention = created.json()["intervention"]
    assert intervention["intervention_id"]
    assert intervention["coverage"] == 0.6

    listed = client.get("/interventions", params={"district": "Kinondoni"}).json()
    assert any(i["intervention_id"] == intervention["intervention_id"]
               for i in listed["interventions"])


def test_intervention_coverage_is_bounded(client):
    response = client.post("/interventions/log", json={
        "disease": "malaria", "district": "Kinondoni",
        "intervention_type": "llin_distribution", "coverage": 5.0,
    })
    assert response.status_code == 422    # coverage is a 0-1 share


def test_intervention_types_carry_an_effect_lag(client):
    types = client.get("/interventions/types").json()
    assert types
    for entry in types:
        assert entry["effect_lag_weeks"] >= 0


def test_response_audit_is_available(client):
    audit = client.get("/interventions/audit/responses", params={"days": 365}).json()
    assert "total_alerts" in audit
    assert "interpretation" in audit


# ------------------------------------------------------------------- admin
def test_registry_validation_endpoint(client):
    payload = client.get("/admin/registry/validate").json()
    assert payload["valid"] is True, payload["problems"]


def test_offline_status_reports_readiness(client):
    payload = client.get("/admin/offline/status").json()
    assert "ready" in payload and "weeks_cached" in payload
    assert payload["required_weeks"] == 2      # acceptance criterion #12


def test_drift_endpoint_reports_without_a_trained_model(client):
    response = client.get("/admin/drift/malaria")
    assert response.status_code == 200
    payload = response.json()
    assert "should_retrain" in payload and "reasons" in payload


# ------------------------------------------------------------- middleware
def test_rate_limit_headers_and_cors(client):
    response = client.options(
        "/predictions",
        headers={"Origin": "http://example.org", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code in (200, 204)


def test_api_key_is_enforced_when_configured(monkeypatch):
    """Open by default, closed once API_KEY is set."""
    from config.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "api_key", "secret-token", raising=False)

    with TestClient(app) as guarded:
        assert guarded.get("/health").status_code == 200          # public path
        assert guarded.get("/predictions").status_code == 401     # protected
        assert guarded.get(
            "/predictions", headers={"X-API-Key": "secret-token"}
        ).status_code == 200
