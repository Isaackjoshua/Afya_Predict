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


# ====================================================================
# Served routes against a POPULATED cache
# ====================================================================
# Every prediction test above is conditional on the cache having content, so on
# a fresh node they assert an empty-response shape and the route bodies never
# execute. That is precisely the blind spot that let three bugs reach a running
# stack: predictions keyed by display name (so every `?disease=` filter returned
# nothing), the per-district route 404ing, and /explain attributing every SHAP
# contribution to an "unknown" source. The fixture below seeds real forecasts so
# those paths are exercised rather than skipped.


@pytest.fixture(scope="module")
def seeded(panel, small_region, tmp_path_factory):
    """Real trained forecasts for two diseases, wired into the API's cache.

    The modules handed to the routes are *reloaded from disk* with no feature
    matrix, which is the state a served API is actually in: weights loaded, the
    feature pipeline never built.
    """
    import importlib

    from offline.local_cache import LocalCache
    from src.models.registry import build_module

    workdir = tmp_path_factory.mktemp("api-seed")
    cache = LocalCache(workdir / "afya.sqlite")
    served: dict = {}

    for slug in ("malaria", "cholera"):
        module = build_module(slug, region=small_region)
        matrix = module.build_feature_matrix(panel)
        module.train(matrix)
        saved_to = module.save(workdir / f"{slug}.pkl")

        predictions = []
        for district in matrix.districts[:3]:
            predictions.extend(module.predict(matrix, district, panel=panel))
        assert cache.save_predictions(predictions) == len(predictions)
        cache.save_alerts(module.detect_outbreak(predictions))

        reloaded = build_module(slug, region=small_region)
        assert reloaded.load(saved_to)
        served[slug] = reloaded

    from src.api import dependencies

    real_get_module = dependencies.get_module

    def fake_get_module(slug: str):
        return served.get(slug) or real_get_module(slug)

    targets = [
        "src.api.dependencies",
        "src.api.routes.predictions",
        "src.api.routes.alerts",
        "src.api.routes.explainability",
        "src.api.routes.interventions",
        "src.api.routes.admin",
    ]
    with pytest.MonkeyPatch.context() as mp:
        for name in targets:
            module_obj = importlib.import_module(name)
            if hasattr(module_obj, "get_cache"):
                mp.setattr(module_obj, "get_cache", lambda cache=cache: cache)
            if hasattr(module_obj, "get_module"):
                mp.setattr(module_obj, "get_module", fake_get_module)
        yield cache


def test_disease_filter_returns_only_that_disease(client, seeded):
    """The bug: predictions were stored under the display name, so this was empty.

    The unfiltered listing looked perfectly healthy throughout, which is why
    nothing caught it until the stack was queried over HTTP.
    """
    unfiltered = client.get("/predictions", params={"limit": 1000}).json()
    assert unfiltered["count"] > 0

    seen = set()
    for slug in ("malaria", "cholera"):
        payload = client.get("/predictions", params={"disease": slug, "limit": 1000}).json()
        assert payload["count"] > 0, f"?disease={slug} returned nothing"
        assert {p["disease"] for p in payload["predictions"]} == {slug}
        seen.update(p["prediction_id"] for p in payload["predictions"])

    # The per-disease slices must account for the whole cache, not a subset.
    assert seen == {p["prediction_id"] for p in unfiltered["predictions"]}


def test_served_predictions_carry_both_slug_and_display_name(client, seeded):
    """`disease` is what callers filter on; `disease_name` is what people read."""
    from src.core.config_loader import load_disease_config

    payload = client.get("/predictions", params={"disease": "malaria", "limit": 5}).json()
    for prediction in payload["predictions"]:
        assert prediction["disease"] == "malaria"
        assert prediction["disease_name"] == load_disease_config("malaria").name
        assert " " not in prediction["disease"]


def test_per_district_route_serves_a_forecast(client, seeded):
    """The bug: this 404'd for every district because the slug never matched."""
    listing = client.get("/predictions", params={"disease": "malaria", "limit": 1}).json()
    district = listing["predictions"][0]["district"]

    response = client.get(f"/predictions/malaria/{district}")
    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["count"] > 0
    assert {p["district"] for p in payload["predictions"]} == {district}


def test_prediction_by_id_round_trips(client, seeded):
    """The bug: this route was unreachable.

    `/{disease}/{district}` is declared in the same router and Starlette matches
    in order, so every by-id lookup was handled by `district_predictions` with
    disease="id" and rejected as an unknown district. It still answered 404, so
    a test that only asserted the status code would have passed against it —
    hence the assertion on *why* an unknown id is refused.
    """
    listing = client.get("/predictions", params={"limit": 1}).json()
    prediction_id = listing["predictions"][0]["prediction_id"]

    payload = client.get(f"/predictions/id/{prediction_id}").json()
    assert payload["prediction"]["prediction_id"] == prediction_id
    assert payload["prediction"]["natural_language_explanation"]

    missing = client.get("/predictions/id/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Unknown prediction_id", (
        "the by-id route is being shadowed by /{disease}/{district} again"
    )


def test_national_map_carries_coordinates_and_risk(client, seeded):
    payload = client.get("/predictions/malaria/map/national").json()
    assert payload["disease"] == "malaria"
    assert payload["districts"]
    for entry in payload["districts"]:
        assert entry["lat"] is not None and entry["lon"] is not None
        assert entry["risk_level"] in ("low", "medium", "high", "critical")
        assert 0.0 <= entry["risk_score"] <= 1.0
    # One row per district, newest week only.
    names = [e["district"] for e in payload["districts"]]
    assert len(names) == len(set(names))


def test_explanation_attributes_contributions_to_real_sources(client, seeded):
    """The bug: /explain reported every contribution as coming from "unknown".

    The feature -> source map lived only on an in-memory FeatureMatrix, and a
    served API loads weights without ever building one — so the fusion evidence
    (shortcoming #2) was silently empty in production while the endpoint still
    returned 200.
    """
    listing = client.get("/predictions", params={"disease": "malaria", "limit": 1}).json()
    prediction_id = listing["predictions"][0]["prediction_id"]

    payload = client.get(f"/explain/{prediction_id}").json()
    assert payload["natural_language_explanation"]
    assert payload["shap_values"]

    contributions = payload["source_contributions"]
    assert contributions, "no source attribution at all"
    named = [c for c in contributions if c["source"] != "unknown"]
    assert len(named) >= 3, f"expected several real sources, got {contributions}"

    unknown_share = sum(c["share"] for c in contributions if c["source"] == "unknown")
    assert unknown_share < 0.5, f"most contribution is unattributed: {contributions}"
    assert abs(sum(c["share"] for c in contributions) - 1.0) < 1e-3


def test_waterfall_is_ordered_and_complete(client, seeded):
    listing = client.get("/predictions", params={"limit": 1}).json()
    prediction_id = listing["predictions"][0]["prediction_id"]

    payload = client.get(f"/explain/{prediction_id}/waterfall", params={"top_n": 5}).json()
    steps = payload["steps"]
    assert steps
    ranked = [abs(s["contribution"]) for s in steps if not s["feature"].endswith("other features")]
    assert ranked == sorted(ranked, reverse=True)
    for step in steps:
        assert step["direction"] in ("increases", "decreases")


def test_alerts_filter_by_slug(client, seeded):
    """Alerts had the same display-name defect, and are read by people."""
    payload = client.get("/alerts", params={"days": 365, "limit": 500}).json()
    if payload["count"] == 0:
        pytest.skip("the seeded panel produced no alerts")

    slugs = {a["disease"] for a in payload["alerts"]}
    assert slugs <= {"malaria", "cholera"}
    for slug in slugs:
        filtered = client.get(
            "/alerts", params={"disease": slug, "days": 365, "limit": 500}
        ).json()
        assert filtered["count"] > 0, f"?disease={slug} returned no alerts"
        assert {a["disease"] for a in filtered["alerts"]} == {slug}
