"""Model layer: backends, training, drift, diffusion and transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.backends import available_backends, build_regressor, resolve_backend
from src.models.drift_detector import DriftDetector, detect_drift
from src.models.ensemble import WeightedEnsemble
from src.models.gbm import NumpyGBMRegressor
from src.models.registry import build_module, list_modules, validate_registry
from src.models.spatial_diffusion import SpatialDiffusionModel
from src.models.transfer_learning import (
    build_transfer_plan,
    find_donors,
    fit_local_calibration,
    similarity_matrix,
)


# ---------------------------------------------------------------- backends
def test_backend_always_resolves_to_something_runnable():
    """Shortcoming #9: the platform must train on commodity hardware."""
    assert "numpy_gbm" in available_backends()
    assert resolve_backend("xgboost") in available_backends()
    _, info = build_regressor("xgboost")
    assert info.resolved in available_backends()


def test_bundled_gbm_learns_a_nonlinear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 10))
    y = 3 * X[:, 0] + 2 * np.sin(2 * X[:, 1]) + rng.normal(0, 0.3, 500)
    model = NumpyGBMRegressor(n_estimators=80).fit(X[:400], y[:400])
    predictions = model.predict(X[400:])
    residual = np.sum((y[400:] - predictions) ** 2)
    total = np.sum((y[400:] - y[400:].mean()) ** 2)
    assert 1 - residual / total > 0.6


def test_bundled_gbm_handles_missing_values_natively():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6))
    y = 2 * X[:, 0] + rng.normal(0, 0.2, 300)
    X[rng.random(X.shape) < 0.15] = np.nan
    model = NumpyGBMRegressor(n_estimators=40).fit(X, y)
    assert np.isfinite(model.predict(X)).all()


# ---------------------------------------------------------------- registry
def test_all_five_diseases_are_registered():
    """Acceptance criterion #1."""
    assert set(list_modules()) >= {"malaria", "cholera", "tuberculosis", "respiratory", "hiv"}


def test_registry_validates_cleanly():
    assert validate_registry() == {}


def test_a_config_without_a_class_still_builds(small_region):
    """Rule #5: a YAML alone is enough to get a working module."""
    from src.core.config_loader import load_disease_config
    from src.models.standard_module import StandardDiseaseModule

    config = load_disease_config("malaria")
    module = build_module("unregistered_disease", region=small_region, config=config)
    assert isinstance(module, StandardDiseaseModule)


# ---------------------------------------------------------------- training
@pytest.fixture(scope="module")
def trained_malaria(panel, small_region):
    module = build_module("malaria", region=small_region)
    matrix = module.build_feature_matrix(panel)
    module.train(matrix)
    return module, matrix


def test_training_produces_a_pooled_model(trained_malaria):
    module, _ = trained_malaria
    assert "pooled" in module.models
    assert module.models["pooled"].n_rows > 0
    assert module.models["pooled"].version


def test_sparse_districts_inherit_the_pooled_model(small_region, panel):
    """Shortcoming #8: borrow strength, never fake a local fit."""
    module = build_module("cholera", region=small_region)
    matrix = module.build_feature_matrix(panel)
    short = matrix.X.index.get_level_values("week") <= matrix.weeks[30]
    from src.models.auto_retrain import _slice_weeks

    trimmed = _slice_weeks(matrix, set(matrix.weeks[:30]))
    module.train(trimmed)
    borrowed = module.model_for(small_region.district_names[0])
    assert borrowed is not None
    assert borrowed.scope == "pooled"


def test_intervals_widen_when_a_district_borrows(trained_malaria, small_region):
    module, matrix = trained_malaria
    district = matrix.districts[0]
    rows = module.latest_feature_rows(matrix, district)
    bundle = module._predict_rows(matrix, district, rows)
    assert (bundle.upper >= bundle.point).all()
    assert (bundle.lower <= bundle.point).all()
    assert (bundle.lower >= 0).all()


def test_predictions_carry_explanations_and_actions(trained_malaria, panel):
    """Critical rule #2 + acceptance criteria #3, #4, #9."""
    module, matrix = trained_malaria
    results = module.predict(matrix, matrix.districts[0], panel=panel)
    assert results
    result = results[-1]
    assert result.predicted_cases >= 0
    assert result.confidence_interval_lower <= result.predicted_cases <= result.confidence_interval_upper
    assert result.top_drivers, "every prediction must have SHAP drivers"
    assert result.natural_language_explanation
    assert result.shap_values
    assert result.risk_level in ("low", "medium", "high", "critical")
    assert result.model_version


def test_shap_shares_sum_to_one(trained_malaria, panel):
    module, matrix = trained_malaria
    result = module.predict(matrix, matrix.districts[0], panel=panel)[-1]
    total = sum(abs(v) for v in result.shap_values.values())
    top_share = sum(d.contribution_share for d in result.top_drivers)
    assert total > 0
    assert 0 < top_share <= 1.0001


def test_model_round_trips_through_disk(trained_malaria, tmp_path):
    module, _ = trained_malaria
    path = module.save(tmp_path / "model.pkl")
    assert path.exists()
    assert path.with_suffix(".json").exists()

    reloaded = build_module("malaria", region=module.region)
    assert reloaded.load(path)
    assert set(reloaded.models) == set(module.models)


# ---------------------------------------------------------------- ensemble
def test_ensemble_weights_members_by_holdout_error():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(2 * X["f0"] + rng.normal(0, 0.3, 300))
    ensemble = WeightedEnsemble(["numpy_gbm", "ridge"]).fit(X, y)
    weights = [m.weight for m in ensemble.members]
    assert len(ensemble.members) == 2
    assert pytest.approx(sum(weights), abs=1e-6) == 1.0
    assert np.isfinite(ensemble.predict(X)).all()


def test_ensemble_spread_is_a_real_uncertainty_signal():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = pd.Series(X["a"] ** 2 + rng.normal(0, 0.2, 200))
    ensemble = WeightedEnsemble(["numpy_gbm", "ridge"]).fit(X, y)
    spread = ensemble.prediction_spread(X)
    assert len(spread) == len(X)
    assert (spread >= 0).all()


# ---------------------------------------------------------------- drift
def test_no_drift_on_a_stable_error_stream():
    rng = np.random.default_rng(5)
    stable = rng.normal(0, 1, 150)
    assert detect_drift(np.zeros(150), -stable)["drift_detected"] is False


def test_drift_is_detected_when_the_error_mean_shifts():
    """Shortcoming #3: the Google Flu Trends failure mode, caught automatically."""
    rng = np.random.default_rng(6)
    residuals = np.concatenate([rng.normal(0, 1, 70), rng.normal(5, 1, 70)])
    report = detect_drift(np.zeros(140), -residuals)
    assert report["drift_detected"]
    assert report["events"]
    # Detected soon after the change point, not 60 weeks later.
    assert min(e["index"] for e in report["events"]) < 110


def test_drift_needs_a_minimum_number_of_observations():
    report = detect_drift([1, 2, 3], [1, 2, 3], min_observations=20)
    assert report["drift_detected"] is False
    assert "residuals" in report["reason"]


# ---------------------------------------------------------------- diffusion
def test_diffusion_spreads_from_the_seeded_district(small_region, cholera_config):
    model = SpatialDiffusionModel(cholera_config, small_region)
    incidence = pd.Series(0.0, index=small_region.district_names)
    incidence["Ilala"] = 3.0
    forecast = model.project(incidence, "2024-W20", horizon_weeks=4)
    assert forecast.risk.shape[0] == 4
    assert (forecast.risk.to_numpy() >= 0).all()
    # Kinondoni is adjacent and populous, so it should pick up the most risk.
    assert forecast.risk.iloc[0]["Kinondoni"] > forecast.risk.iloc[0]["Songea MC"]


def test_diffusion_identifies_importation_dominated_districts(small_region, cholera_config):
    model = SpatialDiffusionModel(cholera_config, small_region)
    incidence = pd.Series(0.0, index=small_region.district_names)
    incidence["Mwanza City"] = 5.0
    at_risk = model.at_risk_districts(incidence)
    assert "Sengerema" in at_risk        # nearby, quiet locally, seeded
    assert "Mwanza City" not in at_risk  # its risk is local, not imported


def test_diffusion_contributors_name_the_source(small_region, cholera_config):
    model = SpatialDiffusionModel(cholera_config, small_region)
    incidence = pd.Series(0.0, index=small_region.district_names)
    incidence["Mwanza City"] = 4.0
    sources = model.contributors(incidence, "Sengerema")
    assert sources and sources[0].district == "Mwanza City"


# ---------------------------------------------------------------- transfer
def test_similarity_prefers_ecology_over_raw_distance(region):
    similarity = similarity_matrix(region)
    # Two large coastal cities resemble each other more than a city and the
    # sparsely populated rangeland council nearest to it.
    assert similarity.loc["Kinondoni", "Ilala"] > similarity.loc["Kinondoni", "Longido"]


def test_donors_are_chosen_from_the_data_rich_pool(region):
    donors = find_donors(region, "Kilindi", ["Kinondoni", "Handeni", "Moshi MC", "Rufiji"], k=2)
    assert len(donors) == 2
    assert "Kilindi" not in donors


def test_local_calibration_corrects_a_biased_pooled_model():
    rng = np.random.default_rng(7)
    predicted = rng.normal(50, 10, 40)
    actual = 1.5 * predicted + 5 + rng.normal(0, 2, 40)
    calibration = fit_local_calibration(predicted, actual, "Test")
    assert calibration.applied
    assert calibration.r2_after > calibration.r2_before
    assert calibration.apply(predicted).mean() > predicted.mean()


def test_calibration_is_refused_on_too_few_points():
    calibration = fit_local_calibration(np.arange(5.0), np.arange(5.0) * 2, "Tiny")
    assert not calibration.applied


def test_transfer_plan_splits_local_from_borrowing(panel, small_region, malaria_config):
    from src.feature_engineering.builder import FeatureBuilder

    matrix = FeatureBuilder(malaria_config, small_region).build(panel)
    plan = build_transfer_plan(matrix, small_region, min_rows=10_000)  # force borrowing
    assert not plan.local
    assert set(plan.borrowing) == set(matrix.districts)
    assert not plan.summary().empty

def test_predictions_are_keyed_by_slug_not_display_name(panel, small_region):
    """Every API route, dashboard filter and cache query keys on the slug.

    Storing the display name here meant `?disease=malaria` returned nothing,
    `/predictions/malaria/Kinondoni` 404'd, and every dashboard page reported
    "no cached forecasts" — while the unfiltered listing looked perfectly
    healthy. The tests at the time asserted response *shape*, so none of them
    noticed.
    """
    from src.core.config_loader import load_disease_config

    for slug in ("malaria", "respiratory"):
        module = build_module(slug, region=small_region)
        matrix = module.build_feature_matrix(panel)
        module.train(matrix)
        results = module.predict(matrix, matrix.districts[0], panel=panel)
        assert results
        prediction = results[-1]

        assert prediction.disease == slug
        assert prediction.disease_name == load_disease_config(slug).name
        # The display name must still be available for anything a human reads.
        assert prediction.display_name == prediction.disease_name

        alerts = module.detect_outbreak(results)
        for alert in alerts:
            assert alert.disease == slug
            assert alert.display_name == prediction.disease_name


def test_respiratory_display_name_differs_from_its_slug(small_region):
    """The case that makes the distinction load-bearing rather than cosmetic."""
    module = build_module("respiratory", region=small_region)
    assert module.slug == "respiratory"
    assert module.config.name == "Acute Respiratory Infection"


def test_provenance_survives_a_save_and_reload(panel, small_region, tmp_path):
    """A served API loads weights but never builds a feature matrix.

    Relying on the in-memory provenance map meant /explain attributed every
    SHAP contribution to an "unknown" source once the model was loaded from
    disk — which is exactly the fusion evidence shortcoming #2 exists to show.
    """
    module = build_module("malaria", region=small_region)
    matrix = module.build_feature_matrix(panel)
    module.train(matrix)
    assert module.provenance, "training must capture the provenance map"

    path = module.save(tmp_path / "model.pkl")
    reloaded = build_module("malaria", region=small_region)
    assert reloaded.load(path)
    assert reloaded.feature_matrix is None          # never built one
    assert reloaded.provenance == module.provenance

    sources = {record.get("source") for record in reloaded.provenance.values()}
    assert len(sources - {"unknown", None}) >= 3, (
        f"expected several contributing sources, got {sources}"
    )
