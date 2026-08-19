"""Explainability: local accuracy, provenance and readable output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.types import DriverExplanation
from src.explainability.counterfactuals import generate_counterfactuals, threshold_counterfactual
from src.explainability.feature_importance import (
    concentration_index,
    dominant_source_warning,
    global_importance,
    proxy_importance,
    source_importance,
)
from src.explainability.natural_language import (
    describe_driver,
    explain_prediction,
    sms_summary,
    summarise_drivers,
)
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.tree_shap import numpy_gbm_shap, permutation_shap
from src.models.gbm import NumpyGBMRegressor


@pytest.fixture(scope="module")
def toy_model():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=["rain", "temp", "noise1", "noise2", "noise3"])
    y = pd.Series(4 * X["rain"] + 2 * X["temp"] + rng.normal(0, 0.2, 300))
    model = NumpyGBMRegressor(n_estimators=60, max_depth=3).fit(X.to_numpy(), y.to_numpy())
    return model, X, y


def test_tree_shap_satisfies_local_accuracy(toy_model):
    """The property that makes a contribution share meaningful."""
    model, X, _ = toy_model
    values, base = numpy_gbm_shap(model, X.head(30).to_numpy())
    predicted = model.predict(X.head(30).to_numpy())
    assert np.allclose(base + values.sum(axis=1), predicted, atol=1e-6)


def test_tree_shap_ranks_the_true_drivers_first(toy_model):
    model, X, _ = toy_model
    values, _ = numpy_gbm_shap(model, X.head(50).to_numpy())
    ranking = np.argsort(np.abs(values).mean(axis=0))[::-1]
    assert set(ranking[:2].tolist()) == {0, 1}   # rain and temp


def test_permutation_shap_also_satisfies_local_accuracy(toy_model):
    model, X, _ = toy_model
    values, base = permutation_shap(
        model.predict, X.head(3).to_numpy(), X.head(60).to_numpy(), n_samples=25
    )
    assert np.allclose(base + values.sum(axis=1), model.predict(X.head(3).to_numpy()), atol=1e-6)


def test_explainer_attaches_mechanism_provenance(panel, malaria_config, small_region):
    from src.models.registry import build_module

    module = build_module("malaria", region=small_region)
    matrix = module.build_feature_matrix(panel)
    module.train(matrix)
    model = module.models["pooled"]

    rows = matrix.X.dropna(how="all").tail(3)
    explanation = ShapExplainer(model, background=matrix.X, provenance=matrix.provenance).explain(rows)
    drivers = explanation.top_drivers()
    assert drivers
    assert all(d.mechanism for d in drivers)
    assert 0 <= sum(d.contribution_share for d in drivers) <= 1.0001
    assert explanation.method in ("shap_package", "exact_tree_shap", "permutation_shap")


def test_importance_aggregates_to_proxies_and_sources(panel, malaria_config, small_region):
    from src.models.registry import build_module

    module = build_module("malaria", region=small_region)
    matrix = module.build_feature_matrix(panel)
    module.train(matrix)
    explanation = ShapExplainer(
        module.models["pooled"], background=matrix.X, provenance=matrix.provenance
    ).explain(matrix.X.dropna(how="all").tail(40))

    assert not global_importance(explanation).empty
    proxies = proxy_importance(explanation)
    sources = source_importance(explanation)
    assert not proxies.empty and not sources.empty
    # Shortcoming #2: no single source should carry the entire model.
    assert 0 < concentration_index(explanation) <= 1.0
    assert len(sources) >= 2


def test_dominant_source_warning_fires_on_a_single_source_model():
    from src.explainability.shap_explainer import ExplanationResult

    explanation = ExplanationResult(
        values=pd.DataFrame({"a": [1.0, 1.0], "b": [0.01, 0.01]}),
        base_value=0.0,
        method="test",
        provenance={"a": {"source": "chirps", "proxy": "rainfall"},
                    "b": {"source": "era5", "proxy": "temperature"}},
    )
    assert dominant_source_warning(explanation, threshold=0.7)


def test_driver_sentences_name_proxy_lag_direction_and_mechanism():
    driver = DriverExplanation(
        feature="rainfall_mm_lag6", proxy="rainfall", lag_weeks=6, value=142.0,
        shap_value=3.2, contribution_share=0.34, direction="increases",
        mechanism="rainfall creates mosquito breeding sites",
    )
    sentence = describe_driver(driver)
    assert "rainfall" in sentence.lower()
    assert "6 weeks ago" in sentence
    assert "34%" in sentence
    assert "breeding sites" in sentence


def test_explanation_flags_low_confidence_prominently():
    drivers = [
        DriverExplanation(feature="f", proxy="rainfall", lag_weeks=3, shap_value=1.0,
                          contribution_share=0.5, direction="increases", mechanism="m")
    ]
    text = explain_prediction(
        disease="Cholera", district="Mwanza City", target_week="2026-W10",
        predicted_cases=120.0, incidence_per_1000=0.26, risk_level="high",
        drivers=drivers, lead_time_weeks=6, ci=(40.0, 300.0),
        data_quality_flags=["DHIS2 reporting gap"], low_confidence=True,
    )
    assert "LOW DATA CONFIDENCE" in text
    assert "Mwanza City" in text and "2026-W10" in text


def test_sms_summary_fits_a_feature_phone():
    """Rule #14: ~87% of handsets are feature phones."""
    drivers = [
        DriverExplanation(feature="f", proxy="rainfall", lag_weeks=3, shap_value=1.0,
                          contribution_share=0.4, direction="increases", mechanism="m")
    ]
    body = sms_summary("Cholera", "Sengerema", "high", 240.0, "2026-W12", drivers,
                       top_action="Pre-position 5,000 ORS kits")
    assert len(body) <= 320
    assert "HIGH" in body and "Sengerema" in body and "ORS" in body


def test_counterfactuals_move_the_prediction(toy_model):
    model, X, _ = toy_model

    class Wrapper:
        feature_names = list(X.columns)

        def predict(self, frame):
            return model.predict(np.asarray(frame, dtype=float))

    drivers = [
        DriverExplanation(feature="rain", proxy="rainfall", lag_weeks=6, shap_value=2.0,
                          contribution_share=0.6, direction="increases", mechanism="m")
    ]
    scenarios = generate_counterfactuals(
        Wrapper(), X.iloc[10], drivers, provenance={"rain": {"kind": "lagged_proxy", "proxy": "rainfall"}}
    )
    assert scenarios
    assert abs(scenarios[0].delta) > 0
    assert "rainfall" in scenarios[0].to_sentence().lower()


def test_threshold_counterfactual_finds_a_crossing(toy_model):
    model, X, _ = toy_model

    class Wrapper:
        def predict(self, frame):
            return model.predict(np.asarray(frame, dtype=float))

    row = X.iloc[10]
    baseline = float(Wrapper().predict(row.to_frame().T)[0])
    crossing = threshold_counterfactual(Wrapper(), row, "rain", baseline + 1.0)
    assert crossing is None or np.isfinite(crossing)


def test_summarise_drivers_handles_an_empty_list():
    assert "no dominant driver" in summarise_drivers([])
