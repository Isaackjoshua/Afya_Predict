"""Feature engineering: fitted lags, leakage safety and provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.types import FeatureSpec
from src.feature_engineering.builder import FeatureBuilder
from src.feature_engineering.interaction_terms import apply_response_shape
from src.feature_engineering.lag_features import (
    add_lag_features,
    fit_optimal_lags,
    lag_dispersion,
    lag_fit_report,
)
from src.feature_engineering.mobility_features import importation_pressure, top_source_districts
from src.feature_engineering.one_health_features import add_one_health_features
from src.feature_engineering.rolling_stats import add_anomalies, add_rolling_means
from src.feature_engineering.seasonality import add_seasonality


@pytest.fixture(scope="module")
def synthetic_panel_values():
    """Two districts where cases follow rainfall at a known 5-week lag."""
    weeks = [f"2022-W{w:02d}" for w in range(1, 53)] + [f"2023-W{w:02d}" for w in range(1, 53)]
    rows = []
    for district, lag in (("A", 5), ("B", 9)):
        rng = np.random.default_rng(hash(district) % 1000)
        rain = 40 + 35 * np.sin(np.linspace(0, 8 * np.pi, len(weeks))) + rng.normal(0, 4, len(weeks))
        cases = np.roll(rain, lag) * 3 + rng.normal(0, 6, len(weeks))
        for i, week in enumerate(weeks):
            rows.append({"district": district, "week": week, "rainfall_mm": rain[i],
                         "cases_malaria": max(cases[i], 0)})
    return pd.DataFrame(rows).set_index(["district", "week"]).sort_index()


def test_lag_features_shift_within_district(synthetic_panel_values):
    lagged = add_lag_features(synthetic_panel_values, ["rainfall_mm"], [1, 3])
    a = lagged.xs("A", level="district")
    assert a["rainfall_mm_lag1"].iloc[1] == pytest.approx(a["rainfall_mm"].iloc[0])
    # District B's first row must be NaN, not district A's last value.
    assert pd.isna(lagged.xs("B", level="district")["rainfall_mm_lag1"].iloc[0])


def test_lag_fitter_recovers_a_known_lag(synthetic_panel_values):
    """Critical rule #3: the lag is discovered, not taken from the config."""
    spec = FeatureSpec(
        name="rainfall", source="chirps", lag_weeks_range=(1, 14),
        optimal_lag_weeks=2, mechanism="test",
    )
    fits = fit_optimal_lags(
        synthetic_panel_values, "cases_malaria", [spec], {"rainfall": "rainfall_mm"}
    )
    # The prior said 2; the data says 5 and 9.
    assert abs(fits["A"]["rainfall"].lag_weeks - 5) <= 1
    assert abs(fits["B"]["rainfall"].lag_weeks - 9) <= 1
    assert fits["A"]["rainfall"].source == "fitted"


def test_lag_dispersion_shows_districts_disagree(synthetic_panel_values):
    """Shortcoming #8: a single transplanted coefficient would be wrong."""
    spec = FeatureSpec(name="rainfall", source="chirps", lag_weeks_range=(1, 14),
                       optimal_lag_weeks=2, mechanism="test")
    fits = fit_optimal_lags(
        synthetic_panel_values, "cases_malaria", [spec], {"rainfall": "rainfall_mm"}
    )
    dispersion = lag_dispersion(fits)
    assert dispersion.loc[0, "unique_lags"] > 1
    assert not lag_fit_report(fits).empty


def test_short_series_fall_back_to_the_pooled_lag(synthetic_panel_values):
    spec = FeatureSpec(name="rainfall", source="chirps", lag_weeks_range=(1, 10),
                       optimal_lag_weeks=2, mechanism="test")
    short = synthetic_panel_values.head(20).copy()
    fits = fit_optimal_lags(short, "cases_malaria", [spec], {"rainfall": "rainfall_mm"})
    assert all(f["rainfall"].source in ("pooled", "prior") for f in fits.values())


def test_rolling_and_anomaly_features_are_backward_looking(synthetic_panel_values):
    frame = add_anomalies(add_rolling_means(synthetic_panel_values, ["rainfall_mm"], [4]), ["rainfall_mm"])
    a = frame.xs("A", level="district")
    assert a["rainfall_mm_roll4"].iloc[0:1].isna().all()
    # The baseline is shifted, so it cannot contain the current observation.
    assert a["rainfall_mm_anomaly"].iloc[:8].isna().all()


def test_seasonality_terms_are_bounded_and_cyclic(synthetic_panel_values):
    frame = add_seasonality(synthetic_panel_values, fourier_order=2)
    assert frame["week_sin"].between(-1, 1).all()
    assert set(frame["season_masika"].unique()) <= {0.0, 1.0}
    assert "fourier_cos2" in frame.columns


@pytest.mark.parametrize(
    "relationship,params",
    [
        ("positive_with_saturation", {"saturation_threshold_mm": 150}),
        ("bell_curve", {"optimal_range_celsius": [26, 28]}),
        ("threshold", {"threshold_mm": 80}),
        ("negative_linear", {}),
    ],
)
def test_response_shapes_are_finite_and_named(relationship, params):
    spec = FeatureSpec(name="x", source="s", relationship=relationship, mechanism="m", **params)
    values = pd.Series(np.linspace(0, 300, 50))
    shaped = apply_response_shape(values, spec)
    assert np.isfinite(shaped).all()
    assert shaped.name == "x_shaped"


def test_saturating_rainfall_declines_past_its_threshold():
    spec = FeatureSpec(name="rainfall", source="chirps",
                       relationship="positive_with_saturation",
                       mechanism="m", saturation_threshold_mm=150)
    shaped = apply_response_shape(pd.Series([50.0, 150.0, 400.0]), spec)
    assert shaped.iloc[1] > shaped.iloc[0]     # rises to the threshold
    assert shaped.iloc[2] < shaped.iloc[1]     # then breeding sites wash out


def test_bell_curve_peaks_inside_the_optimal_range():
    spec = FeatureSpec(name="temperature", source="era5", relationship="bell_curve",
                       mechanism="m", optimal_range_celsius=[26, 28])
    shaped = apply_response_shape(pd.Series([18.0, 27.0, 36.0]), spec)
    assert shaped.iloc[1] > shaped.iloc[0] and shaped.iloc[1] > shaped.iloc[2]


def test_importation_pressure_is_flow_weighted(small_region, panel):
    pressure = importation_pressure(panel.values(), small_region, "cases_malaria", lag_weeks=1)
    assert pressure.notna().any()
    assert (pressure.dropna() >= 0).all()


def test_top_source_districts_ranks_contributors(small_region, panel):
    week = panel.weeks[-5]
    sources = top_source_districts(
        panel.values(), small_region, "Kinondoni", week, "cases_malaria"
    )
    assert not sources.empty
    assert "Kinondoni" not in set(sources["district"])
    assert sources["contributed_risk"].is_monotonic_decreasing


def test_one_health_features_join_the_three_domains(panel):
    frame = add_one_health_features(panel.values(), case_column="cases_malaria")
    for column in ("one_health_animal_anomaly", "one_health_environment_anomaly"):
        assert column in frame.columns or "livestock_outbreak_events" not in panel.value_columns


def test_builder_produces_a_forward_shifted_target(panel, malaria_config, small_region):
    matrix = FeatureBuilder(malaria_config, small_region).build(panel, horizon_weeks=4)
    district = matrix.districts[0]
    target = matrix.y.xs(district, level="district")
    cases = panel.values()["cases_malaria"].xs(district, level="district")
    weeks = list(cases.index)
    # y at week t is the case count at t+4.
    assert target.loc[weeks[0]] == cases.loc[weeks[4]] or pd.isna(target.loc[weeks[0]])
    assert target.iloc[-4:].isna().all()  # the final horizon has no known future


def test_builder_never_leaks_the_contemporaneous_target(panel, malaria_config, small_region):
    matrix = FeatureBuilder(malaria_config, small_region).build(panel, horizon_weeks=4)
    assert "cases_malaria" not in matrix.feature_names
    leaks = [f for f in matrix.feature_names if f.startswith("cases_malaria")]
    # Only explicitly lagged/rolled history is allowed.
    assert all(any(tag in f for tag in ("lag", "roll", "trend", "yoy", "nbr")) for f in leaks)


def test_every_feature_carries_its_mechanism(panel, malaria_config, small_region):
    """Shortcoming #4: explanations need provenance, not column indices."""
    matrix = FeatureBuilder(malaria_config, small_region).build(panel)
    assert matrix.feature_names
    for name in matrix.feature_names:
        record = matrix.describe_feature(name)
        assert record["kind"] != "unknown"
        assert record.get("mechanism")


def test_dropna_rows_keeps_only_usable_rows(panel, malaria_config, small_region):
    matrix = FeatureBuilder(malaria_config, small_region).build(panel).dropna_rows()
    assert matrix.y.notna().all()
    assert len(matrix.X) == len(matrix.y)
