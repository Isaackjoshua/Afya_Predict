"""Adapter contract, quality flagging and panel fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion.base_adapter import FetchMode, TIDY_COLUMNS
from src.data_ingestion.normalizer import IMPUTED_SUFFIX, QUALITY_SUFFIX, Normalizer
from src.data_ingestion.quality_checks import quality_report, validate_frame
from src.data_ingestion.registry import ADAPTER_REGISTRY, available_sources, get_adapter

WEEKS = ("2024-W01", "2024-W12")


@pytest.mark.parametrize("source", sorted(ADAPTER_REGISTRY))
def test_every_adapter_produces_the_tidy_contract(source, small_region):
    """Rule #6: whatever the wire format, the output schema is identical."""
    result = get_adapter(source, region=small_region).run(*WEEKS)
    assert list(result.frame.columns) == TIDY_COLUMNS
    assert not result.is_empty
    assert set(result.frame["district"]) <= set(small_region.district_names)
    assert result.frame["quality"].between(0, 1).all()


@pytest.mark.parametrize("source", sorted(ADAPTER_REGISTRY))
def test_unconfigured_adapters_degrade_instead_of_failing(source, small_region):
    """Rule #1: a missing credential must never crash the pipeline."""
    adapter = get_adapter(source, region=small_region)
    result = adapter.run(*WEEKS)
    assert result.mode in (FetchMode.SYNTHETIC, FetchMode.CACHE, FetchMode.LIVE)
    if result.mode is FetchMode.SYNTHETIC:
        assert any(f.code == "synthetic_fallback" for f in result.flags)
        # Synthetic data may never claim full confidence.
        assert result.mean_quality < 1.0


@pytest.mark.parametrize("source", sorted(ADAPTER_REGISTRY))
def test_synthetic_output_is_deterministic(source, small_region):
    a = get_adapter(source, region=small_region).synthesize(["2024-W05", "2024-W06"])
    b = get_adapter(source, region=small_region).synthesize(["2024-W05", "2024-W06"])
    pd.testing.assert_frame_equal(a, b)


def test_only_search_trends_is_optional():
    """Rule #14: search/social is never a primary predictor."""
    optional = [n for n, cls in ADAPTER_REGISTRY.items() if cls.optional]
    assert optional == ["search_trends"]


def test_rainfall_has_a_wet_and_a_dry_season(small_region):
    frame = get_adapter("chirps", region=small_region).synthesize(
        [f"2024-W{w:02d}" for w in range(1, 53)]
    )
    weekly = frame.groupby("week")["value"].mean()
    assert weekly.max() > 2.5 * weekly.min()


def test_dhis2_leaves_unreported_weeks_missing_not_zero(small_region):
    """Rule #7: never zero-fill surveillance."""
    result = get_adapter("dhis2", region=small_region).run("2023-W01", "2024-W52")
    cases = result.frame[result.frame["variable"] == "cases_malaria"]
    assert cases["value"].isna().any()
    assert (cases.loc[cases["value"].isna(), "quality"] == 0).all()


def test_gravity_fallback_supplies_a_travel_matrix(small_region):
    """Rule #8: spatial prediction works without a telco agreement."""
    matrix = get_adapter("cdr_mobility", region=small_region).travel_matrix()
    assert list(matrix.index) == small_region.district_names
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_quality_checks_flag_without_dropping():
    frame = pd.DataFrame(
        {
            "district": ["A"] * 6,
            "week": [f"2024-W{w:02d}" for w in range(1, 7)],
            "variable": ["temperature_c"] * 6,
            "value": [25.0, 26.0, np.nan, 25.5, 900.0, 26.2],
            "quality": [1.0] * 6,
            "source": ["era5"] * 6,
        }
    )
    validated, flags = validate_frame(frame, source="era5")
    codes = {f.code for f in flags}
    assert {"missing_values", "out_of_range"} <= codes
    assert len(validated) == 6                      # nothing dropped
    assert validated.loc[2, "quality"] == 0.0       # the NaN
    assert validated.loc[4, "quality"] < 1.0        # the 900 C reading


def test_quality_report_summarises_sources():
    frames = {
        "era5": pd.DataFrame(
            {"district": ["A"], "week": ["2024-W01"], "variable": ["temperature_c"],
             "value": [25.0], "quality": [0.9], "source": ["era5"]}
        )
    }
    report = quality_report(frames)
    assert report.loc[0, "completeness"] == 1.0
    assert report.loc[0, "latest_week"] == "2024-W01"


def test_panel_fuses_sources_onto_one_grid(panel, small_region):
    assert panel.source_count() >= 3
    assert set(panel.districts) == set(small_region.district_names)
    assert "rainfall_mm" in panel.value_columns
    assert "cases_malaria" in panel.value_columns
    assert f"rainfall_mm{QUALITY_SUFFIX}" in panel.frame.columns


def test_panel_records_what_it_imputed(panel):
    imputed = [c for c in panel.frame.columns if c.endswith(IMPUTED_SUFFIX)]
    assert imputed
    # Drivers get filled...
    assert panel.frame["rainfall_mm"].notna().all()
    # ...but case counts are never fabricated.
    assert panel.frame["cases_malaria"].isna().any()


def test_panel_flags_thin_fusion(small_region):
    from src.data_ingestion.normalizer import ingest

    thin = ingest(["chirps"], "2024-W01", "2024-W06", region=small_region)
    assert any(f.code == "insufficient_fusion" for f in thin.flags)


def test_normalizer_drops_districts_outside_the_grid(small_region):
    adapter = get_adapter("chirps", region=small_region)
    result = adapter.run("2024-W01", "2024-W03")
    rogue = result.frame.iloc[[0]].copy()
    rogue["district"] = "Atlantis"
    result.frame = pd.concat([result.frame, rogue], ignore_index=True)
    normalized = adapter.normalize(result)
    assert "Atlantis" not in set(normalized.frame["district"])
    assert any(f.code == "unknown_districts" for f in normalized.flags)


def test_available_sources_lists_every_registered_adapter():
    assert set(available_sources()) == set(ADAPTER_REGISTRY)
    assert "search_trends" not in available_sources(include_optional=False)
