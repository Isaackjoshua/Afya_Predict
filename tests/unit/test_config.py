"""Configuration loading and validation."""

from __future__ import annotations

import pytest

from src.core.config_loader import (
    list_disease_configs,
    load_alert_rules,
    validate_disease_config,
)
from src.core.types import DiseaseConfig


def test_five_disease_modules_registered():
    """Acceptance criterion #1: at least five diseases ship configured."""
    slugs = list_disease_configs()
    assert {"malaria", "cholera", "tuberculosis", "respiratory", "hiv"} <= set(slugs)


def test_template_is_not_loaded_as_a_disease():
    assert "_template" not in list_disease_configs()


@pytest.mark.parametrize(
    "slug", ["malaria", "cholera", "tuberculosis", "respiratory", "hiv"]
)
def test_shipped_configs_validate(all_disease_configs, slug):
    assert validate_disease_config(all_disease_configs[slug]) == []


@pytest.mark.parametrize(
    "slug", ["malaria", "cholera", "tuberculosis", "respiratory", "hiv"]
)
def test_every_disease_fuses_at_least_three_sources(all_disease_configs, slug):
    """Acceptance criterion #2 / critical rule #1."""
    assert len(all_disease_configs[slug].required_sources) >= 3


def test_lag_candidates_cover_the_declared_range(malaria_config):
    rainfall = next(p for p in malaria_config.digital_proxies if p.name == "rainfall")
    assert rainfall.lag_candidates == list(range(2, 13))
    assert rainfall.params()["saturation_threshold_mm"] == 150


def test_validation_rejects_a_thin_config():
    bad = DiseaseConfig(
        name="Test",
        code="TST",
        digital_proxies=[
            {"name": "rainfall", "source": "chirps", "lag_weeks_range": (0, 2),
             "optimal_lag_weeks": 1, "mechanism": "x"}
        ],
    )
    problems = validate_disease_config(bad)
    assert any("at least 3" in p for p in problems)
    assert any("recommendations" in p for p in problems)


def test_validation_catches_an_out_of_range_optimal_lag():
    bad = DiseaseConfig(
        name="Test",
        code="TST",
        digital_proxies=[
            {"name": "rain", "source": "chirps", "lag_weeks_range": (2, 6),
             "optimal_lag_weeks": 12, "mechanism": "x"}
        ],
    )
    assert any("outside lag_weeks_range" in p for p in validate_disease_config(bad))


def test_region_grid_is_populated(region):
    assert len(region.districts) > 50
    assert region.population_of("Kinondoni") > 0
    assert region.bbox is not None


def test_alert_rules_load():
    rules = load_alert_rules()
    assert rules["risk_levels"] == ["low", "medium", "high", "critical"]
    assert rules["delivery"]["by_level"]["critical"]

def test_slug_comes_from_the_filename_not_the_display_name(all_disease_configs):
    """A disease's identity is its config filename.

    "Acute Respiratory Infection" lives in `respiratory.yaml`, and its
    surveillance column is `cases_respiratory`. Deriving the slug from the
    display name produced `cases_acute_respiratory_infection`, which exists
    nowhere — the feature builder raised KeyError and the whole module was
    unusable. It went unnoticed because every other shipped disease happens to
    have a display name matching its filename.
    """
    from src.data_ingestion.adapters.dhis2_surveillance import CASE_VARIABLES

    respiratory = all_disease_configs["respiratory"]
    assert respiratory.name == "Acute Respiratory Infection"
    assert respiratory.slug == "respiratory"

    for slug, config in all_disease_configs.items():
        assert config.slug == slug, f"{slug}: slug is {config.slug}"
        assert config.slug in CASE_VARIABLES, (
            f"{slug} has no surveillance column; add it to CASE_VARIABLES in "
            "src/data_ingestion/adapters/dhis2_surveillance.py"
        )


def test_an_in_memory_config_still_gets_a_slug():
    """Scaffolding and tests build configs without a file behind them."""
    from src.core.types import DiseaseConfig

    config = DiseaseConfig(name="Rift Valley Fever", code="RVF")
    assert config.slug == "rift_valley_fever"


def test_every_disease_target_column_resolves(all_disease_configs, small_region):
    """The failure mode this pins broke an entire disease module silently."""
    from src.feature_engineering.builder import FeatureBuilder

    for slug, config in all_disease_configs.items():
        builder = FeatureBuilder(config, small_region)
        assert builder.target_column == f"cases_{slug}", slug
