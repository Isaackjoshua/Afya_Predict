"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.core.config_loader import load_all_disease_configs, load_disease_config, load_region_config
from src.core.geo import subset_region

SAMPLE_DISTRICTS = ["Kinondoni", "Ilala", "Mwanza City", "Sengerema", "Dodoma City", "Songea MC"]


@pytest.fixture(scope="session")
def region():
    """The full Tanzania grid."""
    return load_region_config("tanzania")


@pytest.fixture(scope="session")
def small_region(region):
    """A six-district subset — keeps the model tests fast."""
    return subset_region(region, SAMPLE_DISTRICTS)


@pytest.fixture(scope="session")
def malaria_config():
    return load_disease_config("malaria")


@pytest.fixture(scope="session")
def cholera_config():
    return load_disease_config("cholera")


@pytest.fixture(scope="session")
def all_disease_configs():
    return load_all_disease_configs()


@pytest.fixture(scope="session")
def panel(small_region):
    """A two-year fused panel over the small region (synthetic sources)."""
    from src.data_ingestion.normalizer import ingest

    return ingest(
        ["chirps", "era5", "dhis2", "population_density", "cdr_mobility", "wash_indicators", "modis"],
        "2023-W01",
        "2024-W52",
        region=small_region,
    )
