"""Adapter registry — maps the `source:` key in disease YAMLs to a class."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from config.settings import Settings
from src.data_ingestion.adapters import (
    CDRMobilityAdapter,
    CHIRPSRainfallAdapter,
    DHIS2SurveillanceAdapter,
    ERA5ClimateAdapter,
    LivestockDiseaseAdapter,
    MODISNDVIAdapter,
    PopulationDensityAdapter,
    SearchTrendsAdapter,
    Sentinel5PAirQualityAdapter,
    WASHIndicatorsAdapter,
)
from src.data_ingestion.base_adapter import BaseAdapter
from src.core.types import RegionConfig

ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    CHIRPSRainfallAdapter.source_name: CHIRPSRainfallAdapter,
    ERA5ClimateAdapter.source_name: ERA5ClimateAdapter,
    MODISNDVIAdapter.source_name: MODISNDVIAdapter,
    Sentinel5PAirQualityAdapter.source_name: Sentinel5PAirQualityAdapter,
    DHIS2SurveillanceAdapter.source_name: DHIS2SurveillanceAdapter,
    CDRMobilityAdapter.source_name: CDRMobilityAdapter,
    PopulationDensityAdapter.source_name: PopulationDensityAdapter,
    WASHIndicatorsAdapter.source_name: WASHIndicatorsAdapter,
    LivestockDiseaseAdapter.source_name: LivestockDiseaseAdapter,
    SearchTrendsAdapter.source_name: SearchTrendsAdapter,
}

#: Sources that may never be a model's primary signal (rule #14).
OPTIONAL_SOURCES = {name for name, cls in ADAPTER_REGISTRY.items() if cls.optional}


def available_sources(include_optional: bool = True) -> List[str]:
    return sorted(
        name for name in ADAPTER_REGISTRY if include_optional or name not in OPTIONAL_SOURCES
    )


def get_adapter(
    source: str,
    region: Optional[RegionConfig] = None,
    settings: Optional[Settings] = None,
    **kwargs,
) -> BaseAdapter:
    """Instantiate the adapter registered under `source`."""
    try:
        cls = ADAPTER_REGISTRY[source]
    except KeyError as exc:
        raise KeyError(
            f"Unknown data source {source!r}. Known sources: {', '.join(available_sources())}"
        ) from exc
    return cls(region=region, settings=settings, **kwargs)


def register_adapter(cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
    """Decorator so downstream deployments can plug in their own sources."""
    ADAPTER_REGISTRY[cls.source_name] = cls
    if cls.optional:
        OPTIONAL_SOURCES.add(cls.source_name)
    return cls


def describe_sources(
    region: Optional[RegionConfig] = None, settings: Optional[Settings] = None
) -> List[dict]:
    """Capability report for every registered source (drives `/data/status`)."""
    out = []
    for name in available_sources():
        try:
            out.append(get_adapter(name, region=region, settings=settings).describe())
        except Exception as exc:  # noqa: BLE001
            out.append({"source": name, "error": str(exc), "configured": False})
    return out
