"""Concrete data source adapters, one module per upstream provider."""

from src.data_ingestion.adapters.cdr_mobility import CDRMobilityAdapter  # noqa: F401
from src.data_ingestion.adapters.chirps_rainfall import CHIRPSRainfallAdapter  # noqa: F401
from src.data_ingestion.adapters.dhis2_surveillance import DHIS2SurveillanceAdapter  # noqa: F401
from src.data_ingestion.adapters.era5_climate import ERA5ClimateAdapter  # noqa: F401
from src.data_ingestion.adapters.livestock_disease import LivestockDiseaseAdapter  # noqa: F401
from src.data_ingestion.adapters.modis_ndvi import MODISNDVIAdapter  # noqa: F401
from src.data_ingestion.adapters.population_density import PopulationDensityAdapter  # noqa: F401
from src.data_ingestion.adapters.search_trends import SearchTrendsAdapter  # noqa: F401
from src.data_ingestion.adapters.sentinel5p_airquality import (  # noqa: F401
    Sentinel5PAirQualityAdapter,
)
from src.data_ingestion.adapters.wash_indicators import WASHIndicatorsAdapter  # noqa: F401

__all__ = [
    "CDRMobilityAdapter",
    "CHIRPSRainfallAdapter",
    "DHIS2SurveillanceAdapter",
    "ERA5ClimateAdapter",
    "LivestockDiseaseAdapter",
    "MODISNDVIAdapter",
    "PopulationDensityAdapter",
    "SearchTrendsAdapter",
    "Sentinel5PAirQualityAdapter",
    "WASHIndicatorsAdapter",
]
