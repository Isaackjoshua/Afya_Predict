"""Multi-source data ingestion (shortcomings #2, #6, #7).

Every source is an adapter implementing :class:`~src.data_ingestion.base_adapter.BaseAdapter`.
Adapters emit *tidy* frames — `district, week, variable, value, quality, source`
— so the normaliser can fuse them without knowing anything about GeoTIFFs,
GRIB files or DHIS2 analytics tables.
"""

from src.data_ingestion.base_adapter import (  # noqa: F401
    AdapterResult,
    BaseAdapter,
    FetchMode,
    TIDY_COLUMNS,
)
from src.data_ingestion.registry import (  # noqa: F401
    ADAPTER_REGISTRY,
    available_sources,
    get_adapter,
)
