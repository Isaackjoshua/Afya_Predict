"""Shared FastAPI dependencies.

Modules and the region grid are loaded once per process and reused, which is
what keeps prediction reads inside the 2-second budget: nothing is refitted or
re-parsed on the request path.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional

from fastapi import HTTPException, status

from config.settings import Settings, get_settings
from src.core.config_loader import cached_region_config
from src.core.logging import get_logger
from src.core.types import RegionConfig

log = get_logger("api.deps")


@functools.lru_cache(maxsize=1)
def get_region() -> RegionConfig:
    return cached_region_config()


@functools.lru_cache(maxsize=1)
def get_cache():
    from offline.local_cache import LocalCache

    return LocalCache()


@functools.lru_cache(maxsize=16)
def get_module(slug: str):
    """Build a disease module and load its persisted weights if present."""
    from src.models.registry import build_module, list_modules

    if slug not in list_modules():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown disease {slug!r}. Available: {', '.join(list_modules())}",
        )
    module = build_module(slug, region=get_region())
    module.load()
    return module


def validate_district(district: str) -> str:
    region = get_region()
    if district not in set(region.district_names):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown district {district!r} in region {region.name}",
        )
    return district


def get_scheduler():
    from src.data_ingestion.scheduler import IngestionScheduler

    return IngestionScheduler(region=get_region())


def reset_caches() -> None:
    """Clear the memoised objects — used after a retrain or config change."""
    get_module.cache_clear()
    get_region.cache_clear()
    get_cache.cache_clear()
    cached_region_config.cache_clear()
