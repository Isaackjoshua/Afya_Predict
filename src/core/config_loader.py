"""Load and validate YAML disease/region/alert configuration.

Disease modules are *configuration first*: a new disease is a YAML file plus a
module class, never a change to the core pipeline (critical rule #5).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from config.settings import ALERT_RULES_DIR, DISEASE_CONFIG_DIR, REGION_CONFIG_DIR, get_settings
from src.core.types import BoundingBox, DiseaseConfig, RegionConfig

_TEMPLATE_PREFIX = "_"


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def list_disease_configs(config_dir: Optional[Path] = None) -> List[str]:
    """Return the slugs of every non-template disease config on disk."""
    directory = Path(config_dir or DISEASE_CONFIG_DIR)
    if not directory.exists():
        return []
    return sorted(
        p.stem for p in directory.glob("*.yaml") if not p.name.startswith(_TEMPLATE_PREFIX)
    )


def load_disease_config(slug: str, config_dir: Optional[Path] = None) -> DiseaseConfig:
    """Load `config/diseases/<slug>.yaml` into a validated `DiseaseConfig`."""
    directory = Path(config_dir or DISEASE_CONFIG_DIR)
    path = directory / f"{slug}.yaml"
    if not path.exists():
        available = ", ".join(list_disease_configs(directory)) or "none"
        raise FileNotFoundError(f"No disease config {path}. Available: {available}")
    raw = _read_yaml(path)
    body = dict(raw.get("disease", raw))
    # The filename is the disease's identity, not its display name.
    body.setdefault("config_slug", slug)
    return DiseaseConfig.model_validate(body)


def load_all_disease_configs(config_dir: Optional[Path] = None) -> Dict[str, DiseaseConfig]:
    return {slug: load_disease_config(slug, config_dir) for slug in list_disease_configs(config_dir)}


def load_region_config(name: Optional[str] = None, config_dir: Optional[Path] = None) -> RegionConfig:
    """Load `config/regions/<name>.yaml` into a validated `RegionConfig`."""
    settings = get_settings()
    name = name or settings.default_region
    directory = Path(config_dir or REGION_CONFIG_DIR)
    path = directory / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No region config {path}")
    raw = _read_yaml(path)
    region = dict(raw.get("region", {}))
    boundaries = raw.get("boundaries", {}) or {}
    climate = raw.get("climate", {}) or {}
    defaults = raw.get("defaults", {}) or {}

    region["admin_level"] = boundaries.get("admin_level", "district")
    geojson_path = boundaries.get("geojson_path")
    region["geojson_path"] = geojson_path or None
    if climate.get("bbox"):
        region["bbox"] = BoundingBox(**climate["bbox"])
    region["reporting_completeness"] = defaults.get("reporting_completeness", 0.9)
    region["weather_stations"] = defaults.get("weather_stations", 0)
    region["districts"] = raw.get("districts", []) or []
    return RegionConfig.model_validate(region)


def load_alert_rules(name: str = "default", config_dir: Optional[Path] = None) -> dict:
    """Load cross-cutting alert behaviour from `config/alert_rules/`."""
    directory = Path(config_dir or ALERT_RULES_DIR)
    path = directory / f"{name}.yaml"
    if not path.exists():
        return {}
    return _read_yaml(path)


@functools.lru_cache(maxsize=8)
def cached_region_config(name: Optional[str] = None) -> RegionConfig:
    """Memoised region loader — the district table is read on every request."""
    return load_region_config(name)


@functools.lru_cache(maxsize=64)
def cached_disease_config(slug: str) -> DiseaseConfig:
    return load_disease_config(slug)


def validate_disease_config(config: DiseaseConfig) -> List[str]:
    """Return human-readable problems with a disease config (empty == valid).

    Enforces critical rule #1: at least three fused data sources.
    """
    problems: List[str] = []
    if not config.name:
        problems.append("disease.name is empty")
    if not config.code:
        problems.append("disease.code is empty")
    if len(config.required_sources) < 3:
        problems.append(
            f"only {len(config.required_sources)} non-optional data source(s) configured; "
            "rule #1 requires fusing at least 3"
        )
    for proxy in config.digital_proxies:
        low, high = proxy.lag_weeks_range
        if low > high:
            problems.append(f"proxy {proxy.name!r}: lag_weeks_range is inverted {proxy.lag_weeks_range}")
        if not (low <= proxy.optimal_lag_weeks <= high):
            problems.append(
                f"proxy {proxy.name!r}: optimal_lag_weeks={proxy.optimal_lag_weeks} "
                f"outside lag_weeks_range={proxy.lag_weeks_range}"
            )
        if not proxy.mechanism:
            problems.append(f"proxy {proxy.name!r}: mechanism is empty (causal reasoning is required)")
    thresholds = [config.alerts.low, config.alerts.medium, config.alerts.high, config.alerts.critical]
    if thresholds != sorted(thresholds):
        problems.append("alert thresholds are not monotonically increasing low<medium<high<critical")
    elif len(set(thresholds)) < len(thresholds):
        # All-equal thresholds (the scaffold default of 0.0) would classify every
        # forecast as critical, so this has to fail rather than pass quietly.
        problems.append(
            f"alert thresholds are not distinct ({thresholds}); every forecast would land in "
            "the same risk band. Derive them from the district-level incidence distribution."
        )
    elif config.alerts.critical <= 0:
        problems.append("alerts.critical must be greater than 0 cases per 1,000 per week")
    for level in ("medium", "high", "critical"):
        if not config.recommendations.get(level):
            problems.append(f"no response recommendations configured for level {level!r}")
    return problems
