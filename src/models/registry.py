"""Disease module registry (shortcoming #11, critical rule #5).

Registering a disease is the only wiring step. Everything downstream — the API,
the dashboard, the scheduler, the retraining loop — iterates this registry, so
a new disease appears everywhere at once.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from src.core.config_loader import (
    cached_region_config,
    list_disease_configs,
    load_disease_config,
    validate_disease_config,
)
from src.core.logging import get_logger
from src.core.types import DiseaseConfig, RegionConfig
from src.models.base_model import BaseDiseaseModule
from src.models.disease_modules import (
    CholeraModule,
    HIVModule,
    MalariaModule,
    RespiratoryModule,
    TuberculosisModule,
)
from src.models.standard_module import StandardDiseaseModule

log = get_logger("models.registry")

DISEASE_MODULE_REGISTRY: Dict[str, Type[BaseDiseaseModule]] = {
    MalariaModule.slug: MalariaModule,
    CholeraModule.slug: CholeraModule,
    TuberculosisModule.slug: TuberculosisModule,
    RespiratoryModule.slug: RespiratoryModule,
    HIVModule.slug: HIVModule,
}


def register_module(cls: Type[BaseDiseaseModule]) -> Type[BaseDiseaseModule]:
    """Register a disease module class. Usable as a decorator."""
    if not cls.slug:
        raise ValueError(f"{cls.__name__} must define a non-empty `slug`")
    DISEASE_MODULE_REGISTRY[cls.slug] = cls
    log.info("registered disease module %s", cls.slug)
    return cls


def list_modules() -> List[str]:
    """Every disease with both a config and a module class."""
    configured = set(list_disease_configs())
    return sorted(configured & set(DISEASE_MODULE_REGISTRY)) or sorted(configured)


def build_module(
    slug: str,
    region: Optional[RegionConfig] = None,
    config: Optional[DiseaseConfig] = None,
    settings=None,
) -> BaseDiseaseModule:
    """Instantiate the module for `slug`.

    A disease with a config but no bespoke class still works: it falls back to
    `StandardDiseaseModule`, which is the whole point of the plugin design.
    """
    config = config or load_disease_config(slug)
    region = region or cached_region_config()
    cls = DISEASE_MODULE_REGISTRY.get(slug)
    if cls is None:
        log.info(
            "no bespoke module class for %r; using StandardDiseaseModule from its YAML config", slug
        )
        cls = StandardDiseaseModule
    module = cls(config=config, region=region, settings=settings)
    module.slug = slug
    return module


def build_all(
    region: Optional[RegionConfig] = None, settings=None
) -> Dict[str, BaseDiseaseModule]:
    return {slug: build_module(slug, region=region, settings=settings) for slug in list_modules()}


def validate_registry() -> Dict[str, List[str]]:
    """Config + interface check for every registered disease.

    Run by `scripts/add_new_disease.py --validate` and by the test suite, so a
    half-wired disease fails loudly instead of silently producing nothing.
    """
    required = [
        "get_feature_config", "build_feature_matrix", "train", "predict",
        "detect_outbreak", "get_spatial_risk", "generate_recommendations",
    ]
    problems: Dict[str, List[str]] = {}
    for slug in list_disease_configs():
        issues = validate_disease_config(load_disease_config(slug))
        cls = DISEASE_MODULE_REGISTRY.get(slug)
        if cls is None:
            issues.append(
                f"no module class registered for {slug!r}; "
                "it will fall back to StandardDiseaseModule"
            )
        else:
            for method in required:
                if getattr(cls, method, None) is None:
                    issues.append(f"module class does not implement {method}()")
                elif getattr(getattr(cls, method), "__isabstractmethod__", False):
                    issues.append(f"module class leaves {method}() abstract")
        if issues:
            problems[slug] = issues
    return problems


def describe_all(region: Optional[RegionConfig] = None) -> List[dict]:
    """Capability listing for `GET /diseases`."""
    out = []
    for slug in list_modules():
        try:
            out.append(build_module(slug, region=region).describe())
        except Exception as exc:  # noqa: BLE001
            out.append({"slug": slug, "error": str(exc)})
    return out
