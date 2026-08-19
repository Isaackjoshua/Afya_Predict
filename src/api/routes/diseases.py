"""Disease registry endpoints (shortcoming #11)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.dependencies import get_module, get_region
from src.api.schemas import DiseaseListResponse, DiseaseSummary
from src.core.logging import get_logger

router = APIRouter(prefix="/diseases", tags=["diseases"])
log = get_logger("api.diseases")


@router.get("", response_model=DiseaseListResponse)
def list_diseases() -> DiseaseListResponse:
    """Every registered disease module and its current state."""
    from src.models.registry import describe_all

    entries = [DiseaseSummary(**d) for d in describe_all(region=get_region()) if "error" not in d]
    return DiseaseListResponse(count=len(entries), diseases=entries)


@router.get("/{slug}", response_model=DiseaseSummary)
def get_disease(slug: str) -> DiseaseSummary:
    return DiseaseSummary(**get_module(slug).describe())


@router.get("/{slug}/config", response_model=dict)
def disease_config(slug: str) -> dict:
    """The full YAML-derived configuration, including every proxy's mechanism.

    Published deliberately: an agency should be able to read exactly which
    proxies drive a disease, at what lags, and why, without reading the code.
    """
    from src.core.config_loader import load_disease_config, validate_disease_config

    config = load_disease_config(slug)
    return {
        "config": config.model_dump(mode="json"),
        "validation_problems": validate_disease_config(config),
        "proxy_mechanisms": [
            {
                "proxy": p.name,
                "source": p.source,
                "relationship": p.relationship.value,
                "lag_search_range_weeks": list(p.lag_weeks_range),
                "prior_lag_weeks": p.optimal_lag_weeks,
                "mechanism": p.mechanism,
                "optional": p.optional,
            }
            for p in config.digital_proxies
        ],
    }


@router.get("/{slug}/lags", response_model=dict)
def fitted_lags(slug: str) -> dict:
    """The lags actually fitted per district (critical rule #3).

    This is the evidence that coefficients were learned locally rather than
    transplanted; a wide spread across districts is the expected result.
    """
    from src.feature_engineering.lag_features import lag_dispersion, lag_fit_report

    module = get_module(slug)
    if module.feature_matrix is None or not module.feature_matrix.lag_fits:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"No fitted lags in memory for {slug}. Run POST /predictions/run or the "
                "training script first."
            ),
        )
    fits = module.feature_matrix.lag_fits
    return {
        "disease": slug,
        "per_district": lag_fit_report(fits).to_dict(orient="records"),
        "dispersion": lag_dispersion(fits).to_dict(orient="records"),
        "note": (
            "Lags are fitted per district from local history. Where districts disagree, a "
            "single transplanted coefficient would have been wrong for most of them."
        ),
    }
