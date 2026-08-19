"""Explanation endpoints (shortcoming #4, critical rule #2).

Every prediction the API serves already carries its explanation inline. These
endpoints exist for the dashboard's drill-down: the SHAP waterfall, the
source-contribution breakdown, and the counterfactual panel.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from src.api.dependencies import get_cache, get_module
from src.api.schemas import ExplanationResponse
from src.core.logging import get_logger

router = APIRouter(prefix="/explain", tags=["explainability"])
log = get_logger("api.explain")


@router.get("/{prediction_id}", response_model=ExplanationResponse)
def explain(prediction_id: str) -> ExplanationResponse:
    """Full explanation for one stored prediction."""
    prediction = get_cache().get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown prediction_id")

    # Roll the per-feature SHAP values up to their data sources, so a reader can
    # see the fusion working rather than a wall of engineered column names.
    contributions: dict = {}
    try:
        module = get_module(prediction.disease.lower().replace(" ", "_"))
        provenance = module.feature_matrix.provenance if module.feature_matrix else {}
    except Exception:  # noqa: BLE001 - explanation must not depend on a live model
        provenance = {}

    for feature, value in prediction.shap_values.items():
        source = provenance.get(feature, {}).get("source", "unknown")
        bucket = contributions.setdefault(source, {"source": source, "abs_contribution": 0.0, "features": 0})
        bucket["abs_contribution"] += abs(float(value))
        bucket["features"] += 1

    total = sum(b["abs_contribution"] for b in contributions.values()) or 1.0
    source_contributions = sorted(
        (
            {**bucket, "share": round(bucket["abs_contribution"] / total, 4)}
            for bucket in contributions.values()
        ),
        key=lambda b: b["share"],
        reverse=True,
    )

    return ExplanationResponse(
        prediction_id=prediction.prediction_id,
        disease=prediction.disease,
        district=prediction.district,
        target_week=prediction.target_week,
        risk_level=prediction.risk_level,
        natural_language_explanation=prediction.natural_language_explanation,
        counterfactual=prediction.counterfactual,
        top_drivers=prediction.top_drivers,
        shap_values=prediction.shap_values,
        source_contributions=source_contributions,
        data_quality_flags=prediction.data_quality_flags,
    )


@router.get("/{prediction_id}/waterfall", response_model=dict)
def waterfall(prediction_id: str, top_n: int = Query(10, ge=3, le=30)) -> dict:
    """Ordered contributions for the SHAP waterfall chart."""
    prediction = get_cache().get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown prediction_id")

    ranked = sorted(prediction.shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = ranked[:top_n]
    remainder = sum(value for _, value in ranked[top_n:])
    labels = {d.feature: d for d in prediction.top_drivers}

    steps = [
        {
            "feature": feature,
            "contribution": round(float(value), 5),
            "direction": "increases" if value > 0 else "decreases",
            "proxy": labels[feature].proxy if feature in labels else feature,
            "lag_weeks": labels[feature].lag_weeks if feature in labels else None,
            "mechanism": labels[feature].mechanism if feature in labels else "",
        }
        for feature, value in top
    ]
    if abs(remainder) > 1e-9:
        steps.append(
            {
                "feature": f"{len(ranked) - top_n} other features",
                "contribution": round(float(remainder), 5),
                "direction": "increases" if remainder > 0 else "decreases",
                "proxy": "other", "lag_weeks": None, "mechanism": "",
            }
        )
    return {
        "prediction_id": prediction_id,
        "predicted_cases": prediction.predicted_cases,
        "risk_level": prediction.risk_level,
        "steps": steps,
        "explanation": prediction.natural_language_explanation,
        "counterfactual": prediction.counterfactual,
    }
