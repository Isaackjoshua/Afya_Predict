"""Global and grouped feature importance.

Two views matter to different audiences:

* an epidemiologist wants **proxy-level** importance ("how much does rainfall
  matter for cholera in Mwanza?"), not 117 individual column scores;
* a modeller wants the raw per-feature ranking to debug the fit.

Both are produced here, aggregated through the provenance map that
`FeatureBuilder` recorded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.explainability.shap_explainer import ExplanationResult


def global_importance(explanation: ExplanationResult, top_n: Optional[int] = None) -> pd.DataFrame:
    """Mean |SHAP| per feature with its mechanism attached."""
    importance = explanation.global_importance()
    total = float(importance.sum()) or 1.0
    rows = []
    for feature, value in importance.items():
        record = explanation.provenance.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "importance": float(value),
                "share": round(float(value) / total, 4),
                "proxy": record.get("proxy", feature),
                "kind": record.get("kind", "unknown"),
                "lag_weeks": record.get("lag_weeks", 0),
                "source": record.get("source", "unknown"),
                "mechanism": record.get("mechanism", ""),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.head(top_n) if top_n else frame


def proxy_importance(explanation: ExplanationResult) -> pd.DataFrame:
    """Aggregate feature importance up to the digital proxy that generated it."""
    frame = global_importance(explanation)
    if frame.empty:
        return frame
    grouped = (
        frame.groupby("proxy")
        .agg(
            importance=("importance", "sum"),
            share=("share", "sum"),
            features=("feature", "count"),
            source=("source", "first"),
            mechanism=("mechanism", "first"),
        )
        .sort_values("importance", ascending=False)
        .reset_index()
    )
    grouped["share"] = grouped["share"].round(4)
    return grouped


def source_importance(explanation: ExplanationResult) -> pd.DataFrame:
    """How much each *data source* contributed — the evidence for fusion.

    If one source dominates every disease, the platform has quietly recreated
    the Google Flu Trends single-source failure mode, and this table shows it.
    """
    frame = global_importance(explanation)
    if frame.empty:
        return frame
    grouped = (
        frame.groupby("source")
        .agg(importance=("importance", "sum"), share=("share", "sum"), features=("feature", "count"))
        .sort_values("importance", ascending=False)
        .reset_index()
    )
    grouped["share"] = grouped["share"].round(4)
    return grouped


def kind_importance(explanation: ExplanationResult) -> pd.DataFrame:
    """Importance grouped by feature family (lagged proxy, seasonality, ...)."""
    frame = global_importance(explanation)
    if frame.empty:
        return frame
    return (
        frame.groupby("kind")
        .agg(importance=("importance", "sum"), share=("share", "sum"))
        .sort_values("importance", ascending=False)
        .reset_index()
    )


def concentration_index(explanation: ExplanationResult) -> float:
    """Herfindahl index of source shares: 1.0 means a single-source model.

    Used as a guard rail — a model whose predictions rest on one source is
    exactly what shortcoming #2 warns about.
    """
    shares = source_importance(explanation)["share"]
    if shares.empty:
        return 1.0
    normalised = shares / shares.sum()
    return float((normalised**2).sum())


def dominant_source_warning(explanation: ExplanationResult, threshold: float = 0.7) -> Optional[str]:
    """Warn when one source carries more than `threshold` of total attribution."""
    frame = source_importance(explanation)
    if frame.empty:
        return None
    top = frame.iloc[0]
    total = float(frame["share"].sum()) or 1.0
    share = float(top["share"]) / total
    if share >= threshold:
        return (
            f"{share:.0%} of this model's attribution comes from a single source "
            f"({top['source']}). Single-source dependence is what broke Google Flu Trends; "
            "verify the other feeds are arriving."
        )
    return None
