"""Cross-variable interactions and mechanism-shaped transforms.

Two kinds of feature live here:

* **Interactions** — transmission is multiplicative, not additive. Rain without
  warmth breeds nothing; warmth without rain has no breeding sites. `rain x temp`
  encodes that directly instead of asking the model to discover it from scratch.
* **Response-shape transforms** — the disease YAML declares *how* each proxy
  relates to transmission (`bell_curve`, `positive_with_saturation`,
  `threshold`, ...). Applying that shape turns a raw driver into a feature the
  model can use linearly, and keeps the SHAP explanation interpretable in the
  mechanism's own terms.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.types import FeatureSpec, Relationship


def apply_response_shape(values: pd.Series, spec: FeatureSpec) -> pd.Series:
    """Transform a raw driver according to its declared dose-response shape."""
    params = spec.params()
    relationship = spec.relationship

    if relationship == Relationship.POSITIVE_WITH_SATURATION:
        threshold = _first_param(params, ("saturation_threshold_mm", "saturation_threshold"), None)
        if threshold is None:
            threshold = float(values.quantile(0.9)) or 1.0
        threshold = float(threshold)
        # Rises to the threshold, then declines: breeding sites wash out.
        below = values.clip(upper=threshold) / threshold
        excess = (values - threshold).clip(lower=0) / max(threshold, 1e-9)
        return (below - 0.4 * np.tanh(excess)).rename(f"{spec.name}_shaped")

    if relationship == Relationship.BELL_CURVE:
        optimal = _first_param(
            params, ("optimal_range_celsius", "optimal_range_percent", "optimal_range"), None
        )
        if optimal is None:
            centre = float(values.median())
            width = float(values.std()) or 1.0
        else:
            low, high = float(optimal[0]), float(optimal[1])
            centre = (low + high) / 2.0
            width = max((high - low) / 2.0, 1e-6) * 1.5
        return np.exp(-((values - centre) ** 2) / (2 * width**2)).rename(f"{spec.name}_shaped")

    if relationship == Relationship.THRESHOLD:
        threshold = _first_param(params, ("threshold_mm", "threshold"), None)
        if threshold is None:
            threshold = float(values.quantile(0.85))
        threshold = float(threshold)
        # Smooth step so gradient-boosted trees and linear models both cope.
        return pd.Series(
            1.0 / (1.0 + np.exp(-(values - threshold) / max(threshold * 0.1, 1e-6))),
            index=values.index,
            name=f"{spec.name}_shaped",
        )

    if relationship == Relationship.NEGATIVE_LINEAR:
        return (-values).rename(f"{spec.name}_shaped")

    return values.rename(f"{spec.name}_shaped")


def add_shaped_features(
    frame: pd.DataFrame,
    specs: Sequence[FeatureSpec],
    column_for_proxy: Dict[str, str],
) -> pd.DataFrame:
    """Add one mechanism-shaped column per proxy that declares a non-linear shape."""
    new: Dict[str, pd.Series] = {}
    for spec in specs:
        column = column_for_proxy.get(spec.name)
        if column is None or column not in frame.columns:
            continue
        if spec.relationship in (Relationship.POSITIVE_LINEAR,):
            continue
        shaped = apply_response_shape(frame[column], spec)
        new[f"{spec.name}_shaped"] = shaped
    return _concat(frame, new)


#: Interactions worth adding when both sides are present. Each carries the
#: mechanism it encodes so the explainer can verbalise it.
INTERACTION_LIBRARY: Tuple[Tuple[str, str, str], ...] = (
    ("rainfall", "temperature", "warm standing water is what actually breeds mosquitoes"),
    ("rainfall", "wash_access", "flooding only contaminates supply where sanitation is weak"),
    ("temperature", "humidity", "vector survival needs warmth and moisture together"),
    ("population_density", "mobility_inbound", "imported cases spread fastest in crowded districts"),
    ("air_quality_pm25", "population_density", "pollution exposure scales with the population under it"),
    ("ndvi", "rainfall", "greenness confirms that rainfall persisted as surface moisture"),
)


def add_interactions(
    frame: pd.DataFrame,
    column_for_proxy: Dict[str, str],
    library: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Add multiplicative interactions; returns the frame and a mechanism map."""
    library = library or INTERACTION_LIBRARY
    new: Dict[str, pd.Series] = {}
    mechanisms: Dict[str, str] = {}
    for left, right, mechanism in library:
        left_col, right_col = column_for_proxy.get(left), column_for_proxy.get(right)
        if not left_col or not right_col:
            continue
        if left_col not in frame.columns or right_col not in frame.columns:
            continue
        name = f"{left}_x_{right}"
        # Standardise before multiplying so one large-scale driver cannot
        # dominate the product purely through its units.
        new[name] = _standardise(frame[left_col]) * _standardise(frame[right_col])
        mechanisms[name] = mechanism
    return _concat(frame, new), mechanisms


def _standardise(series: pd.Series) -> pd.Series:
    std = series.std()
    if not np.isfinite(std) or std == 0:
        return series - series.mean()
    return (series - series.mean()) / std


def _first_param(params: Dict, keys: Sequence[str], default):
    for key in keys:
        if key in params:
            return params[key]
    return default


def _concat(frame: pd.DataFrame, new: Dict[str, pd.Series]) -> pd.DataFrame:
    if not new:
        return frame
    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)
