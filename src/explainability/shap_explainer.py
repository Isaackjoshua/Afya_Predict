"""SHAP attribution for every prediction (shortcoming #4, critical rule #2).

Backend order: the `shap` package when installed, exact TreeSHAP for the
bundled ensemble, then sampled permutation Shapley values for anything else.
All three satisfy local accuracy, so a `DriverExplanation`'s
`contribution_share` is a real share of the model's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.core.types import DriverExplanation

log = get_logger("explain.shap")

TOP_DRIVERS = 5


@dataclass
class ExplanationResult:
    """SHAP output for one or more rows, with provenance already joined on."""

    values: pd.DataFrame          # rows x features
    base_value: float
    method: str
    feature_values: Optional[pd.DataFrame] = None
    provenance: Dict[str, dict] = field(default_factory=dict)

    def for_row(self, position: int = -1) -> pd.Series:
        return self.values.iloc[position]

    def top_drivers(self, position: int = -1, n: int = TOP_DRIVERS) -> List[DriverExplanation]:
        """Rank the features driving one prediction, with mechanism text."""
        row = self.for_row(position)
        total = float(row.abs().sum()) or 1.0
        ranked = row.reindex(row.abs().sort_values(ascending=False).index).head(n)

        values_row = (
            self.feature_values.iloc[position] if self.feature_values is not None else None
        )
        out: List[DriverExplanation] = []
        for feature, shap_value in ranked.items():
            record = self.provenance.get(feature, {})
            out.append(
                DriverExplanation(
                    feature=str(feature),
                    proxy=record.get("proxy", str(feature)),
                    lag_weeks=int(record.get("lag_weeks", 0) or 0),
                    value=None if values_row is None else _safe_float(values_row.get(feature)),
                    shap_value=float(shap_value),
                    contribution_share=round(abs(float(shap_value)) / total, 4),
                    direction=(
                        "increases" if shap_value > 1e-9
                        else "decreases" if shap_value < -1e-9
                        else "neutral"
                    ),
                    mechanism=record.get("mechanism", ""),
                )
            )
        return out

    def as_dict(self, position: int = -1) -> Dict[str, float]:
        return {str(k): round(float(v), 6) for k, v in self.for_row(position).items()}

    def global_importance(self) -> pd.Series:
        """Mean |SHAP| per feature — the model's global view."""
        return self.values.abs().mean().sort_values(ascending=False)


class ShapExplainer:
    """Compute SHAP values for a fitted `TrainedModel`."""

    def __init__(
        self,
        model,
        background: Optional[pd.DataFrame] = None,
        provenance: Optional[Dict[str, dict]] = None,
        max_background: int = 100,
        permutation_samples: int = 32,
    ) -> None:
        self.model = model
        self.provenance = provenance or {}
        self.permutation_samples = permutation_samples
        self.background = self._prepare_background(background, max_background)
        self._shap_explainer = None
        self.method = self._choose_method()

    # -- setup -------------------------------------------------------------
    def _prepare_background(
        self, background: Optional[pd.DataFrame], max_rows: int
    ) -> Optional[pd.DataFrame]:
        if background is None or background.empty:
            return None
        aligned = background.reindex(columns=self.model.feature_names)
        clean = aligned.dropna(how="all")
        if clean.empty:
            return None
        if len(clean) > max_rows:
            step = max(1, len(clean) // max_rows)
            clean = clean.iloc[::step].head(max_rows)
        return clean.fillna(clean.median())

    def _choose_method(self) -> str:
        estimator = self.model.estimator
        try:
            import shap  # noqa: F401

            return "shap_package"
        except ImportError:
            pass
        from src.models.gbm import NumpyGBMRegressor

        if isinstance(estimator, NumpyGBMRegressor):
            return "exact_tree_shap"
        return "permutation_shap"

    # -- explanation -------------------------------------------------------
    def explain(self, X: pd.DataFrame) -> ExplanationResult:
        """SHAP values for the given feature rows."""
        aligned = X.reindex(columns=self.model.feature_names)
        matrix = aligned.to_numpy(dtype=float)

        if self.method == "shap_package":
            values, base = self._explain_with_package(matrix)
        elif self.method == "exact_tree_shap":
            from src.explainability.tree_shap import numpy_gbm_shap

            values, base = numpy_gbm_shap(self.model.estimator, matrix)
        else:
            from src.explainability.tree_shap import permutation_shap

            background = (
                self.background.to_numpy(dtype=float)
                if self.background is not None
                else np.nan_to_num(matrix)
            )
            values, base = permutation_shap(
                lambda rows: np.asarray(self.model.estimator.predict(rows), dtype=float),
                np.nan_to_num(matrix, nan=float(np.nanmedian(background))),
                background,
                n_samples=self.permutation_samples,
            )

        return ExplanationResult(
            values=pd.DataFrame(values, index=X.index, columns=self.model.feature_names),
            base_value=float(base),
            method=self.method,
            feature_values=aligned,
            provenance=self.provenance,
        )

    def _explain_with_package(self, matrix: np.ndarray):
        import shap

        if self._shap_explainer is None:
            estimator = self.model.estimator
            background = (
                self.background.to_numpy(dtype=float)
                if self.background is not None
                else matrix
            )
            try:
                self._shap_explainer = shap.TreeExplainer(estimator)
            except Exception:  # noqa: BLE001 - not a supported tree model
                self._shap_explainer = shap.Explainer(estimator.predict, background)
        explanation = self._shap_explainer(matrix)
        values = np.asarray(explanation.values, dtype=float)
        base = explanation.base_values
        base = float(np.mean(np.asarray(base, dtype=float)))
        return values, base


def explain_predictions(
    model,
    X: pd.DataFrame,
    background: Optional[pd.DataFrame] = None,
    provenance: Optional[Dict[str, dict]] = None,
) -> ExplanationResult:
    """One-shot helper used by the disease modules."""
    return ShapExplainer(model, background=background, provenance=provenance).explain(X)


def _safe_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
