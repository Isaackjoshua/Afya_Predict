"""A compact model for edge deployment (rule #6, shortcoming #9).

The full pipeline needs a fused panel, ~120 engineered features and a boosted
ensemble. A district office running on a solar-charged laptop with a phone
tether does not have that. This module distils a trained disease model into a
handful of coefficients over 8-12 inputs, which:

* fits in a few kilobytes of JSON and syncs over a poor link;
* runs a forecast in microseconds with nothing but NumPy;
* is auditable by hand — an epidemiologist can read the coefficients.

The distilled model is *deliberately* worse than the full one. It reports its
own fidelity against the teacher so an operator knows how much accuracy the
offline mode costs, rather than being quietly downgraded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger

log = get_logger("offline.lightweight")

DEFAULT_N_FEATURES = 10


@dataclass
class LightweightModel:
    """Ridge-distilled surrogate of a full disease model."""

    disease: str
    features: List[str] = field(default_factory=list)
    coefficients: List[float] = field(default_factory=list)
    intercept: float = 0.0
    means: List[float] = field(default_factory=list)
    scales: List[float] = field(default_factory=list)
    fidelity_r2: float = 0.0
    teacher_version: str = ""
    residual_std: float = 0.0
    created_at: str = ""
    feature_notes: Dict[str, str] = field(default_factory=dict)

    # -- distillation ------------------------------------------------------
    @classmethod
    def distil(
        cls,
        teacher,
        X: pd.DataFrame,
        n_features: int = DEFAULT_N_FEATURES,
        provenance: Optional[Dict[str, dict]] = None,
        alpha: float = 1.0,
    ) -> "LightweightModel":
        """Fit a small ridge model to reproduce the teacher's own predictions.

        Trained against the teacher's output rather than the raw target: the
        goal is to *behave like* the deployed model offline, not to be a second
        independent model that disagrees with it.
        """
        aligned = X.reindex(columns=teacher.feature_names)
        teacher_predictions = np.asarray(teacher.predict(aligned), dtype=float)

        chosen = cls._select_features(teacher, aligned, teacher_predictions, n_features)
        subset = aligned[chosen]
        matrix = subset.to_numpy(dtype=float)
        medians = np.nanmedian(np.where(np.isfinite(matrix), matrix, np.nan), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(matrix), matrix, medians)

        means = filled.mean(axis=0)
        scales = np.where(filled.std(axis=0) > 0, filled.std(axis=0), 1.0)
        standardised = (filled - means) / scales

        gram = standardised.T @ standardised + alpha * np.eye(standardised.shape[1])
        centred = teacher_predictions - teacher_predictions.mean()
        coefficients = np.linalg.solve(gram, standardised.T @ centred)
        intercept = float(teacher_predictions.mean())

        fitted = standardised @ coefficients + intercept
        total = float(np.sum((teacher_predictions - teacher_predictions.mean()) ** 2))
        fidelity = float(1 - np.sum((teacher_predictions - fitted) ** 2) / total) if total > 0 else 0.0

        notes = {}
        if provenance:
            for feature in chosen:
                record = provenance.get(feature, {})
                notes[feature] = record.get("mechanism", "")

        model = cls(
            disease=getattr(teacher, "scope", "unknown"),
            features=list(chosen),
            coefficients=[float(c) for c in coefficients],
            intercept=intercept,
            means=[float(m) for m in means],
            scales=[float(s) for s in scales],
            fidelity_r2=round(fidelity, 4),
            teacher_version=getattr(teacher, "version", ""),
            residual_std=float(np.std(teacher_predictions - fitted)),
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
            feature_notes=notes,
        )
        log.info(
            "distilled %s into %d features, fidelity R2 %.3f",
            model.disease, len(chosen), model.fidelity_r2,
        )
        return model

    @staticmethod
    def _select_features(teacher, X: pd.DataFrame, predictions: np.ndarray, k: int) -> List[str]:
        """Pick the k features that best explain the teacher's behaviour."""
        from src.models.backends import feature_importances

        importances = feature_importances(getattr(teacher, "estimator", teacher))
        if importances is not None and len(importances) == len(X.columns):
            order = np.argsort(importances)[::-1]
            return [X.columns[i] for i in order[:k]]
        # Fallback: correlation with the teacher's predictions.
        correlations = {}
        for column in X.columns:
            values = X[column].to_numpy(dtype=float)
            mask = np.isfinite(values)
            if mask.sum() < 10 or np.std(values[mask]) == 0:
                continue
            correlations[column] = abs(float(np.corrcoef(values[mask], predictions[mask])[0, 1]))
        ranked = sorted(correlations, key=lambda c: correlations[c], reverse=True)
        return ranked[:k]

    # -- inference ---------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Forecast from a frame or dict of the distilled features."""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        subset = X.reindex(columns=self.features)
        matrix = subset.to_numpy(dtype=float)
        means = np.array(self.means, dtype=float)
        matrix = np.where(np.isfinite(matrix), matrix, means)
        standardised = (matrix - means) / np.array(self.scales, dtype=float)
        return np.clip(standardised @ np.array(self.coefficients, dtype=float) + self.intercept, 0.0, None)

    def predict_with_interval(self, X, z: float = 1.96):
        point = self.predict(X)
        margin = z * self.residual_std
        return point, np.clip(point - margin, 0.0, None), point + margin

    def explain(self, row) -> List[Dict[str, object]]:
        """Per-feature contributions — the offline node keeps explainability."""
        if isinstance(row, dict):
            row = pd.Series(row)
        values = np.array(
            [float(row.get(f, self.means[i])) for i, f in enumerate(self.features)], dtype=float
        )
        values = np.where(np.isfinite(values), values, np.array(self.means, dtype=float))
        standardised = (values - np.array(self.means)) / np.array(self.scales)
        contributions = standardised * np.array(self.coefficients)
        total = float(np.abs(contributions).sum()) or 1.0
        ranked = np.argsort(np.abs(contributions))[::-1]
        return [
            {
                "feature": self.features[i],
                "value": float(values[i]),
                "contribution": float(contributions[i]),
                "share": round(abs(float(contributions[i])) / total, 4),
                "direction": "increases" if contributions[i] > 0 else "decreases",
                "mechanism": self.feature_notes.get(self.features[i], ""),
            }
            for i in ranked
        ]

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "disease": self.disease,
            "features": self.features,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "means": self.means,
            "scales": self.scales,
            "fidelity_r2": self.fidelity_r2,
            "teacher_version": self.teacher_version,
            "residual_std": self.residual_std,
            "created_at": self.created_at,
            "feature_notes": self.feature_notes,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "LightweightModel":
        return cls(**json.loads(Path(path).read_text()))

    @property
    def size_bytes(self) -> int:
        return len(json.dumps(self.to_dict()).encode())

    def describe(self) -> Dict[str, object]:
        return {
            "disease": self.disease,
            "n_features": len(self.features),
            "fidelity_r2": self.fidelity_r2,
            "size_bytes": self.size_bytes,
            "teacher_version": self.teacher_version,
            "usable_offline": self.fidelity_r2 >= 0.6,
            "note": (
                "Distilled surrogate for edge use. Accuracy is below the full model; "
                f"it reproduces {self.fidelity_r2:.0%} of the full model's variance."
            ),
        }
