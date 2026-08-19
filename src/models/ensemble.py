"""Multi-model ensembles (shortcoming #2 applied to models, not just data).

Fusing data sources protects against a feed failing. Fusing *models* protects
against a model class failing: gradient boosting extrapolates badly beyond its
training range, SARIMA ignores the digital proxies entirely, and ridge
underfits non-linear response shapes. Averaging them, with weights earned on
held-out performance, is what keeps the forecast usable when one component
starts drifting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.models.backends import build_regressor, fit_with_optional_weights, usable_feature_mask

log = get_logger("models.ensemble")


@dataclass
class EnsembleMember:
    """One fitted component of an ensemble, with its earned weight."""

    name: str
    estimator: object
    backend: object
    weight: float = 1.0
    validation_mae: float = float("inf")
    residual_std: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)


class WeightedEnsemble:
    """Inverse-error weighted ensemble over several regressor backends.

    Weights come from a held-out tail of the training window, not from the
    training fit, so a member that memorises the training data does not earn
    influence over the forecast.
    """

    def __init__(
        self,
        member_names: Sequence[str],
        validation_fraction: float = 0.2,
        min_validation_rows: int = 20,
        random_state: int = 42,
    ) -> None:
        self.member_names = list(member_names) or ["xgboost"]
        self.validation_fraction = validation_fraction
        self.min_validation_rows = min_validation_rows
        self.random_state = random_state
        self.members: List[EnsembleMember] = []
        self.feature_names: List[str] = []
        self.fitted_at: Optional[datetime] = None

    # -- fitting -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "WeightedEnsemble":
        """Fit every member and weight it by held-out MAE.

        The split is chronological, never random: shuffling a time series
        leaks the future into the validation set and inflates every weight.
        """
        # Same guard as the single-model path: a constant column helps nobody and
        # breaks some backends.
        raw = X.to_numpy(dtype=float)
        keep = usable_feature_mask(raw)
        if not keep.any():
            raise ValueError("no usable features: every column is constant or empty")
        self.feature_names = [c for c, k in zip(X.columns, keep) if k]
        matrix = raw[:, keep]
        target = np.asarray(y, dtype=float)

        split = self._split_point(len(target))
        X_train, y_train = matrix[:split], target[:split]
        X_valid, y_valid = matrix[split:], target[split:]
        weights_train = None if sample_weight is None else np.asarray(sample_weight)[:split]

        members: List[EnsembleMember] = []
        for name in self.member_names:
            estimator, backend = build_regressor(name, random_state=self.random_state)
            try:
                fit_with_optional_weights(estimator, X_train, y_train, weights_train)
            except Exception as exc:  # noqa: BLE001 - one member failing is survivable
                log.warning("ensemble member %s failed to fit: %s", name, exc)
                continue

            if len(y_valid) >= 3:
                predictions = np.asarray(estimator.predict(X_valid), dtype=float)
                mae = float(np.mean(np.abs(predictions - y_valid)))
                residual_std = float(np.std(predictions - y_valid))
            else:
                mae, residual_std = float("inf"), 0.0

            members.append(
                EnsembleMember(
                    name=name, estimator=estimator, backend=backend,
                    validation_mae=mae, residual_std=residual_std,
                )
            )

        if not members:
            raise RuntimeError("no ensemble member could be fitted")

        self.members = self._assign_weights(members)
        self.fitted_at = datetime.utcnow()
        log.info(
            "ensemble fitted: %s",
            ", ".join(f"{m.name}={m.weight:.2f}" for m in self.members),
        )
        return self

    def _split_point(self, n: int) -> int:
        validation = max(self.min_validation_rows, int(self.validation_fraction * n))
        split = n - validation
        # Keep at least two-thirds of the rows for training.
        return max(split, int(0.6 * n)) if split > 0 else n

    def _assign_weights(self, members: List[EnsembleMember]) -> List[EnsembleMember]:
        errors = np.array([m.validation_mae for m in members], dtype=float)
        if not np.isfinite(errors).any():
            for member in members:
                member.weight = 1.0 / len(members)
            return members
        finite = np.where(np.isfinite(errors), errors, np.nanmax(errors[np.isfinite(errors)]) * 10)
        inverse = 1.0 / np.maximum(finite, 1e-9)
        weights = inverse / inverse.sum()
        for member, weight in zip(members, weights):
            member.weight = float(weight)
        return members

    # -- prediction --------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        matrix = self._align(X)
        stacked = np.vstack([member.predict(matrix) for member in self.members])
        weights = np.array([member.weight for member in self.members], dtype=float)
        return np.average(stacked, axis=0, weights=weights)

    def predict_members(self, X) -> pd.DataFrame:
        """Per-member predictions — the ensemble's own disagreement diagnostic."""
        matrix = self._align(X)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(
            {member.name: member.predict(matrix) for member in self.members}, index=index
        )

    def _align(self, X) -> np.ndarray:
        """Restrict incoming rows to the features the ensemble was fitted on."""
        if isinstance(X, pd.DataFrame):
            return X.reindex(columns=self.feature_names).to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def prediction_spread(self, X) -> np.ndarray:
        """Weighted standard deviation across members.

        Wide disagreement is a genuine uncertainty signal and is folded into the
        confidence interval, so the alert honestly reflects model risk as well
        as data noise.
        """
        # predict_members is rows x members; transpose to members x rows.
        members = self.predict_members(X).to_numpy().T
        weights = np.array([m.weight for m in self.members], dtype=float)
        mean = np.average(members, axis=0, weights=weights)
        variance = np.average((members - mean) ** 2, axis=0, weights=weights)
        return np.sqrt(variance)

    # -- introspection -----------------------------------------------------
    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Weighted average of the members' native importances."""
        from src.models.backends import feature_importances

        total = np.zeros(len(self.feature_names), dtype=float)
        found = False
        for member in self.members:
            values = feature_importances(member.estimator)
            if values is not None and len(values) == len(self.feature_names):
                total += member.weight * values
                found = True
        return total if found else None

    def describe(self) -> List[dict]:
        return [
            {
                "member": m.name,
                "backend": str(m.backend),
                "weight": round(m.weight, 4),
                "validation_mae": None if not np.isfinite(m.validation_mae) else round(m.validation_mae, 4),
            }
            for m in self.members
        ]

    def get_params(self, deep: bool = True) -> dict:
        return {"member_names": self.member_names, "validation_fraction": self.validation_fraction}
