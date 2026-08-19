"""The interface every disease module implements (shortcoming #11).

Adding a disease means: write a YAML config, subclass `BaseDiseaseModule`,
register it. The ingestion, training, evaluation, explanation and alerting
pipeline never changes — that is the whole point of the plugin architecture
(critical rule #5).

This base class already implements everything a typical disease needs, so a
concrete module is usually just a few dozen lines that override the parts that
are genuinely disease-specific (which features matter, how risk maps to
incidence, what the response looks like).
"""

from __future__ import annotations

import abc
import json
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.core.logging import get_logger
from src.core.timeutils import shift_week, sort_weeks
from src.core.types import (
    Alert,
    DiseaseConfig,
    DriverExplanation,
    PredictionResult,
    Recommendation,
    RegionConfig,
    RiskLevel,
    SourceRisk,
)
from src.data_ingestion.normalizer import Panel
from src.feature_engineering.builder import FeatureBuilder, FeatureMatrix
from src.models.backends import (
    BackendInfo,
    build_regressor,
    feature_importances,
    fit_with_optional_weights,
    usable_feature_mask,
)

log = get_logger("models.base")

#: Minimum district-weeks before a district gets its own model; below this it
#: uses the pooled model (see `transfer_learning`).
MIN_ROWS_FOR_LOCAL_MODEL = 78

#: Correction applied to holdout residual quantiles so 95% intervals actually
#: cover ~95% of future outcomes. Calibrated against walk-forward results; see
#: `_predict_rows` and `scripts/run_backtest.py`.
INTERVAL_INFLATION = 1.5


@dataclass
class TrainedModel:
    """One fitted estimator plus everything needed to reproduce and audit it."""

    estimator: Any
    backend: BackendInfo
    feature_names: List[str]
    scope: str                       # "pooled" or a district name
    trained_at: datetime = field(default_factory=datetime.utcnow)
    n_rows: int = 0
    train_weeks: Tuple[str, str] = ("", "")
    residual_std: float = 0.0
    residual_quantiles: Dict[str, float] = field(default_factory=dict)
    train_score: float = 0.0
    version: str = ""

    def __post_init__(self) -> None:
        if not self.version:
            stamp = self.trained_at.strftime("%Y%m%dT%H%M%S")
            self.version = f"{self.scope}-{self.backend.resolved}-{stamp}"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        aligned = X.reindex(columns=self.feature_names)
        return np.asarray(self.estimator.predict(aligned.to_numpy(dtype=float)), dtype=float)


@dataclass
class PredictionBundle:
    """Raw model output before explanation and alerting are layered on."""

    point: pd.Series
    lower: pd.Series
    upper: pd.Series
    model_version: str
    quality: pd.Series


class BaseDiseaseModule(abc.ABC):
    """Base class for every disease module."""

    #: registry key; defaults to the config slug
    slug: str = ""

    def __init__(
        self,
        config: DiseaseConfig,
        region: RegionConfig,
        settings=None,
    ) -> None:
        self.config = config
        self.region = region
        self.settings = settings or get_settings()
        self.slug = self.slug or config.slug
        self.log = get_logger(f"models.{self.slug}")
        self.builder = FeatureBuilder(config, region)
        self.models: Dict[str, TrainedModel] = {}   # scope -> model
        self.feature_matrix: Optional[FeatureMatrix] = None
        self.trained_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get_feature_config(self) -> List:
        """Return the digital proxy specs with their lag ranges."""

    @abc.abstractmethod
    def build_feature_matrix(self, panel: Panel, horizon_weeks: Optional[int] = None) -> FeatureMatrix:
        """Transform a fused panel into the model-ready feature matrix."""

    @abc.abstractmethod
    def train(self, matrix: FeatureMatrix, districts: Optional[Sequence[str]] = None) -> None:
        """Fit pooled and per-district models."""

    @abc.abstractmethod
    def predict(
        self, matrix: FeatureMatrix, district: str, horizon_weeks: Optional[int] = None
    ) -> List[PredictionResult]:
        """Forecast with confidence intervals, SHAP and recommendations."""

    @abc.abstractmethod
    def detect_outbreak(
        self, predictions: Sequence[PredictionResult], actuals: Optional[pd.Series] = None
    ) -> List[Alert]:
        """Compare predictions to thresholds and raise alerts."""

    @abc.abstractmethod
    def get_spatial_risk(
        self, district: str, travel_matrix: pd.DataFrame, panel: Optional[Panel] = None,
        week: Optional[str] = None,
    ) -> Tuple[float, List[SourceRisk]]:
        """Importation risk for `district` from its connected districts."""

    @abc.abstractmethod
    def generate_recommendations(self, alert: Alert) -> List[Recommendation]:
        """Actionable response steps for this alert level."""

    # ------------------------------------------------------------------
    # Shared implementation
    # ------------------------------------------------------------------
    @property
    def horizon(self) -> int:
        return self.config.model.forecast_horizon_weeks

    @property
    def target_column(self) -> str:
        return self.builder.target_column

    def is_trained(self) -> bool:
        return bool(self.models)

    def model_for(self, district: str) -> Optional[TrainedModel]:
        """The district's own model when it has one, else the pooled model."""
        return self.models.get(district) or self.models.get("pooled")

    # -- fitting -----------------------------------------------------------
    def _fit_scope(
        self, matrix: FeatureMatrix, scope: str, backend_name: Optional[str] = None
    ) -> Optional[TrainedModel]:
        """Fit one estimator over the rows of `matrix` (already scoped)."""
        usable = matrix.dropna_rows()
        if len(usable.X) < 24:
            self.log.warning("scope %s has only %d usable rows; skipped", scope, len(usable.X))
            return None

        # Drop columns that carry no signal in this training window (all-NaN or
        # constant). They cannot help the fit, and some backends fail outright
        # on them rather than ignoring them.
        full_matrix = usable.X.to_numpy(dtype=float)
        keep = usable_feature_mask(full_matrix)
        if not keep.any():
            self.log.warning("scope %s has no usable features; skipped", scope)
            return None
        dropped = int((~keep).sum())
        if dropped:
            self.log.debug(
                "scope %s: dropped %d constant/empty feature(s) of %d",
                scope, dropped, len(keep),
            )
        feature_names = [c for c, k in zip(usable.X.columns, keep) if k]
        X = full_matrix[:, keep]
        y = usable.y.to_numpy(dtype=float)
        weights = None
        if usable.row_quality is not None:
            # Low-quality inputs contribute less to the fit (rule #7).
            weights = np.clip(usable.row_quality.to_numpy(dtype=float), 0.1, 1.0)

        backend_choice = backend_name or self.config.model.primary

        # Prediction intervals must come from residuals the model has NOT seen.
        # In-sample residuals from a boosted ensemble are far too small, which
        # produced ~37% coverage on a nominal 95% interval in walk-forward
        # testing. So: fit on a chronological 80% to measure honest residuals,
        # then refit on everything for the model that actually serves.
        residuals = self._holdout_residuals(usable, X, y, weights, backend_choice)

        estimator, backend = build_regressor(
            backend_choice, random_state=self.settings.random_seed
        )
        fit_with_optional_weights(estimator, X, y, weights)

        fitted = np.asarray(estimator.predict(X), dtype=float)
        if residuals is None:
            # Too little data to hold anything out: fall back to in-sample
            # residuals inflated by a conservative factor, and say so.
            residuals = (y - fitted) * 2.5
        weeks = usable.weeks
        denominator = float(np.sum((y - y.mean()) ** 2))
        return TrainedModel(
            estimator=estimator,
            backend=backend,
            feature_names=feature_names,
            scope=scope,
            n_rows=len(usable.X),
            train_weeks=(weeks[0], weeks[-1]) if weeks else ("", ""),
            residual_std=float(np.std(residuals)),
            residual_quantiles={
                "q025": float(np.quantile(residuals, 0.025)),
                "q250": float(np.quantile(residuals, 0.25)),
                "q750": float(np.quantile(residuals, 0.75)),
                "q975": float(np.quantile(residuals, 0.975)),
            },
            train_score=float(1 - np.sum((y - fitted) ** 2) / denominator) if denominator > 0 else 0.0,
        )

    def _holdout_residuals(
        self, usable, X: np.ndarray, y: np.ndarray, weights, backend_choice: str
    ) -> Optional[np.ndarray]:
        """Out-of-sample residuals from a chronological internal split.

        The split is by *week*, not by row, so every district's observations for
        a held-out week are held out together — otherwise a district's own
        neighbours in the same week leak the answer.
        """
        weeks = usable.weeks
        if len(weeks) < 30:
            return None
        cutoff = weeks[int(len(weeks) * 0.8)]
        week_index = np.asarray(usable.X.index.get_level_values("week"))
        train_mask = week_index < cutoff
        holdout_mask = ~train_mask
        if train_mask.sum() < 20 or holdout_mask.sum() < 10:
            return None

        estimator, _ = build_regressor(backend_choice, random_state=self.settings.random_seed)
        try:
            fit_with_optional_weights(
                estimator, X[train_mask], y[train_mask],
                None if weights is None else weights[train_mask],
            )
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail training
            self.log.warning("holdout residual fit failed (%s); using inflated in-sample", exc)
            return None
        predictions = np.asarray(estimator.predict(X[holdout_mask]), dtype=float)
        return y[holdout_mask] - predictions

    def fit_models(
        self, matrix: FeatureMatrix, districts: Optional[Sequence[str]] = None
    ) -> Dict[str, TrainedModel]:
        """Fit the pooled model, then per-district models where data allows.

        Districts with too little history keep the pooled model — the simplest
        honest form of transfer learning (shortcoming #8): borrow strength, but
        never pretend a local fit exists when it does not.
        """
        self.feature_matrix = matrix
        districts = list(districts or matrix.districts)
        models: Dict[str, TrainedModel] = {}

        pooled = self._fit_scope(matrix, "pooled")
        if pooled is not None:
            models["pooled"] = pooled
            self.log.info(
                "%s pooled model: %d rows, backend %s, in-sample R2 %.3f",
                self.config.name, pooled.n_rows, pooled.backend, pooled.train_score,
            )

        for district in districts:
            local = matrix.for_district(district).dropna_rows()
            if len(local.X) < MIN_ROWS_FOR_LOCAL_MODEL:
                self.log.debug(
                    "%s: %s has %d rows (<%d); inheriting the pooled model",
                    self.config.name, district, len(local.X), MIN_ROWS_FOR_LOCAL_MODEL,
                )
                continue
            fitted = self._fit_scope(local, district)
            if fitted is not None:
                models[district] = fitted

        self.models = models
        self.trained_at = datetime.utcnow()
        self.log.info(
            "%s trained: 1 pooled + %d district model(s)", self.config.name, len(models) - 1
        )
        return models

    # -- prediction --------------------------------------------------------
    def _predict_rows(
        self, matrix: FeatureMatrix, district: str, rows: pd.DataFrame
    ) -> PredictionBundle:
        """Point forecast plus quality- and residual-aware intervals."""
        model = self.model_for(district)
        if model is None:
            raise RuntimeError(f"{self.config.name}: no trained model available for {district}")

        point = pd.Series(model.predict(rows), index=rows.index).clip(lower=0.0)

        quality = (
            matrix.row_quality.reindex(rows.index).fillna(0.5)
            if matrix.row_quality is not None
            else pd.Series(1.0, index=rows.index)
        )
        # Rule #1/#7: poorer inputs widen intervals rather than blocking output.
        widening = 1.0 + 1.5 * (1.0 - quality)
        if model.scope == "pooled" and district != "pooled":
            widening = widening * 1.25  # borrowed model, not a local fit

        # The internal holdout sits inside the training window, so its residual
        # spread still understates genuine future error. Walk-forward testing
        # measured 85% coverage on a nominal 95% interval without a correction;
        # INTERVAL_INFLATION closes that gap. It is a measured constant, not a
        # guess, and `evaluation.calibration.recalibration_factor` re-derives it
        # from live performance during retraining.
        lower_q = model.residual_quantiles.get("q025", -1.96 * model.residual_std) * INTERVAL_INFLATION
        upper_q = model.residual_quantiles.get("q975", 1.96 * model.residual_std) * INTERVAL_INFLATION
        lower = (point + lower_q * widening).clip(lower=0.0)
        upper = point + upper_q * widening

        return PredictionBundle(
            point=point, lower=lower, upper=upper, model_version=model.version, quality=quality
        )

    @staticmethod
    def interval_bounds(model: TrainedModel, point: np.ndarray, widening=1.0):
        """95% interval around a point forecast, from holdout residuals.

        Shared by `_predict_rows` and the walk-forward evaluator so the
        intervals that get validated are exactly the intervals that ship.
        """
        point = np.asarray(point, dtype=float)
        lower_q = model.residual_quantiles.get("q025", -1.96 * model.residual_std)
        upper_q = model.residual_quantiles.get("q975", 1.96 * model.residual_std)
        lower = np.clip(point + lower_q * INTERVAL_INFLATION * widening, 0.0, None)
        upper = point + upper_q * INTERVAL_INFLATION * widening
        return lower, upper

    def latest_feature_rows(
        self, matrix: FeatureMatrix, district: str, n_weeks: int = 1
    ) -> pd.DataFrame:
        """The most recent complete feature rows for a district."""
        local = matrix.for_district(district)
        rows = local.X.dropna(thresh=int(0.6 * local.X.shape[1]))
        if rows.empty:
            rows = local.X
        return rows.tail(n_weeks)

    # -- risk --------------------------------------------------------------
    def incidence_per_1000(self, cases: float, district: str) -> float:
        population = self.region.population_of(district)
        return float(cases) / max(population, 1) * 1000.0

    def classify_risk(self, incidence: float) -> RiskLevel:
        """Map per-1,000 incidence to a level using the disease's thresholds."""
        thresholds = self.config.alerts
        if incidence >= thresholds.critical:
            return "critical"
        if incidence >= thresholds.high:
            return "high"
        if incidence >= thresholds.medium:
            return "medium"
        return "low"

    def risk_score(self, incidence: float) -> float:
        """Continuous 0-1 severity, piecewise-linear between the thresholds."""
        thresholds = self.config.alerts
        points = [
            (0.0, 0.0),
            (thresholds.low, 0.25),
            (thresholds.medium, 0.5),
            (thresholds.high, 0.75),
            (thresholds.critical, 1.0),
        ]
        points = [(t, s) for t, s in points if np.isfinite(t)]
        for (t0, s0), (t1, s1) in zip(points, points[1:]):
            if incidence <= t1:
                if t1 <= t0:
                    return float(s1)
                return float(s0 + (s1 - s0) * (incidence - t0) / (t1 - t0))
        # Above critical: saturate towards 1.
        excess = incidence / max(thresholds.critical, 1e-9)
        return float(min(1.0, 1.0 - 0.0 + 0.0 * excess)) if excess >= 1 else 1.0

    # -- persistence -------------------------------------------------------
    def artifact_dir(self) -> Path:
        path = Path(self.settings.artifact_dir) / self.slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the fitted models and their metadata."""
        target = Path(path or self.artifact_dir() / "model.pkl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            pickle.dump({"slug": self.slug, "models": self.models,
                         "trained_at": self.trained_at}, handle)
        metadata = {
            "disease": self.config.name,
            "slug": self.slug,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "scopes": sorted(self.models),
            "backends": sorted({m.backend.resolved for m in self.models.values()}),
            "horizon_weeks": self.horizon,
            "versions": {scope: m.version for scope, m in self.models.items()},
        }
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))
        self.log.info("saved %s models to %s", self.config.name, target)
        return target

    def load(self, path: Optional[Path] = None) -> bool:
        source = Path(path or self.artifact_dir() / "model.pkl")
        if not source.exists():
            return False
        with open(source, "rb") as handle:
            payload = pickle.load(handle)
        self.models = payload.get("models", {})
        self.trained_at = payload.get("trained_at")
        return bool(self.models)

    # -- misc --------------------------------------------------------------
    def native_importance(self, district: str = "pooled") -> pd.Series:
        model = self.model_for(district)
        if model is None:
            return pd.Series(dtype=float)
        values = feature_importances(model.estimator)
        if values is None or len(values) != len(model.feature_names):
            return pd.Series(dtype=float)
        return pd.Series(values, index=model.feature_names).sort_values(ascending=False)

    def new_prediction_id(self) -> str:
        return str(uuid.uuid4())

    def describe(self) -> Dict[str, Any]:
        return {
            "disease": self.config.name,
            "slug": self.slug,
            "code": self.config.code,
            "transmission_mode": self.config.transmission_mode.value,
            "proxies": [p.name for p in self.config.digital_proxies],
            "sources": self.config.required_sources,
            "horizon_weeks": self.horizon,
            "spatial_enabled": self.config.spatial.enabled,
            "trained": self.is_trained(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "district_models": max(len(self.models) - 1, 0),
            "thresholds": self.config.alerts.model_dump(),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.slug} trained={self.is_trained()}>"
