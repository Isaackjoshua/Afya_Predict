"""Automated retraining (shortcoming #3, critical rule #4).

Retraining is a scheduled background behaviour, not an operator's chore. Two
things can trigger it:

* **cadence** — the disease config's `retrain_frequency` (monthly by default);
* **drift** — the residual monitor firing on live prediction error.

Before a refit is promoted it must beat the incumbent on a held-out window,
otherwise the new model is discarded and the old one keeps serving. That gate
is what stops an automated loop from quietly degrading itself over a season of
bad data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.core.logging import get_logger
from src.core.timeutils import shift_week, to_epi_week
from src.core.types import RegionConfig
from src.models.base_model import BaseDiseaseModule
from src.models.drift_detector import DriftDetector, residual_series

log = get_logger("models.retrain")

RETRAIN_INTERVAL_DAYS: Dict[str, int] = {
    "weekly": 7,
    "fortnightly": 14,
    "monthly": 30,
    "quarterly": 91,
}


@dataclass
class RetrainDecision:
    """Why a retrain did or did not happen, and what came of it."""

    disease: str
    should_retrain: bool
    reasons: List[str] = field(default_factory=list)
    drift_events: List[dict] = field(default_factory=list)
    performed: bool = False
    promoted: bool = False
    incumbent_mae: Optional[float] = None
    candidate_mae: Optional[float] = None
    decided_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["decided_at"] = self.decided_at.isoformat(timespec="seconds")
        return payload


class AutoRetrainer:
    """Decide, execute and gate model refits for one disease module."""

    def __init__(
        self,
        module: BaseDiseaseModule,
        rolling_window_months: int = 24,
        min_improvement: float = 0.0,
        settings=None,
    ) -> None:
        self.module = module
        self.rolling_window_weeks = int(rolling_window_months * 4.345)
        self.min_improvement = min_improvement
        self.settings = settings or get_settings()

    # -- state -------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        path = Path(self.settings.artifact_dir) / self.module.slug
        path.mkdir(parents=True, exist_ok=True)
        return path / "retrain_state.json"

    def load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except Exception:  # noqa: BLE001
            return {}

    def save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, default=str))

    # -- decision ----------------------------------------------------------
    def should_retrain(
        self,
        residuals: Optional[Sequence[float]] = None,
        now: Optional[datetime] = None,
    ) -> RetrainDecision:
        """Cadence + drift check."""
        now = now or datetime.utcnow()
        decision = RetrainDecision(disease=self.module.slug, should_retrain=False)
        state = self.load_state()

        if not self.module.is_trained():
            decision.should_retrain = True
            decision.reasons.append("no trained model exists yet")
            return decision

        interval = RETRAIN_INTERVAL_DAYS.get(self.module.config.model.retrain_frequency, 30)
        last = state.get("last_retrain")
        if last:
            try:
                elapsed = (now - datetime.fromisoformat(last)).days
                if elapsed >= interval:
                    decision.should_retrain = True
                    decision.reasons.append(
                        f"scheduled refit: {elapsed} days since the last one "
                        f"(cadence {self.module.config.model.retrain_frequency}, {interval} days)"
                    )
            except ValueError:
                decision.should_retrain = True
                decision.reasons.append("unreadable retrain timestamp; refitting to be safe")
        else:
            decision.should_retrain = True
            decision.reasons.append("no retrain history recorded")

        if residuals is not None and len(residuals) >= self.settings.drift_check_min_residuals:
            detector = DriftDetector(min_observations=self.settings.drift_check_min_residuals)
            events = detector.scan(list(residuals))
            if events:
                decision.should_retrain = True
                decision.drift_events = [e.to_dict() for e in events]
                decision.reasons.append(
                    f"concept drift detected by {', '.join(sorted({e.detector for e in events}))} "
                    "— this is the Google Flu Trends failure mode, caught automatically"
                )
        return decision

    # -- execution ---------------------------------------------------------
    def retrain(
        self,
        panel,
        decision: Optional[RetrainDecision] = None,
        end_week: Optional[str] = None,
        holdout_weeks: int = 12,
    ) -> RetrainDecision:
        """Refit on the latest rolling window and promote only if it is better."""
        decision = decision or RetrainDecision(
            disease=self.module.slug, should_retrain=True, reasons=["manual trigger"]
        )
        matrix = self.module.build_feature_matrix(panel)
        weeks = matrix.weeks
        if len(weeks) < 52:
            decision.reasons.append(f"only {len(weeks)} weeks of history; refit skipped")
            return decision

        end_week = end_week or weeks[-1]
        window_start = shift_week(end_week, -self.rolling_window_weeks)
        in_window = [w for w in weeks if window_start <= w <= end_week]
        holdout = set(in_window[-holdout_weeks:])
        training = set(in_window[:-holdout_weeks]) or set(in_window)

        train_matrix = _slice_weeks(matrix, training)
        holdout_matrix = _slice_weeks(matrix, holdout)

        incumbent_mae = self._evaluate(self.module, holdout_matrix)

        previous_models = dict(self.module.models)
        self.module.train(train_matrix)
        candidate_mae = self._evaluate(self.module, holdout_matrix)

        decision.performed = True
        decision.incumbent_mae = incumbent_mae
        decision.candidate_mae = candidate_mae

        improved = (
            incumbent_mae is None
            or candidate_mae is None
            or candidate_mae <= incumbent_mae * (1 - self.min_improvement)
        )
        if improved:
            decision.promoted = True
            decision.reasons.append(
                f"promoted: holdout MAE {candidate_mae:.3f} vs incumbent "
                f"{incumbent_mae:.3f}" if incumbent_mae and candidate_mae
                else "promoted: no incumbent to compare against"
            )
            self.module.save()
            state = self.load_state()
            state.update(
                {
                    "last_retrain": datetime.utcnow().isoformat(timespec="seconds"),
                    "training_weeks": [min(training), max(training)],
                    "holdout_mae": candidate_mae,
                    "promoted": True,
                }
            )
            self.save_state(state)
        else:
            self.module.models = previous_models
            decision.promoted = False
            decision.reasons.append(
                f"rejected: candidate holdout MAE {candidate_mae:.3f} is worse than the "
                f"incumbent's {incumbent_mae:.3f}; keeping the existing model"
            )
        return decision

    def _evaluate(self, module: BaseDiseaseModule, matrix) -> Optional[float]:
        """Mean absolute error of the module's models on a holdout matrix."""
        usable = matrix.dropna_rows()
        if usable.X.empty or not module.is_trained():
            return None
        errors: List[float] = []
        for district in usable.districts:
            model = module.model_for(district)
            if model is None:
                continue
            local = usable.for_district(district)
            if local.X.empty:
                continue
            predictions = model.predict(local.X)
            errors.extend(np.abs(predictions - local.y.to_numpy(dtype=float)).tolist())
        return float(np.mean(errors)) if errors else None

    # -- monitoring --------------------------------------------------------
    def record_residuals(self, actual: Sequence[float], predicted: Sequence[float]) -> int:
        """Append live prediction errors to the monitored stream."""
        residuals = residual_series(actual, predicted)
        state = self.load_state()
        stream = list(state.get("residuals", []))
        stream.extend(float(r) for r in residuals)
        stream = stream[-500:]
        state["residuals"] = stream
        self.save_state(state)
        return len(stream)

    def monitored_residuals(self) -> List[float]:
        return list(self.load_state().get("residuals", []))


def _slice_weeks(matrix, weeks: set):
    """Restrict a `FeatureMatrix` to a set of weeks, keeping its metadata."""
    from src.feature_engineering.builder import FeatureMatrix

    mask = matrix.X.index.get_level_values("week").isin(weeks)
    return FeatureMatrix(
        X=matrix.X[mask],
        y=matrix.y[mask],
        disease=matrix.disease,
        feature_names=list(matrix.feature_names),
        provenance=matrix.provenance,
        lag_fits=matrix.lag_fits,
        row_quality=None if matrix.row_quality is None else matrix.row_quality[mask],
        target_column=matrix.target_column,
        population=matrix.population,
    )


def run_retraining_cycle(
    diseases: Optional[Sequence[str]] = None,
    region: Optional[RegionConfig] = None,
    history_weeks: int = 208,
    end_week: Optional[str] = None,
) -> List[dict]:
    """Check and, where needed, retrain every registered disease."""
    from src.core.config_loader import cached_region_config
    from src.data_ingestion.normalizer import ingest
    from src.models.registry import build_module, list_modules

    region = region or cached_region_config()
    end_week = end_week or to_epi_week(date.today())
    start_week = shift_week(end_week, -history_weeks)

    out: List[dict] = []
    for slug in diseases or list_modules():
        module = build_module(slug, region=region)
        module.load()
        retrainer = AutoRetrainer(module)
        decision = retrainer.should_retrain(residuals=retrainer.monitored_residuals())
        if decision.should_retrain:
            sources = sorted(set(module.config.required_sources) | {"dhis2"})
            panel = ingest(sources, start_week, end_week, region=region)
            decision = retrainer.retrain(panel, decision=decision, end_week=end_week)
        out.append(decision.to_dict())
        log.info(
            "%s: retrain=%s promoted=%s (%s)",
            slug, decision.should_retrain, decision.promoted, "; ".join(decision.reasons),
        )
    return out
