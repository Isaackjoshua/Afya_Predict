"""Walk-forward cross-validation (shortcoming #13).

Random k-fold on a time series leaks the future into the past and produces
flattering, meaningless scores. Walk-forward is the honest alternative: train on
everything up to week `t`, predict weeks `t+1 .. t+h`, roll forward, repeat.
Every reported number is genuinely out of sample.

The splitter also enforces a **purge gap** equal to the forecast horizon, so a
target built by shifting cases forward cannot appear in both the training and
the test fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.core.timeutils import sort_weeks
from src.evaluation.benchmark import BaselineComparison, benchmark_against_naive
from src.evaluation.calibration import interval_calibration
from src.evaluation.metrics import interval_metrics, regression_metrics
from src.evaluation.outbreak_detection import OutbreakEvaluation, evaluate_detection, evaluate_timeliness

log = get_logger("evaluation.walkforward")


@dataclass
class FoldResult:
    """One train/test split's out-of-sample predictions and scores."""

    fold: int
    train_weeks: Tuple[str, str]
    test_weeks: Tuple[str, str]
    n_train: int
    n_test: int
    predictions: pd.DataFrame          # index (district, week), columns actual/predicted/lower/upper
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated out-of-sample performance across all folds."""

    disease: str
    horizon_weeks: int
    folds: List[FoldResult] = field(default_factory=list)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, float] = field(default_factory=dict)
    interval: Dict[str, float] = field(default_factory=dict)
    outbreak: Optional[OutbreakEvaluation] = None
    timeliness: Dict[str, float] = field(default_factory=dict)
    baselines: Optional[BaselineComparison] = None

    @property
    def passes_acceptance(self) -> bool:
        """Acceptance criteria #5 and #6 together.

        When the evaluation window contains no outbreak weeks the detection
        criterion is *not evaluable* rather than failed, so the verdict rests on
        the baseline gate alone and `acceptance_notes` says so explicitly.
        """
        baseline_ok = self.baselines is not None and self.baselines.passes
        if self.outbreak is None or not self.outbreak.evaluable:
            return bool(baseline_ok)
        return bool(self.outbreak.passes_acceptance and baseline_ok)

    @property
    def acceptance_notes(self) -> List[str]:
        notes = []
        if self.outbreak is None:
            notes.append("outbreak detection was not evaluated")
        elif not self.outbreak.evaluable:
            notes.append(self.outbreak.verdict)
        else:
            notes.append(self.outbreak.verdict)
            sensitivity = self.outbreak.metrics.get("recall", float("nan"))
            if np.isfinite(sensitivity) and sensitivity < 0.5 and self.outbreak.auc >= 0.75:
                notes.append(
                    f"ranking is strong (AUC {self.outbreak.auc:.2f}) but sensitivity at the "
                    f"configured threshold is only {sensitivity:.0%}: the point forecast is "
                    "conservative at the peaks. Use the optimal operating point "
                    + (f"({self.outbreak.optimal_threshold:.3f} per 1,000) "
                       if self.outbreak.optimal_threshold else "")
                    + "or the trend-boost rule in config/alert_rules to trigger earlier."
                )
        if self.baselines is not None:
            notes.append(self.baselines.verdict())
        if self.interval:
            coverage = self.interval.get("coverage", float("nan"))
            if np.isfinite(coverage) and abs(coverage - 0.95) > 0.1:
                notes.append(
                    f"interval coverage is {coverage:.0%} against a nominal 95%: intervals need "
                    "recalibration (see evaluation.calibration.recalibration_factor)"
                )
        return notes

    def report(self) -> Dict[str, object]:
        return {
            "disease": self.disease,
            "horizon_weeks": self.horizon_weeks,
            "folds": len(self.folds),
            "n_predictions": len(self.predictions),
            "accuracy": {k: _round(v) for k, v in self.metrics.items()},
            "intervals": {k: _round(v) for k, v in self.interval.items()},
            "outbreak_detection": None if self.outbreak is None else {
                k: _round(v) for k, v in self.outbreak.summary().items()
            },
            "timeliness": {k: _round(v) for k, v in self.timeliness.items()},
            "baselines": None if self.baselines is None else {
                "verdict": self.baselines.verdict(),
                "model_mae": _round(self.baselines.model_mae),
                "skill": {k: _round(v) for k, v in self.baselines.skill.items()},
                "passes": self.baselines.passes,
            },
            "passes_acceptance": self.passes_acceptance,
            "acceptance_notes": self.acceptance_notes,
        }

    def summary_line(self) -> str:
        mae = self.metrics.get("mae", float("nan"))
        auc = self.outbreak.auc if self.outbreak else float("nan")
        lead = self.timeliness.get("mean_lead_time_weeks", float("nan"))
        verdict = "PASS" if self.passes_acceptance else "FAIL"
        return (
            f"{self.disease}: MAE {mae:.2f}, outbreak AUC {auc:.3f}, "
            f"mean lead time {lead:.1f} weeks over {len(self.folds)} folds — {verdict}"
        )


class WalkForwardCV:
    """Roll a model forward through history, scoring only unseen weeks."""

    def __init__(
        self,
        module,
        initial_train_weeks: int = 104,
        step_weeks: int = 8,
        test_weeks: int = 8,
        max_folds: Optional[int] = None,
        purge_horizon: bool = True,
    ) -> None:
        self.module = module
        self.initial_train_weeks = initial_train_weeks
        self.step_weeks = step_weeks
        self.test_weeks = test_weeks
        self.max_folds = max_folds
        self.purge_horizon = purge_horizon

    # -- splitting ---------------------------------------------------------
    def splits(self, weeks: Sequence[str], horizon: int) -> Iterator[Tuple[List[str], List[str]]]:
        """Yield `(train_weeks, test_weeks)` with a purge gap between them."""
        weeks = sort_weeks(weeks)
        gap = horizon if self.purge_horizon else 0
        start = self.initial_train_weeks
        fold = 0
        while start + gap + self.test_weeks <= len(weeks):
            if self.max_folds is not None and fold >= self.max_folds:
                return
            train = weeks[:start]
            test = weeks[start + gap : start + gap + self.test_weeks]
            if not test:
                return
            yield train, test
            start += self.step_weeks
            fold += 1

    # -- running -----------------------------------------------------------
    def run(
        self,
        matrix,
        districts: Optional[Sequence[str]] = None,
        threshold_per_1000: Optional[float] = None,
    ) -> WalkForwardResult:
        """Fit and score across every fold."""
        from src.models.auto_retrain import _slice_weeks

        horizon = self.module.horizon
        weeks = matrix.weeks
        districts = list(districts or matrix.districts)
        populations = {d.name: float(d.population) for d in self.module.region.districts}

        result = WalkForwardResult(disease=self.module.slug, horizon_weeks=horizon)
        frames: List[pd.DataFrame] = []

        for fold_index, (train_weeks, test_weeks) in enumerate(self.splits(weeks, horizon)):
            train_matrix = _slice_weeks(matrix, set(train_weeks))
            test_matrix = _slice_weeks(matrix, set(test_weeks))
            if train_matrix.dropna_rows().X.empty or test_matrix.dropna_rows().X.empty:
                continue

            self.module.train(train_matrix, districts=districts)

            rows = []
            for district in districts:
                model = self.module.model_for(district)
                if model is None:
                    continue
                local = test_matrix.for_district(district).dropna_rows()
                if local.X.empty:
                    continue
                point = np.clip(model.predict(local.X), 0.0, None)
                # Use the module's own interval logic, so what is validated here
                # is exactly what production emits.
                lower, upper = self.module.interval_bounds(model, point)
                population = populations.get(district, 1.0)
                for i, index in enumerate(local.X.index):
                    rows.append(
                        {
                            "district": index[0],
                            "week": index[1],
                            "fold": fold_index,
                            "actual": float(local.y.iloc[i]),
                            "predicted": float(point[i]),
                            "lower": float(lower[i]),
                            "upper": float(upper[i]),
                            "actual_incidence": float(local.y.iloc[i]) / population * 1000.0,
                            "predicted_incidence": float(point[i]) / population * 1000.0,
                        }
                    )
            if not rows:
                continue

            fold_frame = pd.DataFrame(rows).set_index(["district", "week"])
            frames.append(fold_frame)
            result.folds.append(
                FoldResult(
                    fold=fold_index,
                    train_weeks=(train_weeks[0], train_weeks[-1]),
                    test_weeks=(test_weeks[0], test_weeks[-1]),
                    n_train=len(train_matrix.dropna_rows().X),
                    n_test=len(fold_frame),
                    predictions=fold_frame,
                    metrics=regression_metrics(fold_frame["actual"], fold_frame["predicted"]),
                )
            )
            log.info(
                "fold %d: train %s..%s (%d rows) -> test %s..%s, MAE %.2f",
                fold_index, train_weeks[0], train_weeks[-1], len(train_matrix.X),
                test_weeks[0], test_weeks[-1],
                result.folds[-1].metrics.get("mae", float("nan")),
            )

        if not frames:
            log.warning("%s: no usable walk-forward folds", self.module.slug)
            return result

        combined = pd.concat(frames)
        result.predictions = combined
        result.metrics = regression_metrics(combined["actual"], combined["predicted"])
        result.interval = interval_metrics(combined["actual"], combined["lower"], combined["upper"])

        threshold = (
            threshold_per_1000
            if threshold_per_1000 is not None
            else self.module.config.alerts.medium
        )
        result.outbreak = evaluate_detection(
            combined["actual_incidence"], combined["predicted_incidence"], threshold
        )
        national = combined.groupby("week")[["actual_incidence", "predicted_incidence"]].mean()
        national = national.reindex(sort_weeks(national.index.tolist()))
        result.timeliness = evaluate_timeliness(
            national["actual_incidence"], national["predicted_incidence"], threshold, horizon
        )
        result.baselines = self._benchmark(combined, matrix, horizon)
        log.info(result.summary_line())
        return result

    def _benchmark(self, combined: pd.DataFrame, matrix, horizon: int) -> BaselineComparison:
        """Compare the model against the three naive baselines (rule #10)."""
        history_column = matrix.target_column
        history = None
        if history_column in matrix.X.columns:
            history = matrix.X[history_column]
        else:
            lag1 = f"{history_column}_lag1"
            if lag1 in matrix.X.columns:
                history = matrix.X[lag1]
        if history is None:
            return benchmark_against_naive(
                combined["actual"], combined["predicted"], horizon=horizon
            )

        pieces = []
        for district, group in combined.groupby(level="district"):
            local_history = history.xs(district, level="district")
            local = group.droplevel("district")
            comparison_frame = pd.DataFrame(
                {"actual": local["actual"], "predicted": local["predicted"]}
            )
            pieces.append((district, comparison_frame, local_history))

        # Pool the per-district comparisons into one national verdict.
        actual = pd.concat([f["actual"].rename(d) for d, f, _ in pieces], axis=1).stack()
        predicted = pd.concat([f["predicted"].rename(d) for d, f, _ in pieces], axis=1).stack()
        history_stacked = pd.concat([h.rename(d) for d, _, h in pieces], axis=1).stack()
        actual.index.names = predicted.index.names = history_stacked.index.names = ["week", "district"]
        return benchmark_against_naive(
            actual.sort_index(), predicted.sort_index(),
            history=history_stacked.sort_index(), horizon=horizon,
        )


def _round(value, digits: int = 4):
    if isinstance(value, (int, bool)) or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return None if not np.isfinite(number) else round(number, digits)
