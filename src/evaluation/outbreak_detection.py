"""Outbreak detection accuracy at the operational threshold.

Regression error is not the question agencies ask. They ask: when the system
said "outbreak coming", did one arrive; when it stayed quiet, was it right to;
and how much warning did we actually get? This module answers all three,
including **timeliness**, which is the whole justification for a predictive
rather than reactive system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.timeutils import weeks_between
from src.evaluation.metrics import average_precision, classification_metrics, roc_auc


@dataclass
class OutbreakEvaluation:
    """Detection performance for one disease at one threshold."""

    threshold_per_1000: float
    metrics: Dict[str, float] = field(default_factory=dict)
    auc: float = float("nan")
    average_precision: float = float("nan")
    mean_lead_time_weeks: float = float("nan")
    detected_events: int = 0
    total_events: int = 0

    #: Threshold at which F1 is maximised, from `threshold_sweep`. The
    #: configured alert threshold is a policy choice about acceptable false
    #: alarms; this is the point the data itself supports.
    optimal_threshold: Optional[float] = None
    optimal_f1: float = float("nan")

    @property
    def evaluable(self) -> bool:
        """False when the window contains no outbreak weeks to detect.

        Reported separately from a failure: "no events occurred" is not the
        same finding as "the model missed the events".
        """
        return self.total_events > 0 and np.isfinite(self.auc)

    @property
    def passes_acceptance(self) -> bool:
        """Acceptance criterion #5: AUC >= 0.75 for outbreak detection."""
        return bool(self.evaluable and self.auc >= 0.75)

    @property
    def verdict(self) -> str:
        if not self.evaluable:
            return (
                f"NOT EVALUABLE — no week crossed {self.threshold_per_1000:.3f} per 1,000 in this "
                "window, so detection accuracy cannot be measured here"
            )
        if self.passes_acceptance:
            return f"PASS — outbreak-detection AUC {self.auc:.3f} (>= 0.75 required)"
        return f"FAIL — outbreak-detection AUC {self.auc:.3f} is below the required 0.75"

    def summary(self) -> Dict[str, float]:
        return {
            "threshold_per_1000": self.threshold_per_1000,
            "auc": self.auc,
            "average_precision": self.average_precision,
            "sensitivity": self.metrics.get("recall", float("nan")),
            "specificity": self.metrics.get("specificity", float("nan")),
            "precision": self.metrics.get("precision", float("nan")),
            "f1": self.metrics.get("f1", float("nan")),
            "false_alarm_rate": self.metrics.get("false_alarm_rate", float("nan")),
            "mean_lead_time_weeks": self.mean_lead_time_weeks,
            "events_detected": self.detected_events,
            "events_total": self.total_events,
            "optimal_threshold": self.optimal_threshold,
            "optimal_f1": self.optimal_f1,
            "evaluable": self.evaluable,
            "passes_acceptance": self.passes_acceptance,
            "verdict": self.verdict,
        }


def label_outbreaks(
    incidence: pd.Series, threshold_per_1000: float, min_consecutive_weeks: int = 1
) -> pd.Series:
    """Boolean outbreak label from an incidence series.

    Requiring consecutive weeks above threshold filters single-week reporting
    artefacts, which would otherwise dominate the positive class.
    """
    above = incidence >= threshold_per_1000
    if min_consecutive_weeks <= 1:
        return above
    rolling = above.rolling(min_consecutive_weeks, min_periods=min_consecutive_weeks).sum()
    return rolling >= min_consecutive_weeks


def evaluate_detection(
    actual_incidence: pd.Series,
    predicted_incidence: pd.Series,
    threshold_per_1000: float,
    risk_scores: Optional[pd.Series] = None,
    min_consecutive_weeks: int = 1,
) -> OutbreakEvaluation:
    """Score outbreak detection at the disease's alert threshold."""
    aligned = pd.concat(
        [actual_incidence.rename("actual"), predicted_incidence.rename("predicted")], axis=1
    ).dropna()
    if aligned.empty:
        return OutbreakEvaluation(threshold_per_1000=threshold_per_1000)

    actual_label = label_outbreaks(aligned["actual"], threshold_per_1000, min_consecutive_weeks)
    predicted_label = aligned["predicted"] >= threshold_per_1000

    scores = (
        risk_scores.reindex(aligned.index)
        if risk_scores is not None
        else aligned["predicted"]
    )
    evaluation = OutbreakEvaluation(
        threshold_per_1000=threshold_per_1000,
        metrics=classification_metrics(actual_label, predicted_label),
        auc=roc_auc(actual_label, scores),
        average_precision=average_precision(actual_label, scores),
        total_events=int(actual_label.sum()),
        detected_events=int((actual_label & predicted_label).sum()),
    )
    # A point forecast trained on squared error regresses towards the mean, so
    # it can rank outbreak weeks well (high AUC) while rarely crossing the raw
    # threshold itself (low sensitivity). Reporting the operating point that
    # maximises F1 makes that gap visible and gives agencies a usable trigger.
    best_threshold, best_f1 = _optimal_operating_point(actual_label, aligned["predicted"])
    evaluation.optimal_threshold = best_threshold
    evaluation.optimal_f1 = best_f1
    return evaluation


def _optimal_operating_point(labels: pd.Series, scores: pd.Series) -> tuple:
    """The decision threshold on `scores` that maximises F1."""
    values = scores.dropna()
    if values.empty or labels.sum() == 0:
        return None, float("nan")
    candidates = np.unique(np.quantile(values, np.linspace(0.5, 0.999, 40)))
    best_threshold, best_f1 = None, 0.0
    for candidate in candidates:
        metrics = classification_metrics(labels, values >= candidate)
        if metrics["f1"] > best_f1:
            best_threshold, best_f1 = float(candidate), float(metrics["f1"])
    return best_threshold, best_f1


def evaluate_timeliness(
    actual_incidence: pd.Series,
    predicted_incidence: pd.Series,
    threshold_per_1000: float,
    forecast_horizon_weeks: int,
) -> Dict[str, float]:
    """How many weeks of warning the system actually delivered.

    For each observed outbreak onset, find the first week the forecast crossed
    the threshold beforehand. That gap, plus the horizon, is the operational
    lead time — the number that justifies the whole approach.
    """
    aligned = pd.concat(
        [actual_incidence.rename("actual"), predicted_incidence.rename("predicted")], axis=1
    ).dropna()
    if aligned.empty:
        return {"onsets": 0, "detected": 0, "mean_lead_time_weeks": float("nan")}

    weeks = list(aligned.index.astype(str))
    actual_above = (aligned["actual"] >= threshold_per_1000).to_numpy()
    predicted_above = (aligned["predicted"] >= threshold_per_1000).to_numpy()

    onsets = [
        i for i in range(len(actual_above))
        if actual_above[i] and (i == 0 or not actual_above[i - 1])
    ]
    lead_times: List[float] = []
    detected = 0
    for onset in onsets:
        warning = None
        # Look back up to 12 weeks for the first sustained warning.
        for i in range(max(0, onset - 12), onset + 1):
            if predicted_above[i]:
                warning = i
                break
        if warning is not None:
            detected += 1
            lead_times.append(weeks_between(weeks[warning], weeks[onset]) + forecast_horizon_weeks)

    return {
        "onsets": len(onsets),
        "detected": detected,
        "detection_rate": detected / len(onsets) if onsets else float("nan"),
        "mean_lead_time_weeks": float(np.mean(lead_times)) if lead_times else float("nan"),
        "median_lead_time_weeks": float(np.median(lead_times)) if lead_times else float("nan"),
        "max_lead_time_weeks": float(np.max(lead_times)) if lead_times else float("nan"),
    }


def threshold_sweep(
    actual_incidence: pd.Series,
    predicted_incidence: pd.Series,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Detection performance across candidate thresholds.

    Lets a health office choose its own operating point on the
    sensitivity-versus-false-alarm trade-off rather than inheriting ours.
    """
    rows = []
    for threshold in thresholds:
        evaluation = evaluate_detection(actual_incidence, predicted_incidence, threshold)
        rows.append({"threshold_per_1000": threshold, **evaluation.summary()})
    return pd.DataFrame(rows)
