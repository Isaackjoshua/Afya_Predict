"""Forecast accuracy metrics.

Point-forecast metrics (MAE/RMSE) answer "how far off were we"; classification
metrics at the alert threshold answer the question agencies actually ask —
"when you said outbreak, was there one, and did you miss any?". Both are
reported, because a model can look good on MAE while missing every outbreak.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------- regression
def regression_metrics(
    actual: Sequence[float], predicted: Sequence[float], baseline: Optional[Sequence[float]] = None
) -> Dict[str, float]:
    """MAE, RMSE, bias, MAPE-family and R², computed on finite pairs only."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if a.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan"),
                "r2": float("nan"), "smape": float("nan"), "mase": float("nan")}

    errors = p - a
    total = float(np.sum((a - a.mean()) ** 2))
    metrics = {
        "n": int(a.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "median_ae": float(np.median(np.abs(errors))),
        "r2": float(1 - np.sum(errors**2) / total) if total > 0 else float("nan"),
        # sMAPE rather than MAPE: case counts hit zero, and MAPE explodes there.
        "smape": float(
            np.mean(2 * np.abs(errors) / np.maximum(np.abs(a) + np.abs(p), 1e-9)) * 100
        ),
    }
    if baseline is not None:
        b = np.asarray(baseline, dtype=float)[mask]
        baseline_mae = float(np.mean(np.abs(b - a)))
        metrics["mase"] = float(metrics["mae"] / baseline_mae) if baseline_mae > 0 else float("nan")
        metrics["baseline_mae"] = baseline_mae
    else:
        # Scaled against the seasonal-naive-free in-sample first difference.
        scale = float(np.mean(np.abs(np.diff(a)))) if a.size > 1 else 0.0
        metrics["mase"] = float(metrics["mae"] / scale) if scale > 0 else float("nan")
    return metrics


# ----------------------------------------------------------- classification
def classification_metrics(
    actual_positive: Sequence[bool], predicted_positive: Sequence[bool]
) -> Dict[str, float]:
    """Confusion matrix and the derived rates, at a fixed decision threshold."""
    a = np.asarray(actual_positive, dtype=bool)
    p = np.asarray(predicted_positive, dtype=bool)
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]

    tp = int(np.sum(a & p))
    fp = int(np.sum(~a & p))
    fn = int(np.sum(a & ~p))
    tn = int(np.sum(~a & ~p))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced = (recall + specificity) / 2
    return {
        "n": int(n),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced),
        "accuracy": float((tp + tn) / n) if n else 0.0,
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def roc_auc(actual_positive: Sequence[bool], scores: Sequence[float]) -> float:
    """AUC via the rank (Mann-Whitney U) identity, with tie correction.

    Implemented directly so the acceptance criterion (AUC >= 0.75) can be
    checked without scikit-learn present.
    """
    y = np.asarray(actual_positive, dtype=bool)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # Average ranks within tied groups.
    sorted_scores = s[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(actual_positive: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve — the honest metric for rare events."""
    y = np.asarray(actual_positive, dtype=bool)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / y.sum())


def skill_score(model_error: float, baseline_error: float) -> float:
    """1 - model/baseline: positive means the model beats the baseline."""
    if not np.isfinite(baseline_error) or baseline_error <= 0:
        return float("nan")
    return float(1.0 - model_error / baseline_error)


# ------------------------------------------------------------------ interval
def interval_metrics(
    actual: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    nominal: float = 0.95,
) -> Dict[str, float]:
    """Coverage and sharpness of the prediction intervals.

    An interval that never misses but spans the whole plausible range is not
    useful; both numbers have to be read together.
    """
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    a, lo, hi = a[mask], lo[mask], hi[mask]
    if a.size == 0:
        return {"coverage": float("nan"), "nominal": nominal, "mean_width": float("nan"),
                "interval_score": float("nan"), "n": 0}

    inside = (a >= lo) & (a <= hi)
    alpha = 1 - nominal
    width = hi - lo
    penalty = (2 / alpha) * ((lo - a) * (a < lo) + (a - hi) * (a > hi))
    return {
        "n": int(a.size),
        "coverage": float(inside.mean()),
        "nominal": float(nominal),
        "coverage_gap": float(inside.mean() - nominal),
        "mean_width": float(width.mean()),
        # Winkler interval score: lower is better, rewards narrow *and* covering.
        "interval_score": float(np.mean(width + penalty)),
    }


def metrics_frame(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Tabulate a `{name: metrics}` mapping for reporting."""
    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
