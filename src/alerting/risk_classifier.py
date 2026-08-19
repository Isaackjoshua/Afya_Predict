"""Risk classification with trend, importation and uncertainty adjustments.

The raw threshold comparison is only the starting point. `config/alert_rules/`
adds three corrections that matter operationally:

* **trend boost** — a district climbing fast towards a threshold is treated as
  already there, because the response takes weeks to mount;
* **importation escalation** — a connected district can be raised even with
  flat local counts (shortcoming #10);
* **uncertainty penalty** — wide intervals or poor input data pull the level
  back and stamp the alert LOW DATA CONFIDENCE rather than over-claiming
  (rule #7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.core.config_loader import load_alert_rules
from src.core.logging import get_logger
from src.core.types import AlertThresholds, RISK_ORDER, RiskLevel

log = get_logger("alerting.classifier")


@dataclass
class Classification:
    """The level plus an audit trail of every adjustment applied."""

    level: RiskLevel
    base_level: RiskLevel
    score: float
    incidence_per_1000: float
    threshold_crossed: float
    low_confidence: bool = False
    adjustments: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.adjustments is None:
            self.adjustments = []


class RiskClassifier:
    """Turn a predicted incidence into an actionable risk level."""

    def __init__(
        self,
        thresholds: AlertThresholds,
        rules: Optional[dict] = None,
    ) -> None:
        self.thresholds = thresholds
        self.rules = rules if rules is not None else load_alert_rules()
        classification = self.rules.get("classification", {})
        self.trend_rules = classification.get("trend_boost", {})
        self.uncertainty_rules = classification.get("uncertainty_penalty", {})
        self.escalation_rules = self.rules.get("escalation", {})

    # -- base classification ----------------------------------------------
    def base_level(self, incidence: float) -> RiskLevel:
        if incidence >= self.thresholds.critical:
            return "critical"
        if incidence >= self.thresholds.high:
            return "high"
        if incidence >= self.thresholds.medium:
            return "medium"
        return "low"

    def threshold_for(self, level: RiskLevel) -> float:
        return getattr(self.thresholds, level)

    def score(self, incidence: float) -> float:
        """Continuous 0-1 score, piecewise-linear across the thresholds."""
        points = [
            (0.0, 0.0),
            (self.thresholds.low, 0.25),
            (self.thresholds.medium, 0.5),
            (self.thresholds.high, 0.75),
            (self.thresholds.critical, 1.0),
        ]
        for (t0, s0), (t1, s1) in zip(points, points[1:]):
            if incidence <= t1:
                if t1 <= t0:
                    return float(s1)
                return float(np.clip(s0 + (s1 - s0) * (incidence - t0) / (t1 - t0), 0.0, 1.0))
        return 1.0

    # -- full classification ----------------------------------------------
    def classify(
        self,
        incidence: float,
        recent_incidence: Optional[Sequence[float]] = None,
        importation_risk: float = 0.0,
        ci_width_ratio: Optional[float] = None,
        input_quality: Optional[float] = None,
    ) -> Classification:
        base = self.base_level(incidence)
        index = RISK_ORDER.index(base)
        adjustments: List[str] = []

        if self.trend_rules.get("enabled", True) and recent_incidence is not None:
            boost = self._trend_boost(incidence, recent_incidence)
            if boost:
                allowed = int(self.trend_rules.get("max_levels", 1))
                step = min(boost, allowed, len(RISK_ORDER) - 1 - index)
                if step > 0:
                    index += step
                    adjustments.append(
                        f"escalated {step} level(s): incidence is rising faster than "
                        f"{self.trend_rules.get('min_relative_increase', 0.5):.0%} "
                        "above its 4-week mean"
                    )

        importation_threshold = float(self.escalation_rules.get("importation_risk_escalation", 0.7))
        if importation_risk >= importation_threshold and index < len(RISK_ORDER) - 1:
            index += 1
            adjustments.append(
                f"escalated 1 level: importation risk {importation_risk:.0%} exceeds "
                f"{importation_threshold:.0%} — spread is arriving from connected districts"
            )

        low_confidence = False
        if self.uncertainty_rules.get("enabled", True):
            ci_threshold = float(self.uncertainty_rules.get("ci_width_ratio_threshold", 1.5))
            quality_threshold = float(self.uncertainty_rules.get("quality_score_threshold", 0.6))
            wide = ci_width_ratio is not None and ci_width_ratio > ci_threshold
            poor = input_quality is not None and input_quality < quality_threshold
            if wide or poor:
                low_confidence = True
                reason = []
                if wide:
                    reason.append(f"confidence interval is {ci_width_ratio:.1f}x the point estimate")
                if poor:
                    reason.append(f"mean input quality {input_quality:.2f}")
                if index > 0:
                    index -= 1
                    adjustments.append(
                        "de-escalated 1 level for low data confidence (" + "; ".join(reason) + ")"
                    )
                else:
                    adjustments.append("low data confidence (" + "; ".join(reason) + ")")

        return Classification(
            level=RISK_ORDER[index],  # type: ignore[arg-type]
            base_level=base,
            score=round(self.score(incidence), 4),
            incidence_per_1000=round(incidence, 5),
            threshold_crossed=self.threshold_for(RISK_ORDER[index]),  # type: ignore[arg-type]
            low_confidence=low_confidence,
            adjustments=adjustments,
        )

    def _trend_boost(self, incidence: float, recent: Sequence[float]) -> int:
        values = [v for v in recent if v is not None and np.isfinite(v)]
        if len(values) < 2:
            return 0
        baseline = float(np.mean(values[-4:]))
        if baseline <= 0:
            return 1 if incidence > 0 else 0
        relative = (incidence - baseline) / baseline
        minimum = float(self.trend_rules.get("min_relative_increase", 0.5))
        return 1 if relative >= minimum else 0

    # -- helpers -----------------------------------------------------------
    def levels_at_or_above(self, level: RiskLevel) -> List[str]:
        return list(RISK_ORDER[RISK_ORDER.index(level):])

    def should_notify_national(self, level: RiskLevel) -> bool:
        floor = self.escalation_rules.get("notify_national_from", "high")
        return RISK_ORDER.index(level) >= RISK_ORDER.index(floor)

    def should_notify_who(self, level: RiskLevel) -> bool:
        floor = self.escalation_rules.get("notify_who_from", "critical")
        return RISK_ORDER.index(level) >= RISK_ORDER.index(floor)
