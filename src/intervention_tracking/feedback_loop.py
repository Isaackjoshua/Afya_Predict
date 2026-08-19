"""Feed intervention outcomes back into the platform (shortcoming #15).

Three distinct loops close here, and they are separated on purpose:

1. **Response-quality loop** — did the alert actually produce a response, and
   how fast? A system with 8 weeks of lead time and a 9-week response time has
   delivered no value, and only this loop can reveal that.
2. **Model-correction loop** — weeks following a substantial intervention are
   *contaminated* as training targets, because the model would have to attribute
   an averted outbreak to the drivers rather than to the response. Those weeks
   are flagged with a sample weight so the retrainer discounts them, instead of
   silently learning "high rainfall causes low malaria".
3. **Recommendation loop** — response types associated with larger reductions
   are surfaced when similar alerts recur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.core.timeutils import shift_week, weeks_between
from src.core.types import Alert, Intervention

log = get_logger("intervention.feedback")

#: Coverage above which an intervention is assumed to have changed the outcome
#: enough that the affected weeks should be discounted in retraining.
CONTAMINATION_COVERAGE = 0.25
#: Weight applied to contaminated weeks (0 = drop, 1 = keep at full weight).
CONTAMINATED_WEIGHT = 0.3


@dataclass
class ResponseAudit:
    """Did alerts lead to action, and how quickly?"""

    total_alerts: int = 0
    alerts_with_response: int = 0
    mean_response_weeks: Optional[float] = None
    median_response_weeks: Optional[float] = None
    lead_time_weeks: Optional[float] = None
    by_level: Dict[str, dict] = field(default_factory=dict)

    @property
    def response_rate(self) -> float:
        return self.alerts_with_response / self.total_alerts if self.total_alerts else 0.0

    @property
    def lead_time_used(self) -> Optional[float]:
        """Weeks of warning that survived the response delay.

        Negative means the response arrived after the outbreak would already
        have begun — the forecast's advantage was spent on latency.
        """
        if self.lead_time_weeks is None or self.mean_response_weeks is None:
            return None
        return self.lead_time_weeks - self.mean_response_weeks

    def to_dict(self) -> dict:
        return {
            "total_alerts": self.total_alerts,
            "alerts_with_response": self.alerts_with_response,
            "response_rate": round(self.response_rate, 4),
            "mean_response_weeks": self.mean_response_weeks,
            "median_response_weeks": self.median_response_weeks,
            "forecast_lead_time_weeks": self.lead_time_weeks,
            "effective_lead_time_weeks": self.lead_time_used,
            "by_level": self.by_level,
            "interpretation": self._interpretation(),
        }

    def _interpretation(self) -> str:
        if not self.total_alerts:
            return "No alerts issued in this window."
        if not self.alerts_with_response:
            return (
                f"{self.total_alerts} alert(s) issued and none recorded a response. Either "
                "responses are not being logged, or the alerts are not reaching decision-makers."
            )
        used = self.lead_time_used
        base = (
            f"{self.response_rate:.0%} of alerts produced a logged response, "
            f"typically {self.mean_response_weeks:.1f} week(s) after issue."
        )
        if used is None:
            return base
        if used <= 0:
            return (
                base
                + " That is longer than the forecast lead time, so the early warning is being "
                "consumed by response latency — the bottleneck is operational, not predictive."
            )
        return base + f" About {used:.1f} week(s) of the forecast lead time remain usable."


class FeedbackLoop:
    """Turn logged interventions into audits, weights and better advice."""

    def __init__(self, cache=None, settings=None) -> None:
        from offline.local_cache import LocalCache

        self.cache = cache or LocalCache(settings=settings)

    # -- 1. response quality ----------------------------------------------
    def audit_responses(
        self,
        alerts: Sequence[Alert],
        interventions: Optional[Sequence[Intervention]] = None,
    ) -> ResponseAudit:
        from src.core.timeutils import to_epi_week

        interventions = list(
            interventions if interventions is not None else self.cache.get_interventions(limit=2000)
        )
        by_alert: Dict[str, List[Intervention]] = {}
        for item in interventions:
            if item.alert_id:
                by_alert.setdefault(item.alert_id, []).append(item)

        audit = ResponseAudit(total_alerts=len(alerts))
        delays: List[int] = []
        by_level: Dict[str, Dict[str, list]] = {}

        lead_times = [a.lead_time_weeks for a in alerts if a.lead_time_weeks]
        audit.lead_time_weeks = float(np.mean(lead_times)) if lead_times else None

        for alert in alerts:
            responses = by_alert.get(alert.alert_id, [])
            level_bucket = by_level.setdefault(alert.risk_level, {"alerts": [], "delays": []})
            level_bucket["alerts"].append(alert.alert_id)
            if not responses:
                continue
            audit.alerts_with_response += 1
            first = min(responses, key=lambda i: i.started_week)
            delay = weeks_between(to_epi_week(alert.issued_at.date()), first.started_week)
            delays.append(delay)
            level_bucket["delays"].append(delay)

        if delays:
            audit.mean_response_weeks = round(float(np.mean(delays)), 2)
            audit.median_response_weeks = round(float(np.median(delays)), 2)
        audit.by_level = {
            level: {
                "alerts": len(bucket["alerts"]),
                "responses": len(bucket["delays"]),
                "response_rate": round(len(bucket["delays"]) / max(len(bucket["alerts"]), 1), 3),
                "mean_response_weeks": round(float(np.mean(bucket["delays"])), 2)
                if bucket["delays"] else None,
            }
            for level, bucket in by_level.items()
        }
        return audit

    # -- 2. model correction ----------------------------------------------
    def contamination_weights(
        self,
        index: pd.MultiIndex,
        interventions: Optional[Sequence[Intervention]] = None,
        effect_window_weeks: int = 12,
    ) -> pd.Series:
        """Sample weights that discount intervention-affected district-weeks.

        Without this the retrainer learns from outbreaks that were successfully
        averted and concludes the drivers were harmless — the model is punished
        for having worked.
        """
        from src.intervention_tracking.intervention_logger import InterventionLogger

        interventions = list(
            interventions if interventions is not None else self.cache.get_interventions(limit=2000)
        )
        weights = pd.Series(1.0, index=index, name="intervention_weight")
        if not interventions:
            return weights

        affected = 0
        for item in interventions:
            if item.coverage < CONTAMINATION_COVERAGE:
                continue
            lag = InterventionLogger.effect_lag(item.intervention_type)
            start = shift_week(item.started_week, lag)
            end = item.ended_week or shift_week(start, effect_window_weeks)
            mask = (
                (index.get_level_values("district") == item.district)
                & (index.get_level_values("week") >= start)
                & (index.get_level_values("week") <= end)
            )
            if mask.any():
                # Weight scales with coverage: a 30%-coverage campaign is less
                # contaminating than a 90% one.
                weight = 1.0 - (1.0 - CONTAMINATED_WEIGHT) * item.coverage
                weights[mask] = np.minimum(weights[mask], weight)
                affected += int(mask.sum())
        if affected:
            log.info(
                "discounted %d district-week(s) affected by %d logged intervention(s)",
                affected, len(interventions),
            )
        return weights

    def contaminated_weeks(
        self, interventions: Optional[Sequence[Intervention]] = None
    ) -> pd.DataFrame:
        """Explicit list of the district-weeks the retrainer will discount."""
        from src.intervention_tracking.intervention_logger import InterventionLogger

        interventions = list(
            interventions if interventions is not None else self.cache.get_interventions(limit=2000)
        )
        rows = []
        for item in interventions:
            if item.coverage < CONTAMINATION_COVERAGE:
                continue
            lag = InterventionLogger.effect_lag(item.intervention_type)
            rows.append(
                {
                    "district": item.district,
                    "disease": item.disease,
                    "intervention_type": item.intervention_type,
                    "effect_from": shift_week(item.started_week, lag),
                    "coverage": item.coverage,
                    "weight": round(1.0 - (1.0 - CONTAMINATED_WEIGHT) * item.coverage, 3),
                }
            )
        return pd.DataFrame(rows)

    # -- 3. recommendation loop -------------------------------------------
    def preferred_actions(
        self, disease: str, estimates: Optional[Sequence] = None, top_n: int = 3
    ) -> List[Dict[str, object]]:
        """Response types historically associated with the largest reductions."""
        from src.intervention_tracking.impact_estimator import ImpactEstimator

        if not estimates:
            return []
        summary = ImpactEstimator().summarise(estimates)
        if summary.empty:
            return []
        return [
            {
                "intervention_type": row["intervention_type"],
                "mean_associated_difference": round(float(row["mean_associated_difference"]), 2),
                "observations": int(row["n"]),
                "note": (
                    "Associated with the largest observed reductions in this district's history. "
                    "Observational evidence, not a causal estimate."
                ),
            }
            for _, row in summary.head(top_n).iterrows()
        ]
