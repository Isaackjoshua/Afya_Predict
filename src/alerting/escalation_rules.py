"""Automatic escalation based on persistence and trend.

A district sitting at HIGH for three consecutive weeks is a different
operational situation from a district that just crossed HIGH once, even though
the threshold comparison is identical. These rules encode that difference so
sustained risk reaches the right desk without a human noticing the pattern.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.core.config_loader import load_alert_rules
from src.core.logging import get_logger
from src.core.types import Alert, RISK_ORDER

log = get_logger("alerting.escalation")


class EscalationEngine:
    """Apply persistence, trend and importation escalation to an alert."""

    def __init__(self, rules: Optional[dict] = None) -> None:
        self.rules = (rules if rules is not None else load_alert_rules()).get("escalation", {})

    def apply(self, alert: Alert, history: Sequence[Alert]) -> Alert:
        reasons: List[str] = []
        if alert.escalation_reason:
            reasons.append(alert.escalation_reason)

        needed = int(self.rules.get("consecutive_weeks_to_escalate", 2))
        streak = self._streak_at_or_above(history, alert.risk_level)
        index = RISK_ORDER.index(alert.risk_level)

        if streak >= needed and index < len(RISK_ORDER) - 1:
            alert.risk_level = RISK_ORDER[index + 1]  # type: ignore[assignment]
            alert.escalated = True
            reasons.append(
                f"escalated: {streak} consecutive alerts at {RISK_ORDER[index]} or above "
                f"(threshold {needed}) — sustained risk, not a single spike"
            )

        if reasons:
            alert.escalation_reason = "; ".join(reasons)
        return alert

    def _streak_at_or_above(self, history: Sequence[Alert], level: str) -> int:
        floor = RISK_ORDER.index(level)
        ordered = sorted(history, key=lambda a: a.issued_at, reverse=True)
        streak = 0
        for alert in ordered:
            if RISK_ORDER.index(alert.risk_level) >= floor:
                streak += 1
            else:
                break
        return streak

    def notify_targets(self, alert: Alert) -> List[str]:
        """Which recipient groups this alert must reach."""
        targets = ["district_health_officer"]
        national_floor = self.rules.get("notify_national_from", "high")
        who_floor = self.rules.get("notify_who_from", "critical")
        index = RISK_ORDER.index(alert.risk_level)
        if index >= RISK_ORDER.index("medium"):
            targets.append("regional_medical_officer")
        if index >= RISK_ORDER.index(national_floor):
            targets.append("national_eoc")
        if index >= RISK_ORDER.index(who_floor):
            targets.append("who_country_office")
        return targets
