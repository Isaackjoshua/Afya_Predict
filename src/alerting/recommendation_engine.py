"""Turn an alert into costed, timeboxed, owned actions (shortcoming #12).

Recommendation *text* is configuration, not code (critical rule #9): each
disease YAML carries the templates, defaulted to Tanzania's IDSR guidance, and
a health office can edit them without a release. This engine adds the
operational scaffolding around that text — quantities scaled to the district's
population and forecast burden, a deadline matched to severity, and an owner.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from src.core.config_loader import load_alert_rules
from src.core.logging import get_logger
from src.core.types import Alert, DiseaseConfig, RISK_ORDER, Recommendation, RegionConfig, RiskLevel

log = get_logger("alerting.recommendations")

#: Response deadline by severity, in days.
TIMEFRAME_DAYS: Dict[str, int] = {"low": 30, "medium": 21, "high": 7, "critical": 3}

#: Who owns the response at each severity.
RESPONSIBLE: Dict[str, str] = {
    "low": "District Health Management Team",
    "medium": "District Health Management Team",
    "high": "Regional Health Management Team",
    "critical": "National Emergency Operations Centre",
}

#: Commodity planning factors, expressed per predicted case unless noted.
#: Deliberately conservative and openly editable — these are planning figures
#: for pre-positioning, not clinical dosing guidance.
COMMODITY_RULES: Dict[str, Dict[str, tuple]] = {
    "cholera": {
        "ORS": (6.0, "sachets per predicted case"),
        "chlorine tablets": (0.05, "tablets per person at risk"),
        "RDT": (1.5, "rapid tests per predicted case"),
    },
    "malaria": {
        "RDT": (2.0, "rapid tests per predicted case"),
        "ACT": (1.2, "treatment courses per predicted case"),
        "LLIN": (0.4, "nets per person in affected wards"),
    },
    "respiratory": {
        "amoxicillin DT": (1.1, "courses per predicted case"),
        "oxygen cylinders": (0.02, "cylinders per predicted case"),
    },
    "tuberculosis": {
        "GeneXpert cartridges": (8.0, "cartridges per predicted case"),
    },
    "hiv": {
        "HIV test kits": (12.0, "kits per predicted diagnosis"),
        "PrEP months": (3.0, "months of PrEP per predicted diagnosis"),
    },
}

#: Share of the district population treated as "at risk" for coverage maths.
AT_RISK_SHARE: Dict[str, float] = {"low": 0.02, "medium": 0.05, "high": 0.12, "critical": 0.25}


class RecommendationEngine:
    """Build the recommendation list attached to an alert."""

    def __init__(
        self,
        config: DiseaseConfig,
        region: RegionConfig,
        rules: Optional[dict] = None,
    ) -> None:
        self.config = config
        self.region = region
        self.rules = rules if rules is not None else load_alert_rules()

    # -- main --------------------------------------------------------------
    def build(self, alert: Alert) -> List[Recommendation]:
        """Configured actions for this level, plus derived logistics steps."""
        level = alert.risk_level
        actions = self._templates_for(level)
        timeframe = TIMEFRAME_DAYS.get(level, 14)
        owner = RESPONSIBLE.get(level, "District Health Management Team")

        out: List[Recommendation] = []
        for action in actions:
            out.append(
                Recommendation(
                    action=self._fill(action, alert),
                    priority=level,
                    timeframe_days=timeframe,
                    responsible=owner,
                    quantity=self._quantity_hint(action, alert),
                    rationale=self._rationale(alert),
                )
            )

        out.extend(self._commodity_recommendations(alert, timeframe, owner))
        spatial = self._spatial_recommendation(alert, timeframe, owner)
        if spatial:
            out.append(spatial)
        if alert.low_data_confidence:
            out.insert(
                0,
                Recommendation(
                    action=(
                        "Verify this signal with field reports before committing resources — "
                        "input data for this district was incomplete or estimated."
                    ),
                    priority=level,
                    timeframe_days=min(timeframe, 3),
                    responsible="District Surveillance Officer",
                    rationale=self._confidence_rationale(alert),
                ),
            )
        return out

    # -- pieces ------------------------------------------------------------
    def _templates_for(self, level: RiskLevel) -> List[str]:
        """Actions for this level, inheriting everything below it.

        A critical alert should not silently drop the "increase RDT stock"
        step that the medium level specified.
        """
        configured = self.config.recommendations or {}
        wanted = RISK_ORDER[1 : RISK_ORDER.index(level) + 1] if level != "low" else ()
        actions: List[str] = []
        for name in wanted:
            for action in configured.get(name, []) or []:
                if action not in actions:
                    actions.append(action)
        if not actions:
            actions = list(configured.get(level, []) or [])
        if not actions and level != "low":
            actions = [
                f"Review {self.config.name.lower()} preparedness for this district "
                "(no response template configured — add one in "
                f"config/diseases/{self.config.slug}.yaml)"
            ]
        return actions

    def _commodity_recommendations(
        self, alert: Alert, timeframe: int, owner: str
    ) -> List[Recommendation]:
        """Quantify the pre-positioning implied by the forecast."""
        rules = COMMODITY_RULES.get(self.config.slug, {})
        if not rules or alert.risk_level == "low":
            return []
        try:
            population = float(self.region.population_of(alert.district))
        except KeyError:
            population = 0.0
        at_risk = population * AT_RISK_SHARE.get(alert.risk_level, 0.05)
        cases = max(alert.predicted_cases, 0.0)

        out: List[Recommendation] = []
        for commodity, (factor, basis) in rules.items():
            quantity = cases * factor if "per predicted" in basis else at_risk * factor
            if quantity < 1:
                continue
            out.append(
                Recommendation(
                    action=(
                        f"Pre-position {_round_stock(quantity):,} {commodity} in {alert.district} "
                        f"ahead of week {alert.target_week}"
                    ),
                    priority=alert.risk_level,
                    timeframe_days=timeframe,
                    responsible="Medical Stores Department / District Pharmacist",
                    quantity=f"{_round_stock(quantity):,} ({factor} {basis})",
                    rationale=(
                        f"Sized from the forecast of {cases:,.0f} cases "
                        f"({alert.predicted_incidence_per_1000:.2f} per 1,000) "
                        f"{alert.lead_time_weeks} weeks ahead."
                    ),
                )
            )
        return out

    def _spatial_recommendation(
        self, alert: Alert, timeframe: int, owner: str
    ) -> Optional[Recommendation]:
        """Name the corridor when risk is arriving rather than growing locally."""
        if alert.importation_risk < 0.3 or not alert.source_districts:
            return None
        names = ", ".join(s.district for s in alert.source_districts[:3])
        return Recommendation(
            action=(
                f"Coordinate cross-district surveillance with {names}: "
                f"{alert.importation_risk:.0%} of {alert.district}'s risk is imported along "
                "travel corridors, so screening and messaging should follow the route, "
                "not just the destination."
            ),
            priority=alert.risk_level,
            timeframe_days=timeframe,
            responsible=owner,
            rationale=(
                "Mobility-weighted diffusion identified these districts as the dominant "
                "sources of importation pressure."
            ),
        )

    def _rationale(self, alert: Alert) -> str:
        drivers = ", ".join(
            f"{d.proxy or d.feature} ({d.contribution_share:.0%})" for d in alert.top_drivers[:3]
        )
        base = (
            f"{alert.risk_level.upper()} risk: {alert.predicted_incidence_per_1000:.2f} per 1,000 "
            f"forecast for week {alert.target_week}, threshold {alert.threshold_crossed:.2f}, "
            f"{alert.lead_time_weeks} weeks of lead time."
        )
        return f"{base} Main drivers: {drivers}." if drivers else base

    def _confidence_rationale(self, alert: Alert) -> str:
        if alert.data_quality_flags:
            return "Input issues: " + "; ".join(alert.data_quality_flags[:3])
        return "Confidence interval is wide relative to the point estimate."

    def _fill(self, action: str, alert: Alert) -> str:
        """Substitute `{district}`/`{week}`/`{cases}` placeholders in templates."""
        return (
            action.replace("{district}", alert.district)
            .replace("{week}", alert.target_week)
            .replace("{cases}", f"{alert.predicted_cases:,.0f}")
            .replace("{region}", alert.region)
        )

    def _quantity_hint(self, action: str, alert: Alert) -> Optional[str]:
        """Rescale a hard-coded quantity in a template to this district's size."""
        match = re.search(r"([\d,]{3,})\s+([A-Za-z]+)", action)
        if not match:
            return None
        try:
            template_quantity = float(match.group(1).replace(",", ""))
        except ValueError:
            return None
        try:
            population = float(self.region.population_of(alert.district))
        except KeyError:
            return None
        # Templates are written for a ~400,000-person council.
        scaled = template_quantity * population / 400_000.0
        if abs(scaled - template_quantity) / max(template_quantity, 1) < 0.15:
            return None
        return (
            f"~{_round_stock(scaled):,} {match.group(2)} scaled to "
            f"{alert.district}'s population of {population:,.0f}"
        )


def _round_stock(value: float) -> int:
    """Round a stock figure to a sensible ordering granularity."""
    if value >= 10_000:
        return int(round(value / 1000.0) * 1000)
    if value >= 1_000:
        return int(round(value / 100.0) * 100)
    if value >= 100:
        return int(round(value / 10.0) * 10)
    return int(round(value))
