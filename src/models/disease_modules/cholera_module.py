"""Cholera: climate + WASH model with a heavier spatial weight.

Cholera differs from malaria in three ways that matter to the model:

* **WASH access is the structural gate.** Heavy rain floods latrines only where
  sanitation is weak; the same rainfall in a district with 85% improved
  sanitation produces far less transmission. The `rain x WASH-gap` interaction
  carries most of that signal.
* **Two very different lags coexist.** Flood contamination acts within 1-4
  weeks, while sea-surface and reservoir warming acts on a ~4-month lag. The
  configured ranges cover both, and the fitter picks per district.
* **Spread follows corridors.** Cholera moves along transport routes faster
  than it grows locally, so `importation_weight` is 0.45 against malaria's 0.3.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.core.types import Recommendation, RiskLevel, Alert
from src.models.standard_module import StandardDiseaseModule

#: Districts below this improved-water coverage are treated as structurally
#: vulnerable: any credible signal there is escalated rather than averaged away.
WASH_VULNERABILITY_THRESHOLD = 0.5


class CholeraModule(StandardDiseaseModule):
    slug = "cholera"

    def adjust_risk_level(
        self,
        level: RiskLevel,
        incidence: float,
        importation_risk: float,
        low_confidence: bool,
    ) -> RiskLevel:
        level = super().adjust_risk_level(level, incidence, importation_risk, low_confidence)
        return level

    def escalate_for_wash(self, district: str, level: RiskLevel) -> RiskLevel:
        """Raise the level in structurally vulnerable districts."""
        from src.core.types import RISK_ORDER

        try:
            wash = self.region.get(district).wash_access
        except KeyError:
            return level
        index = RISK_ORDER.index(level)
        if wash < WASH_VULNERABILITY_THRESHOLD and index in (1, 2):
            return RISK_ORDER[index + 1]  # type: ignore[return-value]
        return level

    def generate_recommendations(self, alert: Alert) -> List[Recommendation]:
        """Standard recommendations plus a water-point action sized to the district."""
        recommendations = super().generate_recommendations(alert)
        if alert.risk_level == "low":
            return recommendations
        try:
            district = self.region.get(alert.district)
        except KeyError:
            return recommendations

        # Roughly one improved water point per 250 people is the planning norm
        # used for rural chlorination campaigns.
        water_points = max(int(district.population * (1 - district.wash_access) / 250), 1)
        recommendations.append(
            Recommendation(
                action=(
                    f"Test and chlorinate approximately {water_points:,} community water points "
                    f"in {alert.district}; {(1 - district.wash_access):.0%} of the population "
                    "lacks improved water access, which is the structural driver here."
                ),
                priority=alert.risk_level,
                timeframe_days=7 if alert.risk_level in ("high", "critical") else 14,
                responsible="District Water Engineer / Environmental Health Officer",
                quantity=f"{water_points:,} water points",
                rationale=(
                    "Cholera risk is the product of contamination pressure and sanitation "
                    "coverage; the coverage side is the one an agency can change in weeks."
                ),
            )
        )
        return recommendations
