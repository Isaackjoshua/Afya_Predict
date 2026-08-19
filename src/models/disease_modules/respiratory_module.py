"""Acute respiratory infection: air quality plus seasonality.

ARI is syndromic — influenza, RSV, SARS-CoV-2 and bacterial pneumonia all land
in the same DHIS2 counter. The module therefore forecasts *syndromic burden*,
which is what determines oxygen and antibiotic demand, and says so in the
explanation rather than implying a pathogen it cannot identify.

Disease-specific behaviour: a short 4-week horizon (respiratory dynamics move
faster than the climate signals malaria depends on), and a pollution-episode
flag that fires when particulates spike well above the district's own baseline.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.types import Alert, PredictionResult, Recommendation
from src.models.standard_module import StandardDiseaseModule

#: Z-score above which recent PM2.5 counts as an acute pollution episode.
POLLUTION_EPISODE_Z = 1.5


class RespiratoryModule(StandardDiseaseModule):
    slug = "respiratory"

    def generate_recommendations(self, alert: Alert) -> List[Recommendation]:
        recommendations = super().generate_recommendations(alert)
        if self._pollution_episode() and alert.risk_level in ("medium", "high", "critical"):
            recommendations.insert(
                0,
                Recommendation(
                    action=(
                        f"Issue an air-quality advisory for {alert.district}: reduce outdoor "
                        "exertion, improve cooking ventilation, and prioritise children under 5 "
                        "and people with chronic lung disease for early review."
                    ),
                    priority=alert.risk_level,
                    timeframe_days=3,
                    responsible="District Environmental Health Officer",
                    rationale=(
                        "Particulate levels are well above this district's own baseline, which "
                        "precedes respiratory presentations by one to two weeks."
                    ),
                ),
            )
        return recommendations

    def detect_outbreak(
        self,
        predictions: Sequence[PredictionResult],
        actuals: Optional[pd.Series] = None,
    ) -> List[Alert]:
        alerts = super().detect_outbreak(predictions, actuals)
        for alert in alerts:
            alert.explanation += (
                " This is a syndromic forecast covering influenza, RSV, SARS-CoV-2 and bacterial "
                "pneumonia together; sentinel sampling is needed to identify the circulating "
                "pathogen."
            )
        return alerts

    def _pollution_episode(self) -> bool:
        matrix = self.feature_matrix
        if matrix is None or "pm25_ug_m3_zscore" not in matrix.X.columns:
            return False
        recent = matrix.X["pm25_ug_m3_zscore"].tail(4).to_numpy(dtype=float)
        finite = recent[np.isfinite(recent)]
        return bool(finite.size and float(np.nanmax(finite)) >= POLLUTION_EPISODE_Z)
