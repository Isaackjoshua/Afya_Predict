"""Tuberculosis: density-driven risk mapping over a long horizon.

TB is not a weather-driven epidemic, and pretending otherwise would produce a
confidently wrong seasonal forecast. What this module actually models is
**where notification intensity will be highest**, driven by crowding, indoor
air quality and labour-migration connectivity, over a 12-week horizon.

Two consequences:

* the forecast is deliberately smoothed — week-to-week TB notification noise is
  reporting variation, not transmission, so a spiky forecast would be spurious;
* alerts are framed as programme-targeting signals, not outbreak warnings.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.types import Alert, PredictionResult
from src.models.standard_module import StandardDiseaseModule

#: Weeks of smoothing applied to the point forecast.
SMOOTHING_WEEKS = 4


class TuberculosisModule(StandardDiseaseModule):
    slug = "tuberculosis"

    def predict(self, matrix, district: str, **kwargs) -> List[PredictionResult]:
        """Standard prediction, then damp week-to-week reporting noise."""
        n_weeks = kwargs.pop("n_weeks", 1)
        results = super().predict(
            matrix, district, n_weeks=max(n_weeks, SMOOTHING_WEEKS), **kwargs
        )
        if len(results) <= 1:
            return results

        smoothed = float(np.mean([r.predicted_cases for r in results[-SMOOTHING_WEEKS:]]))
        latest = results[-1]
        latest.predicted_cases = round(smoothed, 2)
        incidence = self.incidence_per_1000(smoothed, district)
        latest.risk_level = self.adjust_risk_level(
            self.classify_risk(incidence), incidence, latest.importation_risk, False
        )
        latest.risk_score = round(self.risk_score(incidence), 4)
        latest.natural_language_explanation += (
            f" (TB notifications are smoothed over {SMOOTHING_WEEKS} weeks: week-to-week "
            "variation in TB reporting reflects case-finding effort more than transmission.)"
        )
        return results[-n_weeks:] if n_weeks < len(results) else results

    def detect_outbreak(
        self,
        predictions: Sequence[PredictionResult],
        actuals: Optional[pd.Series] = None,
    ) -> List[Alert]:
        """Frame TB alerts as case-finding signals rather than outbreak warnings."""
        alerts = super().detect_outbreak(predictions, actuals)
        for alert in alerts:
            alert.explanation = (
                "TB case-finding signal (not an acute outbreak alert): " + alert.explanation
            )
        return alerts
