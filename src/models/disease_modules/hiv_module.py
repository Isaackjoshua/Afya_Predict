"""HIV: mobility-based hotspot risk mapping.

This module is deliberately the most constrained of the five.

HIV incidence is not forecastable week to week from digital proxies, and any
system claiming otherwise should be distrusted. What *is* defensible is
**hotspot targeting**: identifying which districts, over a 12-week horizon,
carry structural risk concentration — labour-migration corridors, urban
network density, economic displacement — so that testing, PrEP and linkage
services are placed where the yield is highest.

Ethical constraints encoded here, not left to the operator:

* aggregate-only. The module never accepts or emits anything below district x
  week, so no individual or key-population group can be re-identified.
* no stigmatising language in generated output.
* every alert states its own limits, so a targeting signal is never read as an
  outbreak prediction.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from src.core.types import Alert, PredictionResult, Recommendation
from src.models.standard_module import StandardDiseaseModule

DISCLAIMER = (
    "This is a service-targeting signal, not an outbreak forecast: it estimates where "
    "new-diagnosis yield is likely to concentrate over the coming months so that testing, "
    "PrEP and linkage services can be placed accordingly. It says nothing about any "
    "individual, and it is computed only at district level."
)


class HIVModule(StandardDiseaseModule):
    slug = "hiv"

    def predict(self, matrix, district: str, **kwargs) -> List[PredictionResult]:
        results = super().predict(matrix, district, **kwargs)
        for result in results:
            result.natural_language_explanation += f" {DISCLAIMER}"
        return results

    def detect_outbreak(
        self,
        predictions: Sequence[PredictionResult],
        actuals: Optional[pd.Series] = None,
    ) -> List[Alert]:
        alerts = super().detect_outbreak(predictions, actuals)
        for alert in alerts:
            alert.explanation = f"HIV programme-targeting signal. {alert.explanation}"
        return alerts

    def generate_recommendations(self, alert: Alert) -> List[Recommendation]:
        recommendations = super().generate_recommendations(alert)
        recommendations.append(
            Recommendation(
                action=(
                    f"Co-design any {alert.district} outreach with local community "
                    "organisations before deployment, and report only aggregate results."
                ),
                priority=alert.risk_level,
                timeframe_days=21,
                responsible="District AIDS Control Coordinator",
                rationale=(
                    "Geographic targeting can stigmatise the communities it is meant to serve "
                    "unless it is designed with them; aggregate-only reporting is a hard "
                    "constraint of this module."
                ),
            )
        )
        return recommendations
