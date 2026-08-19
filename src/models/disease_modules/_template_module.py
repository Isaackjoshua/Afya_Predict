"""Template for a new disease module.

Copy this file (or run `python scripts/add_new_disease.py`), rename the class,
set `slug` to match the YAML filename, and override only what is genuinely
specific to the disease. `StandardDiseaseModule` already implements the whole
`BaseDiseaseModule` contract, so an empty subclass is a working module.

Checklist for a new disease
---------------------------
1. `config/diseases/<slug>.yaml` — proxies, lag ranges, mechanisms, thresholds,
   recommendations. At least three non-optional sources (rule #1).
2. This class, registered in `src/models/registry.py` (the script does it).
3. `python scripts/add_new_disease.py --validate <slug>` to check the contract.
4. A backtest: `python scripts/run_backtest.py --disease <slug>` — the model
   must beat all three naive baselines before it is deployed (rule #10).
"""

from __future__ import annotations

from typing import List

from src.core.types import Alert, Recommendation, RiskLevel
from src.models.standard_module import StandardDiseaseModule


class TemplateDiseaseModule(StandardDiseaseModule):
    """Rename me. Set `slug` to the disease's YAML filename stem."""

    slug = "_template"

    # --- Optional overrides ------------------------------------------------
    # def adjust_risk_level(self, level, incidence, importation_risk, low_confidence) -> RiskLevel:
    #     """Apply a biological or programmatic constraint the thresholds miss.
    #
    #     Example: a temperature floor below which transmission is implausible,
    #     or a structural-vulnerability escalation.
    #     """
    #     return super().adjust_risk_level(level, incidence, importation_risk, low_confidence)

    # def generate_recommendations(self, alert: Alert) -> List[Recommendation]:
    #     """Add disease-specific actions on top of the YAML templates."""
    #     recommendations = super().generate_recommendations(alert)
    #     return recommendations

    # def predict(self, matrix, district: str, **kwargs):
    #     """Post-process the forecast — smoothing, caps, extra disclaimers."""
    #     return super().predict(matrix, district, **kwargs)
