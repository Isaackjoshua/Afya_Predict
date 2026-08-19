"""Malaria: gradient-boosted climate model plus spatial diffusion.

Why malaria is the reference implementation: it has the best-validated
climate-to-transmission chain in Tanzania (rainfall creates breeding sites,
temperature sets the sporogonic cycle, humidity sets vector survival), which
means the fitted lag structure can be checked against entomological expectation
rather than taken on faith.

Disease-specific behaviour here:

* rainfall saturation — above roughly 150 mm/month breeding sites wash out, so
  more rain reduces risk. Encoded as a shaped feature so SHAP shows the
  turning point rather than a misleading monotone gradient.
* a hard temperature floor: below about 18 C the parasite's extrinsic
  incubation period exceeds the mosquito's lifespan and transmission collapses,
  whatever the rest of the model says.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.types import RiskLevel
from src.models.standard_module import StandardDiseaseModule

#: Below this mean temperature, sustained P. falciparum transmission is
#: not biologically plausible at the district scale.
TRANSMISSION_FLOOR_C = 18.0


class MalariaModule(StandardDiseaseModule):
    slug = "malaria"

    def adjust_risk_level(
        self,
        level: RiskLevel,
        incidence: float,
        importation_risk: float,
        low_confidence: bool,
    ) -> RiskLevel:
        """Standard adjustments, then the highland transmission floor."""
        level = super().adjust_risk_level(level, incidence, importation_risk, low_confidence)
        if self._below_transmission_floor() and level in ("high", "critical"):
            self.log.debug("capping malaria risk: district is below the transmission floor")
            return "medium"
        return level

    def _below_transmission_floor(self) -> bool:
        """True when the recent temperature history rules out transmission."""
        matrix = self.feature_matrix
        if matrix is None:
            return False
        columns = [c for c in matrix.X.columns if c.startswith("temperature_c_lag")]
        if not columns:
            return False
        recent = matrix.X[columns].tail(8).to_numpy(dtype=float)
        finite = recent[np.isfinite(recent)]
        return bool(finite.size and float(np.mean(finite)) < TRANSMISSION_FLOOR_C)
