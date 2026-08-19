"""Estimate what an intervention actually changed (shortcoming #15).

The identification problem, stated plainly: malaria falls after a bed-net
campaign. It also falls every year at the end of the rains. Attributing the
whole drop to the campaign is wrong, and so is attributing none of it.

This module does not claim to solve causal inference. It provides three
triangulating estimates, each with its own stated assumption, and reports them
together with an explicit confidence label:

1. **Forecast counterfactual** — the platform predicted this district's
   trajectory *before* the intervention. The gap between that forecast and the
   observed outcome is the effect, assuming the forecast would have stayed
   accurate. This is the estimate the platform is uniquely able to make.
2. **Difference-in-differences** — compare the change in treated districts
   against comparable untreated districts over the same weeks, which nets out
   anything seasonal or national.
3. **Pre/post** — the naive comparison, reported only as a reference point
   because it is the one people reach for by default and it is usually wrong.

Where the three disagree, the report says so rather than picking a favourite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.core.timeutils import shift_week, sort_weeks
from src.core.types import Intervention

log = get_logger("intervention.impact")

MIN_WEEKS_EACH_SIDE = 4


@dataclass
class ImpactEstimate:
    """Triangulated effect estimate for one intervention."""

    intervention_id: str
    disease: str
    district: str
    intervention_type: str
    weeks_evaluated: int = 0
    forecast_counterfactual_effect: Optional[float] = None
    did_effect: Optional[float] = None
    pre_post_effect: Optional[float] = None
    relative_effect: Optional[float] = None
    control_districts: List[str] = field(default_factory=list)
    confidence: str = "insufficient_data"
    caveats: List[str] = field(default_factory=list)

    @property
    def estimates(self) -> Dict[str, Optional[float]]:
        return {
            "forecast_counterfactual": self.forecast_counterfactual_effect,
            "difference_in_differences": self.did_effect,
            "pre_post": self.pre_post_effect,
        }

    @property
    def agreement(self) -> Optional[str]:
        """Do the independent estimates point the same way?"""
        values = [v for v in (self.forecast_counterfactual_effect, self.did_effect) if v is not None]
        if len(values) < 2:
            return None
        signs = {np.sign(v) for v in values}
        if len(signs) > 1:
            return "conflicting"
        spread = abs(values[0] - values[1]) / max(abs(np.mean(values)), 1e-9)
        return "consistent" if spread < 0.5 else "same_direction_different_magnitude"

    def narrative(self) -> str:
        if self.confidence == "insufficient_data":
            return (
                f"Not enough observation weeks around this {self.intervention_type} in "
                f"{self.district} to estimate an effect yet."
            )
        primary = (
            self.forecast_counterfactual_effect
            if self.forecast_counterfactual_effect is not None
            else self.did_effect
        )
        if primary is None:
            return f"No usable estimate for the {self.intervention_type} in {self.district}."
        direction = "fewer" if primary < 0 else "more"
        text = (
            f"After the {self.intervention_type} in {self.district}, observed cases ran about "
            f"{abs(primary):,.0f} {direction} than expected over {self.weeks_evaluated} weeks"
        )
        if self.relative_effect is not None:
            text += f" ({abs(self.relative_effect):.0%} {'below' if primary < 0 else 'above'} expectation)"
        if self.agreement == "conflicting":
            text += (
                ". The counterfactual and difference-in-differences estimates disagree in sign, "
                "so treat this as unresolved rather than an effect"
            )
        elif self.control_districts:
            text += f", compared against {len(self.control_districts)} similar untreated district(s)"
        text += f". Confidence: {self.confidence}."
        if self.caveats:
            text += " " + " ".join(self.caveats)
        return text

    def to_dict(self) -> dict:
        return {
            "intervention_id": self.intervention_id,
            "disease": self.disease,
            "district": self.district,
            "intervention_type": self.intervention_type,
            "weeks_evaluated": self.weeks_evaluated,
            "estimates": self.estimates,
            "relative_effect": self.relative_effect,
            "agreement": self.agreement,
            "control_districts": self.control_districts,
            "confidence": self.confidence,
            "caveats": self.caveats,
            "narrative": self.narrative(),
        }


class ImpactEstimator:
    """Compare observed outcomes against forecast and control counterfactuals."""

    def __init__(self, region=None, min_weeks: int = MIN_WEEKS_EACH_SIDE) -> None:
        self.region = region
        self.min_weeks = min_weeks

    def estimate(
        self,
        intervention: Intervention,
        observed: pd.Series,
        forecast: Optional[pd.Series] = None,
        control_series: Optional[pd.DataFrame] = None,
        effect_lag_weeks: Optional[int] = None,
    ) -> ImpactEstimate:
        """Estimate the effect of one intervention.

        `observed` is the treated district's weekly case series indexed by week;
        `forecast` is what the platform predicted for those same weeks *before*
        the intervention; `control_series` holds comparable untreated districts.
        """
        from src.intervention_tracking.intervention_logger import InterventionLogger

        lag = (
            effect_lag_weeks
            if effect_lag_weeks is not None
            else InterventionLogger.effect_lag(intervention.intervention_type)
        )
        estimate = ImpactEstimate(
            intervention_id=intervention.intervention_id,
            disease=intervention.disease,
            district=intervention.district,
            intervention_type=intervention.intervention_type,
        )

        weeks = sort_weeks(observed.index.astype(str).tolist())
        start = intervention.started_week
        effect_start = shift_week(start, lag)
        pre = [w for w in weeks if w < start]
        post = [w for w in weeks if w >= effect_start]

        if len(pre) < self.min_weeks or len(post) < self.min_weeks:
            estimate.caveats.append(
                f"needs at least {self.min_weeks} weeks before the intervention and "
                f"{self.min_weeks} after its {lag}-week effect lag"
            )
            return estimate

        estimate.weeks_evaluated = len(post)
        observed_post = observed.reindex(post).astype(float)

        # 1. Forecast counterfactual.
        if forecast is not None:
            expected = forecast.reindex(post).astype(float)
            pair = pd.concat([observed_post, expected], axis=1).dropna()
            if len(pair) >= self.min_weeks:
                effect = float((pair.iloc[:, 0] - pair.iloc[:, 1]).sum())
                estimate.forecast_counterfactual_effect = effect
                baseline = float(pair.iloc[:, 1].sum())
                if baseline > 0:
                    estimate.relative_effect = effect / baseline

        # 2. Difference in differences.
        if control_series is not None and not control_series.empty:
            did, controls = self._difference_in_differences(observed, control_series, pre, post)
            estimate.did_effect = did
            estimate.control_districts = controls

        # 3. Pre/post — reported, but never trusted alone.
        pre_mean = float(observed.reindex(pre).astype(float).mean())
        post_mean = float(observed_post.mean())
        estimate.pre_post_effect = (post_mean - pre_mean) * len(post)
        estimate.caveats.append(
            "The pre/post figure is shown for reference only: it cannot separate the "
            "intervention from the season."
        )

        estimate.confidence = self._confidence(estimate, intervention)
        return estimate

    def _difference_in_differences(
        self,
        observed: pd.Series,
        control_series: pd.DataFrame,
        pre: Sequence[str],
        post: Sequence[str],
    ):
        """(treated post - treated pre) - (control post - control pre)."""
        controls = [
            c for c in control_series.columns
            if control_series[c].reindex(pre).notna().sum() >= self.min_weeks
            and control_series[c].reindex(post).notna().sum() >= self.min_weeks
        ]
        if not controls:
            return None, []
        treated_change = float(observed.reindex(post).mean() - observed.reindex(pre).mean())
        control_change = float(
            control_series[controls].reindex(post).mean().mean()
            - control_series[controls].reindex(pre).mean().mean()
        )
        return (treated_change - control_change) * len(post), controls

    def _confidence(self, estimate: ImpactEstimate, intervention: Intervention) -> str:
        """A deliberately conservative label — this is not a trial."""
        signals = 0
        if estimate.forecast_counterfactual_effect is not None:
            signals += 1
        if estimate.did_effect is not None:
            signals += 1
        if intervention.coverage >= 0.5:
            signals += 1
        if estimate.weeks_evaluated >= 8:
            signals += 1

        if estimate.agreement == "conflicting":
            return "unresolved"
        if signals >= 4:
            return "moderate"      # the ceiling: observational, never "high"
        if signals >= 2:
            return "low"
        return "very_low"

    # -- portfolio view ----------------------------------------------------
    def summarise(self, estimates: Sequence[ImpactEstimate]) -> pd.DataFrame:
        """Which response types are associated with the largest reductions.

        Explicitly associational. It is a prioritisation aid for planning, not
        evidence of causation, and the column names say so.
        """
        rows = []
        for estimate in estimates:
            primary = (
                estimate.forecast_counterfactual_effect
                if estimate.forecast_counterfactual_effect is not None
                else estimate.did_effect
            )
            if primary is None:
                continue
            rows.append(
                {
                    "intervention_type": estimate.intervention_type,
                    "disease": estimate.disease,
                    "district": estimate.district,
                    "associated_case_difference": primary,
                    "relative_effect": estimate.relative_effect,
                    "confidence": estimate.confidence,
                    "weeks_evaluated": estimate.weeks_evaluated,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["intervention_type", "n", "mean_associated_difference", "note"]
            )
        frame = pd.DataFrame(rows)
        grouped = (
            frame.groupby("intervention_type")
            .agg(
                n=("associated_case_difference", "size"),
                mean_associated_difference=("associated_case_difference", "mean"),
                mean_relative_effect=("relative_effect", "mean"),
            )
            .sort_values("mean_associated_difference")
            .reset_index()
        )
        grouped["note"] = "associational, not causal — observational estimate"
        return grouped
