"""Counterfactual reasoning: "what if rainfall had been 20% lower?"

Attribution says which drivers mattered; a counterfactual says what would have
had to be different for the answer to change. That is the form an official can
act on — it identifies the lever, and (with the intervention tracker) whether
pulling it worked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.types import DriverExplanation
from src.explainability.natural_language import label_for

DEFAULT_PERTURBATIONS = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)


@dataclass
class Counterfactual:
    """One "if X had been Y" scenario and its effect on the forecast."""

    feature: str
    proxy: str
    lag_weeks: int
    original_value: float
    counterfactual_value: float
    relative_change: float
    original_prediction: float
    counterfactual_prediction: float

    @property
    def delta(self) -> float:
        return self.counterfactual_prediction - self.original_prediction

    @property
    def relative_delta(self) -> float:
        if self.original_prediction <= 0:
            return 0.0
        return self.delta / self.original_prediction

    def to_sentence(self, risk_before: str = "", risk_after: str = "") -> str:
        direction = "lower" if self.relative_change < 0 else "higher"
        effect = "fall" if self.delta < 0 else "rise"
        sentence = (
            f"If {label_for_proxy(self.proxy)}"
            + (f" {self.lag_weeks} weeks ago" if self.lag_weeks else "")
            + f" had been {abs(self.relative_change):.0%} {direction} "
            f"({self.original_value:,.1f} -> {self.counterfactual_value:,.1f}), "
            f"predicted cases would {effect} by {abs(self.delta):,.0f} "
            f"({abs(self.relative_delta):.0%})"
        )
        if risk_before and risk_after and risk_before != risk_after:
            sentence += f", moving risk from {risk_before.upper()} to {risk_after.upper()}"
        return sentence + "."


def label_for_proxy(proxy: str) -> str:
    from src.explainability.natural_language import PROXY_LABELS

    return PROXY_LABELS.get(proxy, proxy.replace("_", " "))


def generate_counterfactuals(
    model,
    row: pd.Series,
    drivers: Sequence[DriverExplanation],
    provenance: Optional[Dict[str, dict]] = None,
    perturbations: Sequence[float] = DEFAULT_PERTURBATIONS,
    top_k: int = 3,
) -> List[Counterfactual]:
    """Perturb the top drivers one at a time and re-run the model.

    Only *modifiable observable* drivers are perturbed — asking "what if the
    week of the year had been different" is not an actionable question.
    """
    provenance = provenance or {}
    frame = row.to_frame().T
    baseline = float(model.predict(frame)[0])

    out: List[Counterfactual] = []
    for driver in drivers:
        record = provenance.get(driver.feature, {})
        if record.get("kind") in ("seasonality", "connectivity"):
            continue
        if driver.feature not in row.index:
            continue
        original = row[driver.feature]
        if not np.isfinite(original) or original == 0:
            continue

        best: Optional[Counterfactual] = None
        for change in perturbations:
            perturbed = frame.copy()
            new_value = float(original) * (1 + change)
            perturbed.loc[:, driver.feature] = new_value
            prediction = float(model.predict(perturbed)[0])
            candidate = Counterfactual(
                feature=driver.feature,
                proxy=driver.proxy or record.get("proxy", driver.feature),
                lag_weeks=driver.lag_weeks,
                original_value=float(original),
                counterfactual_value=new_value,
                relative_change=change,
                original_prediction=baseline,
                counterfactual_prediction=prediction,
            )
            if best is None or abs(candidate.delta) > abs(best.delta):
                best = candidate
        if best is not None and abs(best.relative_delta) > 0.02:
            out.append(best)
        if len(out) >= top_k:
            break
    return sorted(out, key=lambda c: abs(c.delta), reverse=True)


def headline_counterfactual(
    model,
    row: pd.Series,
    drivers: Sequence[DriverExplanation],
    classify_risk,
    incidence_fn,
    provenance: Optional[Dict[str, dict]] = None,
) -> Optional[str]:
    """The single most informative counterfactual, phrased for the dashboard."""
    scenarios = generate_counterfactuals(model, row, drivers, provenance=provenance)
    if not scenarios:
        return None
    top = scenarios[0]
    before = classify_risk(incidence_fn(top.original_prediction))
    after = classify_risk(incidence_fn(top.counterfactual_prediction))
    return top.to_sentence(risk_before=before, risk_after=after)


def threshold_counterfactual(
    model,
    row: pd.Series,
    feature: str,
    target_prediction: float,
    search_range: tuple = (-0.9, 3.0),
    steps: int = 60,
) -> Optional[float]:
    """The value of `feature` at which the forecast would hit `target_prediction`.

    Answers the operational question directly: "how much more rain before this
    district crosses the outbreak threshold?"
    """
    if feature not in row.index:
        return None
    original = float(row[feature])
    if not np.isfinite(original) or original == 0:
        return None

    frame = row.to_frame().T
    multipliers = np.linspace(1 + search_range[0], 1 + search_range[1], steps)
    predictions = []
    for multiplier in multipliers:
        perturbed = frame.copy()
        perturbed.loc[:, feature] = original * multiplier
        predictions.append(float(model.predict(perturbed)[0]))
    predictions = np.array(predictions)

    crossings = np.where(np.diff(np.sign(predictions - target_prediction)) != 0)[0]
    if not len(crossings):
        return None
    index = int(crossings[0])
    return float(original * multipliers[index])
