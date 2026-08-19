"""Turn SHAP attributions into sentences a health officer can act on.

Numbers alone did not solve the trust problem: "0.34" means nothing at a
district health management team meeting, while "rainfall six weeks ago was well
above normal, which floods breeding sites — that alone accounts for a third of
this elevation" does.

Every sentence names the proxy, the lag, the direction and the mechanism,
because the mechanism is what makes a forecast checkable against local
knowledge.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.core.types import DriverExplanation, RiskLevel

#: Human-readable label per proxy, with the unit officials actually use.
PROXY_LABELS: Dict[str, str] = {
    "rainfall": "rainfall",
    "temperature": "temperature",
    "humidity": "humidity",
    "ndvi": "vegetation greenness",
    "air_quality_pm25": "fine-particulate air pollution",
    "air_quality_no2": "nitrogen-dioxide pollution",
    "mobility_inbound": "inbound travel",
    "mobility_outbound": "outbound travel",
    "population_density": "population density",
    "wash_access": "safe water and sanitation coverage",
    "sanitation": "improved sanitation coverage",
    "surface_water": "standing surface water",
    "case_history": "recent reported cases",
    "importation": "importation pressure from connected districts",
    "neighbourhood": "conditions in neighbouring districts",
    "mobility": "how connected this district is",
    "season": "the time of year",
    "one_health": "converging animal and environmental signals",
}

LEVEL_PHRASES: Dict[str, str] = {
    "low": "Risk is within the usual range",
    "medium": "Risk is elevated",
    "high": "Risk is high",
    "critical": "Risk is critical",
}


def label_for(driver: DriverExplanation) -> str:
    base = PROXY_LABELS.get(driver.proxy)
    if base:
        return base
    if "_x_" in driver.proxy:
        left, right = driver.proxy.split("_x_", 1)
        return (
            f"the combination of {PROXY_LABELS.get(left, left)} "
            f"and {PROXY_LABELS.get(right, right)}"
        )
    return driver.proxy.replace("_", " ")


def describe_driver(driver: DriverExplanation, include_mechanism: bool = True) -> str:
    """One sentence for one driver."""
    label = label_for(driver)
    when = (
        f" {driver.lag_weeks} week{'s' if driver.lag_weeks != 1 else ''} ago"
        if driver.lag_weeks > 0
        else ""
    )
    direction = "pushing risk up" if driver.direction == "increases" else "pulling risk down"
    share = f"{driver.contribution_share:.0%}"
    value = "" if driver.value is None else f" (measured {_format_value(driver.value)})"

    sentence = f"{label.capitalize()}{when}{value} is {direction}, accounting for {share} of the signal"
    if include_mechanism and driver.mechanism:
        sentence += f" — {driver.mechanism}"
    return sentence + "."


def explain_prediction(
    disease: str,
    district: str,
    target_week: str,
    predicted_cases: float,
    incidence_per_1000: float,
    risk_level: RiskLevel,
    drivers: Sequence[DriverExplanation],
    lead_time_weeks: int = 0,
    ci: Optional[tuple] = None,
    importation_risk: float = 0.0,
    source_districts: Optional[Sequence] = None,
    data_quality_flags: Optional[Sequence[str]] = None,
    low_confidence: bool = False,
) -> str:
    """Compose the full natural-language explanation for one prediction."""
    parts: List[str] = []

    horizon = (
        f" in {lead_time_weeks} week{'s' if lead_time_weeks != 1 else ''}"
        if lead_time_weeks
        else ""
    )
    headline = (
        f"{LEVEL_PHRASES.get(risk_level, 'Risk is elevated')} for {disease.lower()} in "
        f"{district}{horizon} (week {target_week}): about {predicted_cases:,.0f} cases, "
        f"{incidence_per_1000:.2f} per 1,000 people"
    )
    if ci and np.isfinite(ci[0]) and np.isfinite(ci[1]):
        headline += f" (95% interval {ci[0]:,.0f}-{ci[1]:,.0f})"
    parts.append(headline + ".")

    positive = [d for d in drivers if d.direction == "increases"][:3]
    negative = [d for d in drivers if d.direction == "decreases"][:2]

    if positive:
        parts.append("What is driving this: " + " ".join(describe_driver(d) for d in positive))
    if negative:
        parts.append("Holding it back: " + " ".join(describe_driver(d, include_mechanism=False) for d in negative))

    if importation_risk >= 0.3 and source_districts:
        names = ", ".join(str(getattr(s, "district", s)) for s in list(source_districts)[:3])
        parts.append(
            f"Roughly {importation_risk:.0%} of this risk is imported rather than local, "
            f"arriving mainly from {names}."
        )

    if low_confidence:
        parts.append(
            "LOW DATA CONFIDENCE — verify with field reports before acting. "
            + (
                "Input issues: " + "; ".join(list(data_quality_flags)[:3]) + "."
                if data_quality_flags
                else "Several input feeds were incomplete or estimated."
            )
        )
    elif data_quality_flags:
        parts.append("Data notes: " + "; ".join(list(data_quality_flags)[:2]) + ".")

    return " ".join(parts)


def summarise_drivers(drivers: Sequence[DriverExplanation], limit: int = 3) -> str:
    """Short comma-joined phrase, for SMS and alert headlines."""
    if not drivers:
        return "no dominant driver identified"
    phrases = []
    for driver in list(drivers)[:limit]:
        when = f" {driver.lag_weeks}w ago" if driver.lag_weeks > 0 else ""
        arrow = "up" if driver.direction == "increases" else "down"
        phrases.append(f"{label_for(driver)}{when} ({arrow}, {driver.contribution_share:.0%})")
    return ", ".join(phrases)


def sms_summary(
    disease: str,
    district: str,
    risk_level: RiskLevel,
    predicted_cases: float,
    target_week: str,
    drivers: Sequence[DriverExplanation],
    top_action: Optional[str] = None,
    low_confidence: bool = False,
) -> str:
    """A <=320-character alert for feature phones — 87% of Tanzanian handsets."""
    body = (
        f"AFYA-PREDICT {risk_level.upper()}: {disease} risk in {district}, week {target_week}. "
        f"~{predicted_cases:,.0f} cases expected. Drivers: {summarise_drivers(drivers, 2)}."
    )
    if top_action:
        body += f" ACTION: {top_action}"
    if low_confidence:
        body += " (LOW DATA CONFIDENCE - verify locally)"
    return body[:320]


def _format_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 0.01:
        return f"{value:.2f}"
    return f"{value:.3g}"
