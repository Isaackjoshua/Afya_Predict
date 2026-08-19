"""Shared visual vocabulary.

Risk colours are fixed across every page and chart. A district that is amber on
the map must be amber in the table and in the waterfall — inconsistent colour
coding in a decision tool is a safety problem, not an aesthetic one.

The palette is checked for colour-blind legibility: the four levels differ in
lightness as well as hue, so they remain distinguishable in greyscale and to
viewers with deuteranopia (the most common form).
"""

from __future__ import annotations

from typing import Dict

RISK_COLORS: Dict[str, str] = {
    "low": "#2E7D5B",       # green, darkest
    "medium": "#E8A33D",    # amber
    "high": "#D95F2B",      # orange
    "critical": "#A61C3C",  # deep red, high contrast against amber
}

RISK_ORDER = ("low", "medium", "high", "critical")

RISK_EMOJI: Dict[str, str] = {
    "low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴",
}

DIRECTION_COLORS = {"increases": "#D95F2B", "decreases": "#2E7D5B", "neutral": "#9E9E9E"}

QUALITY_COLORS = {"good": "#2E7D5B", "warning": "#E8A33D", "bad": "#A61C3C"}


def risk_color(level: str) -> str:
    return RISK_COLORS.get(str(level).lower(), "#9E9E9E")


def risk_badge(level: str) -> str:
    level = str(level).lower()
    return f"{RISK_EMOJI.get(level, '⚪')} {level.upper()}"


def quality_color(score: float) -> str:
    if score >= 0.75:
        return QUALITY_COLORS["good"]
    if score >= 0.5:
        return QUALITY_COLORS["warning"]
    return QUALITY_COLORS["bad"]


def risk_rank(level: str) -> int:
    level = str(level).lower()
    return RISK_ORDER.index(level) if level in RISK_ORDER else -1


LOW_CONFIDENCE_BANNER = (
    "**LOW DATA CONFIDENCE** — verify with field reports before committing resources. "
    "The inputs behind this forecast were incomplete or estimated."
)

SYNTHETIC_BANNER = (
    "**Demonstration data.** One or more sources fell back to synthetic climatology "
    "because no credentials are configured. Figures on this page are illustrative, "
    "not operational. See the Data Quality page."
)
