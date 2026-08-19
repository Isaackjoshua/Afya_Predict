"""Recommendation cards — the "so what do we do?" half of every alert.

Each card states the action, the owner, the deadline and the quantity, because
an alert that omits any of those pushes the hard part back onto the reader
(shortcoming #12).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from dashboard.theme import risk_badge, risk_color


def recommendation_card(recommendation: dict, index: Optional[int] = None) -> str:
    """Render one recommendation as markdown."""
    priority = str(recommendation.get("priority", "medium")).lower()
    number = f"{index}. " if index is not None else ""
    lines = [
        f"**{number}{recommendation.get('action', '')}**",
        "",
        f"- **Owner:** {recommendation.get('responsible', 'unassigned')}",
        f"- **Within:** {recommendation.get('timeframe_days', '?')} days",
    ]
    if recommendation.get("quantity"):
        lines.append(f"- **Quantity:** {recommendation['quantity']}")
    if recommendation.get("rationale"):
        lines.append(f"- **Why:** {recommendation['rationale']}")
    lines.append(f"- **Priority:** {risk_badge(priority)}")
    return "\n".join(lines)


def render_recommendations(st, recommendations: List[dict], expanded_first: int = 2) -> None:
    """Render a list of recommendations into a Streamlit container."""
    if not recommendations:
        st.info(
            "No response actions configured for this risk level. Add them under "
            "`recommendations:` in the disease YAML — they are configuration, not code."
        )
        return
    for i, recommendation in enumerate(recommendations, 1):
        header = recommendation.get("action", "")
        header = header if len(header) <= 90 else header[:87] + "..."
        with st.expander(f"{i}. {header}", expanded=i <= expanded_first):
            st.markdown(recommendation_card(recommendation))


def alert_header(st, alert: dict) -> None:
    """Consistent alert header used on every page that shows one."""
    level = str(alert.get("risk_level", "low")).lower()
    st.markdown(
        f"<div style='padding:0.8rem 1rem;border-left:6px solid {risk_color(level)};"
        f"background:rgba(0,0,0,0.03);border-radius:4px'>"
        f"<b style='font-size:1.05rem'>{risk_badge(level)} &nbsp; {alert.get('disease','')} "
        f"&mdash; {alert.get('district','')}</b><br>"
        f"<span style='opacity:0.85'>week {alert.get('target_week','')} &nbsp;|&nbsp; "
        f"{alert.get('predicted_cases', 0):,.0f} cases forecast "
        f"({alert.get('predicted_incidence_per_1000', 0):.2f} per 1,000) &nbsp;|&nbsp; "
        f"{alert.get('lead_time_weeks', 0)} weeks of lead time</span></div>",
        unsafe_allow_html=True,
    )
    if alert.get("low_data_confidence"):
        from dashboard.theme import LOW_CONFIDENCE_BANNER

        st.warning(LOW_CONFIDENCE_BANNER)


def summary_metrics(counts: Dict[str, int]) -> List[tuple]:
    """(label, value, colour) triples for a risk-level metric row."""
    from dashboard.theme import RISK_ORDER

    return [
        (level.capitalize(), counts.get(level, 0), risk_color(level)) for level in RISK_ORDER
    ]
