"""SHAP waterfall — the visual form of "why did the model say this?".

Two design decisions worth keeping:

* bars are labelled with the **proxy and its fitted lag** ("rainfall, 6 weeks
  ago"), not the engineered column name ("rainfall_mm_lag6"). The reader is an
  epidemiologist, not the person who wrote the feature pipeline.
* the mechanism appears in the hover text, so the chart carries its own causal
  explanation rather than requiring a lookup elsewhere.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from dashboard.theme import DIRECTION_COLORS


def _label(driver: dict) -> str:
    proxy = (driver.get("proxy") or driver.get("feature") or "").replace("_", " ")
    lag = driver.get("lag_weeks") or 0
    if lag:
        return f"{proxy}, {lag}w ago"
    return proxy or str(driver.get("feature", ""))


def shap_waterfall(drivers: List[dict], predicted_cases: Optional[float] = None,
                   title: str = "What drove this forecast"):
    """Horizontal contribution chart. Returns a plotly figure."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    if not drivers:
        return None

    frame = pd.DataFrame(drivers)
    contribution_column = "shap_value" if "shap_value" in frame else "contribution"
    if contribution_column not in frame:
        return None

    frame = frame.reindex(frame[contribution_column].abs().sort_values().index)
    frame["label"] = [_label(row) for row in frame.to_dict("records")]
    frame["color"] = frame.get("direction", "neutral").map(DIRECTION_COLORS).fillna("#9E9E9E")
    frame["hover"] = frame.apply(
        lambda r: (
            f"<b>{r['label']}</b><br>"
            f"contribution: {r[contribution_column]:+.3f}<br>"
            + (f"share: {r['contribution_share']:.0%}<br>" if "contribution_share" in r else "")
            + (f"measured: {r['value']:.2f}<br>" if r.get("value") is not None else "")
            + (f"<i>{r.get('mechanism', '')}</i>" if r.get("mechanism") else "")
        ),
        axis=1,
    )

    figure = go.Figure(go.Bar(
        x=frame[contribution_column], y=frame["label"], orientation="h",
        marker_color=frame["color"], customdata=frame[["hover"]],
        hovertemplate="%{customdata[0]}<extra></extra>",
    ))
    figure.add_vline(x=0, line={"color": "#455A64", "width": 1.2})
    subtitle = (
        f"  (forecast: {predicted_cases:,.0f} cases)" if predicted_cases is not None else ""
    )
    figure.update_layout(
        title=f"{title}{subtitle}",
        height=max(320, 34 * len(frame) + 120),
        xaxis_title="contribution to the forecast (SHAP value)",
        yaxis_title="",
        margin={"r": 20, "t": 60, "l": 10, "b": 40},
        showlegend=False,
    )
    return figure


def source_contribution_chart(contributions: List[dict], title: str = "Contribution by data source"):
    """Which *sources* carried the forecast — the visual fusion check.

    If one bar dominates every disease, the platform has quietly recreated the
    single-source dependence that broke Google Flu Trends.
    """
    try:
        import plotly.express as px
    except ImportError:
        return None
    if not contributions:
        return None

    frame = pd.DataFrame(contributions).sort_values("share", ascending=True)
    figure = px.bar(
        frame, x="share", y="source", orientation="h",
        labels={"share": "share of total attribution", "source": ""},
        height=max(260, 36 * len(frame) + 100),
    )
    figure.update_layout(title=title, margin={"r": 20, "t": 50, "l": 10, "b": 40},
                         xaxis_tickformat=".0%")
    return figure


def driver_table(drivers: List[dict]) -> pd.DataFrame:
    """Tabular fallback when plotly is unavailable, and the accessible view."""
    if not drivers:
        return pd.DataFrame()
    frame = pd.DataFrame(drivers)
    frame["driver"] = [_label(row) for row in frame.to_dict("records")]
    columns = [c for c in ("driver", "value", "shap_value", "contribution_share",
                           "direction", "mechanism") if c in frame.columns]
    out = frame[columns].copy()
    if "contribution_share" in out:
        out["contribution_share"] = out["contribution_share"].map(lambda v: f"{v:.0%}")
    return out.reset_index(drop=True)
