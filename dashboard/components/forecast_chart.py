"""Time-series forecast chart with confidence bands and alert thresholds.

The threshold lines matter as much as the forecast line: an official reads this
chart to answer "are we about to cross into HIGH", and a chart without the
thresholds drawn makes them do that arithmetic in their head.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from dashboard.theme import RISK_COLORS


def forecast_chart(
    history: Optional[pd.DataFrame] = None,
    forecast: Optional[pd.DataFrame] = None,
    thresholds: Optional[Dict[str, float]] = None,
    population: Optional[float] = None,
    title: str = "Forecast",
    y_label: str = "cases per week",
):
    """Observed history plus forecast with a 95% band. Returns a plotly figure."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    figure = go.Figure()

    if history is not None and not history.empty:
        figure.add_trace(go.Scatter(
            x=history["week"], y=history["cases"],
            name="observed", mode="lines+markers",
            line={"color": "#37474F", "width": 2}, marker={"size": 4},
        ))

    if forecast is not None and not forecast.empty:
        forecast = forecast.sort_values("target_week")
        # Band first so the point estimate draws on top of it.
        if {"confidence_interval_lower", "confidence_interval_upper"} <= set(forecast.columns):
            figure.add_trace(go.Scatter(
                x=list(forecast["target_week"]) + list(forecast["target_week"])[::-1],
                y=list(forecast["confidence_interval_upper"])
                  + list(forecast["confidence_interval_lower"])[::-1],
                fill="toself", fillcolor="rgba(21,101,192,0.16)",
                line={"color": "rgba(255,255,255,0)"},
                name="95% interval", hoverinfo="skip",
            ))
        figure.add_trace(go.Scatter(
            x=forecast["target_week"], y=forecast["predicted_cases"],
            name="forecast", mode="lines+markers",
            line={"color": "#1565C0", "width": 2.5, "dash": "dot"}, marker={"size": 7},
        ))

    # Thresholds are configured per 1,000 population; convert to case counts so
    # they sit on the same axis as the forecast.
    if thresholds and population:
        for level, per_1000 in thresholds.items():
            if not per_1000 or level == "low":
                continue
            figure.add_hline(
                y=per_1000 * population / 1000.0,
                line={"color": RISK_COLORS.get(level, "#999"), "width": 1.4, "dash": "dash"},
                annotation_text=f"{level} ({per_1000}/1,000)",
                annotation_position="right",
                annotation_font_size=10,
            )

    figure.update_layout(
        title=title, height=430, hovermode="x unified",
        xaxis_title="epidemiological week", yaxis_title=y_label,
        margin={"r": 20, "t": 50, "l": 50, "b": 40},
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def multi_disease_chart(frames: Dict[str, pd.DataFrame], title: str = "Risk by disease"):
    """Overlay normalised risk scores for several diseases in one district."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    figure = go.Figure()
    for disease, frame in frames.items():
        if frame is None or frame.empty:
            continue
        ordered = frame.sort_values("target_week")
        figure.add_trace(go.Scatter(
            x=ordered["target_week"], y=ordered["risk_score"],
            name=disease, mode="lines+markers",
        ))
    figure.update_layout(
        title=title, height=400, hovermode="x unified",
        xaxis_title="epidemiological week",
        yaxis_title="risk score (0-1, comparable across diseases)",
        yaxis_range=[0, 1.02],
        margin={"r": 20, "t": 50, "l": 50, "b": 40},
    )
    return figure


def accuracy_chart(frame: pd.DataFrame, title: str = "Predicted vs actual"):
    """Model-performance scatter with the ideal 1:1 line drawn in."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    if frame.empty:
        return None

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=frame["actual"], y=frame["predicted"], mode="markers",
        marker={"size": 6, "opacity": 0.6, "color": "#1565C0"}, name="forecasts",
    ))
    limit = float(max(frame["actual"].max(), frame["predicted"].max()))
    figure.add_trace(go.Scatter(
        x=[0, limit], y=[0, limit], mode="lines", name="perfect forecast",
        line={"color": "#9E9E9E", "dash": "dash", "width": 1.4},
    ))
    figure.update_layout(
        title=title, height=430, xaxis_title="actual cases", yaxis_title="predicted cases",
        margin={"r": 20, "t": 50, "l": 50, "b": 40},
    )
    return figure
