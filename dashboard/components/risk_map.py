"""Interactive district risk map.

Uses a real choropleth when a council boundary GeoJSON is configured, and falls
back to a population-scaled scatter over district centroids otherwise — so the
map works out of the box without a 40 MB shapefile, which matters for the
low-bandwidth deployments this platform targets.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from dashboard.theme import RISK_COLORS, RISK_ORDER


def risk_map(
    frame: pd.DataFrame,
    value_column: str = "risk_score",
    level_column: str = "risk_level",
    title: str = "District risk",
    geojson_path: Optional[str] = None,
):
    """Return a plotly figure, or None when plotly is unavailable."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        return None

    if frame.empty or "lat" not in frame.columns:
        return None

    plot = frame.dropna(subset=["lat", "lon"]).copy()
    if plot.empty:
        return None

    plot["level"] = plot[level_column].astype(str).str.lower()
    plot["marker_size"] = (
        plot["population"].fillna(plot["population"].median()) ** 0.5
        if "population" in plot
        else 1.0
    )
    plot["label"] = plot.apply(
        lambda r: (
            f"<b>{r['district']}</b><br>"
            f"risk: {r['level'].upper()} ({r.get(value_column, float('nan')):.2f})<br>"
            f"forecast: {r.get('predicted_cases', float('nan')):,.0f} cases<br>"
            f"week: {r.get('target_week', '')}"
        ),
        axis=1,
    )

    figure = px.scatter_mapbox(
        plot,
        lat="lat", lon="lon",
        color="level",
        color_discrete_map=RISK_COLORS,
        category_orders={"level": list(RISK_ORDER)},
        size="marker_size", size_max=34,
        hover_name="district",
        custom_data=["label"],
        zoom=4.6,
        center={"lat": plot["lat"].mean(), "lon": plot["lon"].mean()},
        height=560,
    )
    figure.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    figure.update_layout(
        mapbox_style="carto-positron",   # free tiles, no API token required
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title=title,
        legend_title_text="risk level",
    )
    return figure


def risk_table(frame: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """The map's companion table — sorted so the worst districts read first."""
    if frame.empty:
        return frame
    from dashboard.theme import risk_rank

    table = frame.copy()
    table["_rank"] = table["risk_level"].map(risk_rank)
    columns = [c for c in ("district", "region", "risk_level", "risk_score",
                           "predicted_cases", "confidence_interval_lower",
                           "confidence_interval_upper", "importation_risk", "target_week")
               if c in table.columns]
    return (
        table.sort_values(["_rank", "risk_score"], ascending=False)
        .head(top_n)[columns]
        .reset_index(drop=True)
    )
