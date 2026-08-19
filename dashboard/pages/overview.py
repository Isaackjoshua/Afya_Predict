"""National risk overview — the page a duty officer opens first.

Answers three questions in the first screen: where is risk highest, what changed
since last week, and is there anything I should act on today.
"""

from __future__ import annotations

import pandas as pd

from dashboard import data_access as da
from dashboard.components.risk_map import risk_map, risk_table
from dashboard.theme import RISK_ORDER, risk_badge, risk_color


def render(st) -> None:
    st.header("National risk overview")

    diseases = da.diseases()
    if not diseases:
        st.error("No disease modules registered.")
        return

    left, right = st.columns([3, 1])
    with left:
        disease = st.selectbox("Disease", diseases, key="overview_disease")
    with right:
        top_n = st.number_input("Districts listed", 5, 60, 15, step=5)

    frame = da.latest_per_district(disease)
    if frame.empty:
        st.warning(
            f"No cached forecasts for **{disease}**.\n\n"
            "Populate this node with:\n"
            "```bash\npython scripts/seed_historical_data.py --disease "
            f"{disease}\n```"
        )
        return

    counts = frame["risk_level"].value_counts().to_dict()
    columns = st.columns(len(RISK_ORDER) + 1)
    for column, level in zip(columns, RISK_ORDER):
        column.metric(f"{risk_badge(level)}", counts.get(level, 0))
    columns[-1].metric("Districts", len(frame))

    actionable = frame[frame["risk_level"].isin(("high", "critical"))]
    if len(actionable):
        names = ", ".join(actionable.sort_values("risk_score", ascending=False)["district"].head(6))
        st.error(
            f"**{len(actionable)} district(s) at HIGH or CRITICAL risk:** {names}. "
            "Open District detail for the drivers and the recommended response."
        )
    else:
        st.success("No district is currently forecast above the MEDIUM threshold.")

    target_weeks = sorted(frame["target_week"].dropna().unique())
    if target_weeks:
        st.caption(
            f"Forecast target week: **{target_weeks[-1]}**"
            + (f" (range {target_weeks[0]} – {target_weeks[-1]})" if len(target_weeks) > 1 else "")
        )

    figure = risk_map(frame, title=f"{disease.capitalize()} risk by district")
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("Install plotly for the interactive map: `pip install plotly`")

    st.subheader("Highest-risk districts")
    table = risk_table(frame, top_n=int(top_n))
    st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Importation risk — where spread is arriving from"):
        imported = frame[frame.get("importation_risk", pd.Series(dtype=float)).fillna(0) > 0.2]
        if imported.empty:
            st.write("No district is currently dominated by imported risk.")
        else:
            st.write(
                "These districts' risk is driven substantially by travel from other "
                "districts rather than by local conditions — a purely temporal model "
                "would miss them entirely."
            )
            st.dataframe(
                imported[["district", "region", "risk_level", "importation_risk"]]
                .sort_values("importation_risk", ascending=False)
                .reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

    with st.expander("Regional summary"):
        if "region" in frame.columns:
            summary = (
                frame.groupby("region")
                .agg(districts=("district", "count"),
                     mean_risk=("risk_score", "mean"),
                     max_risk=("risk_score", "max"),
                     total_forecast_cases=("predicted_cases", "sum"))
                .sort_values("max_risk", ascending=False)
                .round(3)
                .reset_index()
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
