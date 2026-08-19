"""Multi-disease overlay for one district.

Risk *scores* are comparable across diseases even though case counts are not —
0.8 means the same thing for cholera and for malaria. That is what makes a
single prioritisation view possible for a district team with one budget.
"""

from __future__ import annotations

import pandas as pd

from dashboard import data_access as da
from dashboard.components.forecast_chart import multi_disease_chart
from dashboard.theme import RISK_ORDER, risk_badge, risk_rank


def render(st) -> None:
    st.header("Disease comparison")
    st.caption(
        "Case counts are not comparable between diseases; risk scores are. "
        "A 0.8 for cholera and a 0.8 for malaria both mean 'near the critical "
        "threshold for this disease'."
    )

    districts = sorted(da.district_frame()["name"])
    district = st.selectbox("District", districts, key="compare_district")
    diseases = da.diseases()

    frames = {}
    rows = []
    for disease in diseases:
        frame = da.predictions(disease=disease, district=district, limit=30)
        if frame.empty:
            continue
        frames[disease] = frame
        latest = frame.sort_values("target_week").iloc[-1]
        rows.append({
            "disease": disease,
            "risk": latest["risk_level"],
            "risk_score": round(float(latest["risk_score"]), 3),
            "forecast_cases": round(float(latest["predicted_cases"])),
            "target_week": latest["target_week"],
            "importation": round(float(latest.get("importation_risk", 0)), 3),
            "actions": len(latest.get("recommendations") or []),
        })

    if not rows:
        st.warning(
            f"No cached forecasts for **{district}**.\n\n"
            "```bash\npython scripts/seed_historical_data.py\n```"
        )
        return

    table = pd.DataFrame(rows)
    table["_rank"] = table["risk"].map(risk_rank)
    table = table.sort_values(["_rank", "risk_score"], ascending=False).drop(columns="_rank")

    st.subheader(f"Current standing in {district}")
    st.dataframe(table.reset_index(drop=True), use_container_width=True, hide_index=True)

    worst = table.iloc[0]
    if worst["risk"] in ("high", "critical"):
        st.error(
            f"**Top priority: {worst['disease']}** at {risk_badge(worst['risk'])} "
            f"({worst['forecast_cases']:,} cases forecast for {worst['target_week']}). "
            "Open District detail for the response package."
        )
    else:
        st.success(f"No disease is above MEDIUM in {district} this week.")

    figure = multi_disease_chart(frames, title=f"Risk trajectory, {district}")
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Where each disease is worst, nationally")
    national_rows = []
    for disease in diseases:
        frame = da.latest_per_district(disease)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_rank"] = frame["risk_level"].map(risk_rank)
        top = frame.sort_values(["_rank", "risk_score"], ascending=False).head(3)
        national_rows.append({
            "disease": disease,
            "districts_at_high_or_above": int((frame["_rank"] >= 2).sum()),
            "worst_districts": ", ".join(top["district"]),
            "peak_risk_score": round(float(frame["risk_score"].max()), 3),
        })
    if national_rows:
        st.dataframe(pd.DataFrame(national_rows), use_container_width=True, hide_index=True)

    with st.expander("Why the diseases behave differently"):
        details = da.disease_details()
        if not details.empty:
            columns = [c for c in ("slug", "transmission_mode", "horizon_weeks",
                                   "spatial_enabled", "proxies", "sources") if c in details]
            view = details[columns].copy()
            for column in ("proxies", "sources"):
                if column in view:
                    view[column] = view[column].apply(
                        lambda v: ", ".join(v) if isinstance(v, list) else v
                    )
            st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption(
            "Forecast horizons differ because the underlying biology does: malaria's "
            "climate signal gives 8 weeks of lead time, acute respiratory infection "
            "only 4."
        )
