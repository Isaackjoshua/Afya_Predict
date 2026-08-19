"""Per-district forecast, explanation and response package.

This is the page that has to earn an official's trust: it shows the forecast,
then immediately shows *why*, in mechanism terms, and then what to do about it.
"""

from __future__ import annotations

import pandas as pd

from dashboard import data_access as da
from dashboard.components.forecast_chart import forecast_chart
from dashboard.components.recommendation_card import render_recommendations
from dashboard.components.shap_waterfall import (
    driver_table, shap_waterfall, source_contribution_chart,
)
from dashboard.theme import LOW_CONFIDENCE_BANNER, risk_badge, risk_color


def render(st) -> None:
    st.header("District detail")

    diseases = da.diseases()
    districts = sorted(da.district_frame()["name"])

    col1, col2 = st.columns(2)
    disease = col1.selectbox("Disease", diseases, key="detail_disease")
    district = col2.selectbox("District", districts, key="detail_district")

    frame = da.predictions(disease=disease, district=district, limit=60)
    if frame.empty:
        st.warning(
            f"No cached forecast for **{disease}** in **{district}**.\n\n"
            "```bash\npython scripts/seed_historical_data.py "
            f"--disease {disease} --districts \"{district}\"\n```"
        )
        return

    frame = frame.sort_values("target_week")
    latest = frame.iloc[-1].to_dict()

    st.markdown(
        f"<div style='padding:0.9rem 1.1rem;border-left:6px solid "
        f"{risk_color(latest['risk_level'])};background:rgba(0,0,0,0.03);border-radius:4px'>"
        f"<b style='font-size:1.15rem'>{risk_badge(latest['risk_level'])} &nbsp; "
        f"{latest['district']} &mdash; week {latest['target_week']}</b></div>",
        unsafe_allow_html=True,
    )

    metrics = st.columns(4)
    metrics[0].metric("Forecast cases", f"{latest['predicted_cases']:,.0f}")
    metrics[1].metric(
        "95% interval",
        f"{latest['confidence_interval_lower']:,.0f}–{latest['confidence_interval_upper']:,.0f}",
    )
    metrics[2].metric("Risk score", f"{latest['risk_score']:.2f}")
    metrics[3].metric("Importation", f"{latest.get('importation_risk', 0):.0%}")

    if latest.get("data_quality_flags"):
        flags = latest["data_quality_flags"]
        if any("LOW DATA CONFIDENCE" in str(f) for f in flags):
            st.warning(LOW_CONFIDENCE_BANNER)

    # --- forecast chart ---------------------------------------------------
    history = da.observations(disease, district=district, limit=200)
    thresholds = _thresholds(disease)
    population = _population(district)
    figure = forecast_chart(
        history=history if not history.empty else None,
        forecast=frame, thresholds=thresholds, population=population,
        title=f"{disease.capitalize()} in {district}",
    )
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)

    # --- explanation ------------------------------------------------------
    st.subheader("Why this forecast")
    st.info(latest.get("natural_language_explanation", "No explanation recorded."))
    if latest.get("counterfactual"):
        st.markdown(f"**What would change it:** {latest['counterfactual']}")

    drivers = latest.get("top_drivers") or []
    waterfall = shap_waterfall(drivers, predicted_cases=latest["predicted_cases"])
    if waterfall is not None:
        st.plotly_chart(waterfall, use_container_width=True)
    table = driver_table(drivers)
    if not table.empty:
        with st.expander("Driver detail"):
            st.dataframe(table, use_container_width=True, hide_index=True)

    detail = da.prediction_detail(latest["prediction_id"]) or {}
    contributions = _source_contributions(detail)
    if contributions:
        with st.expander("Contribution by data source — is the fusion working?"):
            st.caption(
                "If a single source dominates, the platform has recreated the "
                "single-source dependence that broke Google Flu Trends."
            )
            chart = source_contribution_chart(contributions)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            st.dataframe(pd.DataFrame(contributions), use_container_width=True, hide_index=True)

    # --- spatial ----------------------------------------------------------
    sources = latest.get("source_districts") or []
    if sources:
        st.subheader("Where the risk is arriving from")
        st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)

    # --- action -----------------------------------------------------------
    st.subheader("Recommended response")
    render_recommendations(st, latest.get("recommendations") or [])

    with st.expander("Metadata"):
        st.json({
            "prediction_id": latest.get("prediction_id"),
            "model_version": latest.get("model_version"),
            "forecast_date": str(latest.get("forecast_date")),
            "data_quality_flags": latest.get("data_quality_flags"),
            "data_freshness": latest.get("data_freshness"),
        })


def _thresholds(disease: str) -> dict:
    try:
        from src.core.config_loader import load_disease_config

        return load_disease_config(disease).alerts.model_dump()
    except Exception:  # noqa: BLE001
        return {}


def _population(district: str) -> float:
    try:
        return float(da.region().population_of(district))
    except Exception:  # noqa: BLE001
        return 0.0


def _source_contributions(detail: dict) -> list:
    shap_values = (detail or {}).get("shap_values") or {}
    if not shap_values:
        return []
    try:
        from src.models.registry import build_module

        module = build_module(detail.get("disease", "").lower().replace(" ", "_"))
        provenance = module.feature_matrix.provenance if module.feature_matrix else {}
    except Exception:  # noqa: BLE001
        provenance = {}

    buckets: dict = {}
    for feature, value in shap_values.items():
        source = provenance.get(feature, {}).get("source", "unknown")
        buckets[source] = buckets.get(source, 0.0) + abs(float(value))
    total = sum(buckets.values()) or 1.0
    return sorted(
        ({"source": s, "abs_contribution": round(v, 4), "share": round(v / total, 4)}
         for s, v in buckets.items()),
        key=lambda b: b["share"], reverse=True,
    )
