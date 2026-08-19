"""Log responses and track their effect (shortcoming #15).

The loop this page closes: an alert recommended an action, somebody did it, and
the platform now needs to know *what was done, when, and with what coverage* —
otherwise the effect can never be estimated and the model never learns from
its own operational consequences.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from dashboard import data_access as da
from dashboard.theme import risk_badge


def render(st) -> None:
    st.header("Intervention tracker")

    tab_log, tab_history, tab_impact, tab_audit = st.tabs(
        ["Log a response", "History", "Estimated impact", "Response audit"]
    )

    with tab_log:
        _log_form(st)
    with tab_history:
        _history(st)
    with tab_impact:
        _impact(st)
    with tab_audit:
        _audit(st)


def _log_form(st) -> None:
    st.caption(
        "Coverage and timing are captured, not just the fact that something happened: "
        "5,000 nets reaching 12% of a district three weeks late is a different exposure "
        "from 50,000 reaching 80% on time, and the two must not be scored the same."
    )

    types = da.intervention_types()
    type_options = list(types["type"]) if not types.empty else ["other"]
    labels = dict(zip(types.get("type", []), types.get("label", []))) if not types.empty else {}

    open_alerts = da.alerts(days=120, limit=200)
    alert_options = {"(not linked to an alert)": None}
    if not open_alerts.empty:
        for _, row in open_alerts.head(50).iterrows():
            key = (f"{risk_badge(row['risk_level'])} {row['disease']} — {row['district']} "
                   f"(week {row['target_week']})")
            alert_options[key] = row["alert_id"]

    with st.form("log_intervention"):
        col1, col2 = st.columns(2)
        disease = col1.selectbox("Disease", da.diseases())
        district = col2.selectbox("District", sorted(da.district_frame()["name"]))

        col3, col4 = st.columns(2)
        intervention_type = col3.selectbox(
            "Response type", type_options,
            format_func=lambda t: labels.get(t, t.replace("_", " ")),
        )
        alert_label = col4.selectbox("Triggered by", list(alert_options))

        col5, col6, col7 = st.columns(3)
        started_week = col5.text_input("Started (ISO week)", _current_week())
        coverage = col6.slider("Population coverage", 0.0, 1.0, 0.5, 0.05)
        quantity = col7.number_input("Quantity delivered", min_value=0.0, value=0.0, step=100.0)

        col8, col9 = st.columns(2)
        unit = col8.text_input("Unit", "units")
        logged_by = col9.text_input("Logged by", "dashboard")
        notes = st.text_area("Notes", "")

        if st.form_submit_button("Log response"):
            payload = {
                "disease": disease, "district": district,
                "intervention_type": intervention_type,
                "started_week": started_week.strip() or _current_week(),
                "coverage": float(coverage),
                "quantity": float(quantity) or None,
                "unit": unit or None,
                "alert_id": alert_options[alert_label],
                "notes": notes, "logged_by": logged_by,
            }
            result = da.log_intervention(payload)
            if result:
                st.success(f"Logged. Intervention id: `{result['intervention_id']}`")
                st.caption(
                    "This record now feeds three loops: response-time auditing, "
                    "impact estimation, and down-weighting the affected weeks so the "
                    "model is not penalised for an outbreak this response helped avert."
                )
            else:
                st.error("Could not log the response.")

    if not types.empty:
        with st.expander("Response types and their effect lags"):
            st.caption(
                "The effect lag places the evaluation window. Scoring a bed-net "
                "campaign on the week it started would measure nothing."
            )
            st.dataframe(types, use_container_width=True, hide_index=True)


def _history(st) -> None:
    col1, col2 = st.columns(2)
    disease = col1.selectbox("Disease", ["(all)"] + da.diseases(), key="hist_disease")
    district = col2.selectbox(
        "District", ["(all)"] + sorted(da.district_frame()["name"]), key="hist_district"
    )
    frame = da.interventions(
        disease=None if disease == "(all)" else disease,
        district=None if district == "(all)" else district,
    )
    if frame.empty:
        st.info("No responses logged yet.")
        return

    columns = [c for c in ("started_week", "disease", "district", "intervention_type",
                           "coverage", "quantity", "unit", "logged_by", "alert_id", "notes")
               if c in frame.columns]
    st.dataframe(frame[columns].sort_values("started_week", ascending=False)
                 .reset_index(drop=True), use_container_width=True, hide_index=True)

    st.subheader("Coverage by response type")
    if "intervention_type" in frame:
        summary = (
            frame.groupby("intervention_type")
            .agg(responses=("intervention_id", "count"),
                 mean_coverage=("coverage", "mean"))
            .round(3).sort_values("responses", ascending=False).reset_index()
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


def _impact(st) -> None:
    st.caption(
        "Three counterfactuals, reported together. Malaria falls after a bed-net "
        "campaign — but it also falls every year at the end of the rains. Where the "
        "estimates disagree, this page says so rather than picking a favourite."
    )
    frame = da.interventions()
    if frame.empty:
        st.info("No responses logged yet.")
        return

    options = {
        f"{row['started_week']} — {row['intervention_type']} in {row['district']} "
        f"({row['disease']})": row["intervention_id"]
        for _, row in frame.iterrows()
    }
    choice = st.selectbox("Response", list(options), key="impact_choice")
    if not st.button("Estimate impact", key="impact_button"):
        return

    with st.spinner("Comparing observed outcomes against the counterfactuals…"):
        estimate = _estimate(options[choice])

    if estimate is None:
        st.error("Impact could not be estimated.")
        return
    if estimate.get("error"):
        st.warning(estimate["error"])
        return

    st.info(estimate.get("narrative", ""))
    estimates = estimate.get("estimates", {})
    cols = st.columns(3)
    for column, (name, value) in zip(cols, estimates.items()):
        column.metric(name.replace("_", " "), "n/a" if value is None else f"{value:+,.0f}")

    st.metric("Confidence", estimate.get("confidence", "unknown"))
    st.caption(
        "Confidence is capped at *moderate*: this is observational data, not a trial. "
        "No combination of signals promotes it to 'high'."
    )
    for caveat in estimate.get("caveats", []):
        st.write(f"- {caveat}")


def _audit(st) -> None:
    st.caption(
        "The operational question a forecasting system must keep asking of itself: "
        "did the warning reach someone who acted, and did it arrive early enough to "
        "matter?"
    )
    days = st.slider("Look-back window (days)", 30, 365, 90, key="audit_days")
    audit = da.response_audit(days=days)
    if not audit:
        st.info("No alerts in this window.")
        return

    cols = st.columns(4)
    cols[0].metric("Alerts issued", audit.get("total_alerts", 0))
    cols[1].metric("With a logged response", audit.get("alerts_with_response", 0))
    cols[2].metric("Response rate", f"{audit.get('response_rate', 0):.0%}")
    effective = audit.get("effective_lead_time_weeks")
    cols[3].metric(
        "Effective lead time",
        "n/a" if effective is None else f"{effective:.1f} wk",
        help="Forecast lead time minus the average response delay. Negative means the "
             "early warning is being consumed by response latency.",
    )

    st.info(audit.get("interpretation", ""))
    by_level = audit.get("by_level") or {}
    if by_level:
        st.dataframe(
            pd.DataFrame([{"risk_level": k, **v} for k, v in by_level.items()]),
            use_container_width=True, hide_index=True,
        )


def _current_week() -> str:
    from src.core.timeutils import to_epi_week

    return to_epi_week(date.today())


def _estimate(intervention_id: str):
    import os

    if da.API_URL:
        try:
            import requests

            response = requests.get(
                f"{da.API_URL}/interventions/{intervention_id}/impact", timeout=60
            )
            if response.ok:
                return response.json()
            return {"error": response.json().get("detail", response.text)}
        except Exception:  # noqa: BLE001
            pass
    try:
        from src.intervention_tracking.impact_estimator import ImpactEstimator

        cache = da._cache()
        matches = [i for i in cache.get_interventions(limit=2000)
                   if i.intervention_id == intervention_id]
        if not matches:
            return {"error": "Unknown intervention."}
        intervention = matches[0]

        observations = cache.get_observations(intervention.disease, limit=5000)
        if not observations:
            return {"error": (
                "No observed case history cached for this disease, so no counterfactual "
                "can be built. Run scripts/seed_historical_data.py first."
            )}
        frame = pd.DataFrame(observations)
        wide = frame.pivot_table(index="week", columns="district",
                                 values="cases", aggfunc="mean").sort_index()
        if intervention.district not in wide.columns:
            return {"error": f"No cached observations for {intervention.district}."}

        observed = wide[intervention.district]
        controls = wide.drop(columns=[intervention.district])
        forecasts = cache.latest_predictions(
            disease=intervention.disease, district=intervention.district, limit=200
        )
        forecast_series = (
            pd.Series({p.target_week: p.predicted_cases for p in forecasts}).sort_index()
            if forecasts else None
        )
        estimate = ImpactEstimator(region=da.region()).estimate(
            intervention, observed, forecast=forecast_series, control_series=controls
        )
        return estimate.to_dict()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
