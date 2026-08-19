"""Accuracy tracking over time (shortcoming #13).

Publishing the model's own error is a deliberate choice. A forecasting system
that shows only its predictions and never its track record is asking to be
trusted rather than earning it.
"""

from __future__ import annotations

import pandas as pd

from dashboard import data_access as da
from dashboard.components.forecast_chart import accuracy_chart


def render(st) -> None:
    st.header("Model performance")
    st.caption(
        "Every figure here is out of sample: the model is refitted on history up to "
        "each fold and scored on weeks it has never seen, with a purge gap equal to "
        "the forecast horizon."
    )

    disease = st.selectbox("Disease", da.diseases(), key="perf_disease")

    st.subheader("Forecasts against what actually happened")
    scored = _score_recent(disease)
    if scored.empty:
        st.info(
            "No forecast has reached its target week yet on this node, so accuracy "
            "cannot be measured here.\n\nRun a full backtest instead:\n"
            f"```bash\npython scripts/run_backtest.py --disease {disease}\n```"
        )
    else:
        from src.evaluation.metrics import regression_metrics

        metrics = regression_metrics(scored["actual"], scored["predicted"])
        row = st.columns(4)
        row[0].metric("MAE", f"{metrics['mae']:.1f}")
        row[1].metric("RMSE", f"{metrics['rmse']:.1f}")
        row[2].metric("R²", f"{metrics['r2']:.3f}")
        row[3].metric("Bias", f"{metrics['bias']:+.1f}")
        st.caption(
            "Bias is the average signed error. A persistent negative bias means the "
            "model under-forecasts, which matters more than its magnitude when the "
            "output is used to size a supply order."
        )
        figure = accuracy_chart(scored, title=f"{disease.capitalize()}: forecast vs actual")
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)
        st.dataframe(scored.tail(30).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    st.subheader("Full walk-forward backtest")
    st.caption(
        "Refits the model once per fold, so this takes minutes rather than seconds."
    )
    col1, col2, col3 = st.columns(3)
    history_weeks = col1.number_input("History (weeks)", 104, 520, 260, step=52)
    max_folds = col2.number_input("Folds", 1, 8, 3)
    districts = col3.multiselect(
        "Districts (blank = all)", sorted(da.district_frame()["name"]),
        default=sorted(da.district_frame()["name"])[:4],
    )

    if st.button("Run backtest", key="perf_backtest"):
        with st.spinner("Running walk-forward validation…"):
            report = _backtest(disease, int(history_weeks), int(max_folds), districts or None)
        if report is None:
            st.error("Backtest failed — see the server log.")
        elif report.get("error"):
            st.error(report["error"])
        else:
            _render_report(st, report)

    st.subheader("Concept drift")
    st.caption(
        "Google Flu Trends was fitted once in 2008 and never refitted; by 2013 it "
        "was overestimating by 140% and nobody noticed automatically. This is what "
        "watches for the same failure here."
    )
    drift = _drift(disease)
    if drift is None:
        st.info("No residual history recorded yet for this disease.")
    else:
        cols = st.columns(3)
        cols[0].metric("Residuals monitored", drift.get("monitored_residuals", 0))
        cols[1].metric("Retrain due", "yes" if drift.get("should_retrain") else "no")
        cols[2].metric("Drift events", len(drift.get("drift_events") or []))
        for reason in drift.get("reasons", []):
            st.write(f"- {reason}")
        if drift.get("drift_events"):
            st.dataframe(pd.DataFrame(drift["drift_events"]),
                         use_container_width=True, hide_index=True)
        if drift.get("last_retrain"):
            st.caption(f"Last retrain: {drift['last_retrain']}")


def _render_report(st, report: dict) -> None:
    accuracy = report.get("accuracy", {})
    row = st.columns(4)
    row[0].metric("MAE", accuracy.get("mae"))
    row[1].metric("RMSE", accuracy.get("rmse"))
    row[2].metric("R²", accuracy.get("r2"))
    row[3].metric("Folds", report.get("folds"))

    verdict = report.get("passes_acceptance")
    (st.success if verdict else st.error)(
        "Meets the acceptance criteria — fit for deployment."
        if verdict else
        "Does NOT meet the acceptance criteria — not fit for deployment (critical rule #10)."
    )
    for note in report.get("acceptance_notes", []):
        st.write(f"- {note}")

    intervals = report.get("intervals", {})
    if intervals:
        st.subheader("Interval calibration")
        coverage = intervals.get("coverage")
        st.metric("95% interval coverage", f"{coverage:.1%}" if coverage else "n/a")
        st.caption(
            "A '95%' interval that covers far less is worse than no interval, because "
            "it invites false confidence in a resource decision."
        )

    baselines = report.get("baselines")
    if baselines:
        st.subheader("Against the naive baselines")
        st.write(baselines.get("verdict", ""))
        st.dataframe(
            pd.DataFrame([{"baseline": k, "skill_score": v}
                          for k, v in (baselines.get("skill") or {}).items()]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Skill score = 1 − model error / baseline error. A model that cannot beat "
            "'same week last year' has learned nothing worth deploying."
        )

    outbreak = report.get("outbreak_detection")
    if outbreak:
        st.subheader("Outbreak detection")
        st.write(outbreak.get("verdict", ""))
        st.json({k: v for k, v in outbreak.items() if k != "verdict"})


def _score_recent(disease: str) -> pd.DataFrame:
    """Join cached forecasts to observed outcomes for weeks that have arrived."""
    predictions = da.predictions(disease=disease, limit=2000)
    observations = da.observations(disease, limit=5000)
    if predictions.empty or observations.empty:
        return pd.DataFrame()
    merged = predictions.merge(
        observations[["district", "week", "cases"]],
        left_on=["district", "target_week"], right_on=["district", "week"], how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    return (
        merged.rename(columns={"cases": "actual", "predicted_cases": "predicted"})
        [["district", "target_week", "actual", "predicted",
          "confidence_interval_lower", "confidence_interval_upper", "risk_level"]]
        .sort_values("target_week")
    )


def _backtest(disease: str, history_weeks: int, max_folds: int, districts) -> dict | None:
    try:
        from src.core.geo import subset_region
        from src.data_ingestion.normalizer import ingest
        from src.evaluation.walk_forward_cv import WalkForwardCV
        from src.models.registry import build_module
        from src.core.timeutils import shift_week, to_epi_week
        from datetime import date

        region = da.region()
        if districts:
            region = subset_region(region, districts)
        module = build_module(disease, region=region)
        end_week = to_epi_week(date.today())
        start_week = shift_week(end_week, -history_weeks)
        sources = sorted(set(module.config.required_sources) | {"dhis2"})
        panel = ingest(sources, start_week, end_week, region=region)
        matrix = module.build_feature_matrix(panel)
        cv = WalkForwardCV(module, initial_train_weeks=min(156, history_weeks // 2),
                           step_weeks=26, test_weeks=26, max_folds=max_folds)
        result = cv.run(matrix)
        if not result.folds:
            return {"error": "Not enough history for a single walk-forward fold."}
        return result.report()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _drift(disease: str) -> dict | None:
    try:
        from src.models.auto_retrain import AutoRetrainer
        from src.models.registry import build_module

        module = build_module(disease, region=da.region())
        module.load()
        retrainer = AutoRetrainer(module)
        residuals = retrainer.monitored_residuals()
        decision = retrainer.should_retrain(residuals=residuals)
        return {
            "monitored_residuals": len(residuals),
            "should_retrain": decision.should_retrain,
            "reasons": decision.reasons,
            "drift_events": decision.drift_events,
            "last_retrain": retrainer.load_state().get("last_retrain"),
        }
    except Exception:  # noqa: BLE001
        return None
