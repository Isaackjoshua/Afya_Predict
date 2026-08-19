"""Data freshness and quality monitor (shortcoming #7).

An operator must be able to tell, before acting, whether a forecast rests on
live DHIS2 data or on a synthetic fallback. Hiding that behind a clean-looking
prediction is how a system loses trust the first time it is wrong.
"""

from __future__ import annotations

import pandas as pd

from dashboard import data_access as da
from dashboard.theme import SYNTHETIC_BANNER, quality_color


def render(st) -> None:
    st.header("Data quality and freshness")

    frame = da.data_status()
    if frame.empty:
        st.warning("No ingestion has run on this node yet. Run `python -m src.data_ingestion.scheduler --all`.")
        return

    in_use = frame[frame.get("in_use", True).astype(bool)] if "in_use" in frame else frame
    live = in_use[in_use.get("mode") == "live"] if "mode" in in_use else in_use.iloc[0:0]
    synthetic = in_use[in_use.get("mode") == "synthetic"] if "mode" in in_use else in_use.iloc[0:0]
    stale = in_use[in_use.get("stale", False).astype(bool)] if "stale" in in_use else in_use.iloc[0:0]

    metrics = st.columns(4)
    metrics[0].metric("Sources in use", len(in_use))
    metrics[1].metric("Live", len(live))
    metrics[2].metric("Synthetic", len(synthetic))
    metrics[3].metric("Stale", len(stale))

    # Critical rule #1: never fewer than three fused sources.
    if len(live) >= 3:
        st.success(f"{len(live)} source(s) returning live data — the fusion rule is met.")
    else:
        st.error(
            f"Only {len(live)} source(s) are returning live data. Critical rule #1 requires "
            "at least 3. Forecasts remain available, with widened confidence intervals and "
            "reduced confidence."
        )
    if len(synthetic):
        st.warning(SYNTHETIC_BANNER)

    for warning in da.data_status_warnings():
        st.info(warning)

    st.subheader("Per-source status")
    columns = [c for c in ("source", "in_use", "configured", "optional", "mode",
                           "update_frequency_days", "age_hours", "stale", "rows",
                           "mean_quality", "latest_data_date", "error")
               if c in frame.columns]
    st.dataframe(frame[columns], use_container_width=True, hide_index=True)

    st.caption(
        "**mode** — `live` means a real upstream fetch; `cache` means replayed from disk; "
        "`synthetic` means a deterministic climatology stood in so the pipeline could run. "
        "**optional** sources (search trends) can never act as a primary predictor."
    )

    with st.expander("What each source contributes"):
        st.markdown(
            """
| Source | Provides | Why it matters |
|---|---|---|
| CHIRPS | rainfall | breeding sites (malaria), flooding and contamination (cholera) |
| ERA5 | temperature, humidity, wind | vector development, pathogen survival, droplet persistence |
| MODIS | vegetation greenness (NDVI) | resting habitat, soil moisture, standing water |
| Sentinel-5P | NO2, PM2.5 proxy | airway irritation driving respiratory presentations |
| DHIS2 | case and mortality counts | the target the models are fitted against |
| CDR mobility | origin-destination flows | where disease travels next |
| WorldPop | population, density | contact rate, and the denominator for incidence |
| WASH/JMP | water and sanitation coverage | the structural gate on waterborne transmission |
| Livestock | animal outbreak reports | One Health spillover signal |
| Google Trends | search interest | optional urban boost only — never primary (rule #14) |
"""
        )

    st.subheader("Recent input quality")
    disease = st.selectbox("Disease", da.diseases(), key="quality_disease")
    weeks = st.slider("Weeks to sample", 4, 52, 12)
    if st.button("Sample recent data quality", key="quality_sample"):
        with st.spinner("Ingesting a sample window…"):
            summary = _sample_quality(disease, weeks)
        if summary is None:
            st.error("Quality sampling failed — see the server log.")
        else:
            st.dataframe(summary["sources"], use_container_width=True, hide_index=True)
            if summary["flags"]:
                st.subheader("Quality flags raised")
                st.dataframe(pd.DataFrame(summary["flags"]), use_container_width=True,
                             hide_index=True)
            st.caption(
                "Missing surveillance weeks are kept as nulls with zero quality, never "
                "zero-filled: a district that did not report is not a district with no cases."
            )

    st.subheader("Offline readiness")
    status = da.offline_status()
    ready = status.get("ready", status.get("offline_ready", False))
    (st.success if ready else st.warning)(status.get("message", ""))
    st.json({k: v for k, v in status.items() if k not in ("message", "last_sync")})


def _sample_quality(disease: str, weeks: int):
    try:
        from datetime import date

        from src.core.config_loader import load_disease_config
        from src.core.timeutils import shift_week, to_epi_week
        from src.data_ingestion.normalizer import ingest

        end_week = to_epi_week(date.today())
        start_week = shift_week(end_week, -(weeks - 1))
        region = da.region()
        sample = region.district_names[:6]
        sources = sorted(set(load_disease_config(disease).required_sources) | {"dhis2"})
        panel = ingest(sources, start_week, end_week, region=region,
                       districts=sample, impute=False)

        values = panel.values()
        rows = []
        for source in sorted(set(panel.sources.values())):
            variables = [v for v, s in panel.sources.items() if s == source and v in values]
            if not variables:
                continue
            subset = values[variables]
            rows.append({
                "source": source,
                "mode": panel.modes.get(source),
                "variables": len(variables),
                "completeness": round(float(subset.notna().to_numpy().mean()), 4),
            })
        return {
            "sources": pd.DataFrame(rows),
            "flags": [f.model_dump() for f in panel.flags if f.severity != "info"],
        }
    except Exception:  # noqa: BLE001
        return None
