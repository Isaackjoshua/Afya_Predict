"""AFYA-PREDICT dashboard.

    streamlit run dashboard/app.py

Set `API_URL` to read through a central API; without it the dashboard reads the
local SQLite cache directly, which is what lets a district office keep working
through a connectivity outage (shortcoming #14).

Design intent worth preserving when editing:

* **Every risk figure is one click from its explanation.** The dashboard exists
  because officials would not act on BlueDot-style black-box scores. A page that
  shows a risk number without a route to its drivers defeats the purpose.
* **Data provenance is never hidden.** The sidebar always states whether the page
  is reading live data or a synthetic fallback.
"""

from __future__ import annotations

import os
import pathlib
import sys

# Make the repo importable when Streamlit is launched from anywhere.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import __version__  # noqa: E402

PAGES = {
    "Overview": ("dashboard.pages.overview", "National risk heatmap and today's priorities"),
    "District detail": ("dashboard.pages.district_detail", "Forecast, drivers and response for one district"),
    "Disease comparison": ("dashboard.pages.disease_comparison", "Multi-disease risk in one district"),
    "Data quality": ("dashboard.pages.data_quality", "Source freshness, fusion status and completeness"),
    "Model performance": ("dashboard.pages.model_performance", "Out-of-sample accuracy and drift"),
    "Intervention tracker": ("dashboard.pages.intervention_tracker", "Log responses and estimate their effect"),
}


def main() -> None:
    st.set_page_config(
        page_title="AFYA-PREDICT",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    from dashboard import data_access as da

    with st.sidebar:
        st.title("AFYA-PREDICT")
        st.caption("Predictive disease intelligence · East Africa and beyond")

        choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
        st.caption(PAGES[choice][1])
        st.divider()

        # Provenance, always visible: nobody should have to guess whether the
        # numbers on screen came from live data.
        source_label = da.data_source_label()
        (st.success if "live API" in source_label else st.info)(f"Reading: {source_label}")

        try:
            status = da.cache_status()
            st.caption(
                f"{status['predictions']} forecasts · {status['alerts']} alerts · "
                f"{status['weeks_cached']} week(s) cached"
            )
            if not status["offline_ready"]:
                st.warning("Fewer than 2 weeks cached — this node is not offline-ready.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Local store unavailable: {exc}")

        st.divider()
        st.caption(f"v{__version__} · MIT licensed")
        st.caption("[Source](https://github.com/Isaackjoshua/Afya_Predict)")

    module_path = PAGES[choice][0]
    try:
        import importlib

        page = importlib.import_module(module_path)
        page.render(st)
    except Exception as exc:  # noqa: BLE001 - a broken page must not take down the app
        st.error(f"This page failed to render: {exc}")
        with st.expander("Traceback"):
            import traceback

            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
