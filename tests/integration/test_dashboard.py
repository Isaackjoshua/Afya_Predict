"""Dashboard smoke tests.

Streamlit is an optional extra, and a browser-driven test would be far too slow
to run on every change. Instead these tests drive each page's `render()` through
a minimal stub that records what was drawn — which is enough to catch the
failures that actually happen in practice: a renamed column, a missing key, a
component that raises when the cache is empty.

The stub deliberately does not emulate Streamlit's widget semantics. It returns
the first option for every selector, so each page renders its default view.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List

import pytest

from dashboard.components.recommendation_card import recommendation_card, summary_metrics
from dashboard.components.risk_map import risk_table
from dashboard.components.shap_waterfall import driver_table
from dashboard.theme import RISK_ORDER, risk_badge, risk_color, risk_rank


# --------------------------------------------------------------- the stub
class StubStreamlit:
    """Records draw calls so a test can assert the page produced something."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.errors: List[str] = []
        self.session_state: Dict[str, Any] = {}

    # -- selection widgets: always take the first option -------------------
    def selectbox(self, label, options, index=0, key=None, format_func=None, **kwargs):
        options = list(options)
        return options[index] if options else None

    def multiselect(self, label, options, default=None, **kwargs):
        return list(default) if default is not None else []

    def radio(self, label, options, **kwargs):
        return list(options)[0]

    def slider(self, label, minimum, maximum, value=None, step=None, **kwargs):
        return value if value is not None else minimum

    def number_input(self, label, minimum=0, maximum=100, value=None, step=None, **kwargs):
        return value if value is not None else minimum

    def text_input(self, label, value="", **kwargs):
        return value

    def text_area(self, label, value="", **kwargs):
        return value

    def button(self, label, **kwargs):
        return False          # never trigger the expensive on-click paths

    def form_submit_button(self, label, **kwargs):
        return False

    def checkbox(self, label, value=False, **kwargs):
        return value

    # -- layout ------------------------------------------------------------
    def columns(self, spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [self for _ in range(count)]

    # Columns and tabs are used both as objects (`col.metric(...)`) and as
    # context managers (`with col:`), so the stub has to support both.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def tabs(self, names):
        return [self for _ in names]

    @contextmanager
    def expander(self, label, expanded=False):
        self.calls.append(f"expander:{label}")
        yield self

    @contextmanager
    def form(self, key, **kwargs):
        yield self

    @contextmanager
    def spinner(self, text=""):
        yield self

    @contextmanager
    def sidebar(self):
        yield self

    def divider(self):
        self.calls.append("divider")

    # -- output ------------------------------------------------------------
    def _record(self, kind):
        def handler(*args, **kwargs):
            self.calls.append(kind)
            if kind == "error":
                self.errors.append(str(args[0]) if args else "")
            return None
        return handler

    def __getattr__(self, name):
        # Any other Streamlit call (write, markdown, dataframe, metric, ...)
        # is recorded rather than implemented.
        return self._record(name)


@pytest.fixture
def st():
    return StubStreamlit()


PAGES = [
    "dashboard.pages.overview",
    "dashboard.pages.district_detail",
    "dashboard.pages.disease_comparison",
    "dashboard.pages.data_quality",
    "dashboard.pages.model_performance",
    "dashboard.pages.intervention_tracker",
]


# ---------------------------------------------------------------- pages
@pytest.mark.parametrize("module_path", PAGES)
def test_every_page_renders_without_raising(module_path, st):
    """A page that raises takes down the operator's only view of the system."""
    import importlib

    page = importlib.import_module(module_path)
    page.render(st)          # must not raise
    assert st.calls, f"{module_path} drew nothing at all"


@pytest.mark.parametrize("module_path", PAGES)
def test_no_page_reports_an_internal_error(module_path, st):
    import importlib

    importlib.import_module(module_path).render(st)
    internal = [e for e in st.errors if "Traceback" in e or "Exception" in e]
    assert not internal, f"{module_path} surfaced an internal error: {internal}"


def test_app_module_declares_every_page():
    pytest.importorskip("streamlit", reason="streamlit is an optional extra")
    from dashboard.app import PAGES as APP_PAGES

    declared = {module for module, _ in APP_PAGES.values()}
    assert declared == set(PAGES)
    for _, description in APP_PAGES.values():
        assert description, "every page needs a one-line description in the sidebar"


# ----------------------------------------------------------- data access
def test_data_access_falls_back_to_the_local_cache():
    """Shortcoming #14: the dashboard must work with no API reachable."""
    from dashboard import data_access as da

    label = da.data_source_label()
    assert "cache" in label or "live API" in label
    assert da.diseases()
    assert not da.district_frame().empty
    # These must return empty frames, never raise, on a bare node.
    assert da.predictions(disease="malaria") is not None
    assert da.alerts(disease="malaria") is not None
    assert da.interventions() is not None


def test_offline_and_cache_status_are_readable():
    from dashboard import data_access as da

    status = da.cache_status()
    assert "predictions" in status and "offline_ready" in status
    readiness = da.offline_status()
    assert "message" in readiness


# ------------------------------------------------------------- components
def test_risk_colours_are_distinct_and_ordered():
    """Consistent colour coding in a decision tool is a safety property."""
    colours = [risk_color(level) for level in RISK_ORDER]
    assert len(set(colours)) == len(RISK_ORDER)
    assert [risk_rank(level) for level in RISK_ORDER] == [0, 1, 2, 3]
    assert risk_rank("nonsense") == -1


def test_risk_badge_includes_the_level_name():
    for level in RISK_ORDER:
        assert level.upper() in risk_badge(level)


def test_driver_table_labels_by_proxy_and_lag_not_column_name():
    """The reader is an epidemiologist, not the feature-pipeline author."""
    table = driver_table([{
        "feature": "rainfall_mm_lag6", "proxy": "rainfall", "lag_weeks": 6,
        "value": 142.0, "shap_value": 2.1, "contribution_share": 0.34,
        "direction": "increases", "mechanism": "rainfall creates breeding sites",
    }])
    assert table.loc[0, "driver"] == "rainfall, 6w ago"
    assert table.loc[0, "contribution_share"] == "34%"
    assert "breeding sites" in table.loc[0, "mechanism"]


def test_driver_table_handles_no_drivers():
    assert driver_table([]).empty


def test_recommendation_card_states_owner_deadline_and_quantity():
    """Shortcoming #12: an action without an owner or a deadline is not actionable."""
    card = recommendation_card({
        "action": "Pre-position 5,000 ORS kits",
        "responsible": "District Pharmacist",
        "timeframe_days": 14,
        "quantity": "5,000 sachets",
        "rationale": "Sized from the forecast",
        "priority": "high",
    }, index=1)
    assert "Pre-position 5,000 ORS kits" in card
    assert "District Pharmacist" in card
    assert "14 days" in card
    assert "5,000 sachets" in card
    assert "HIGH" in card


def test_summary_metrics_cover_every_level():
    metrics = summary_metrics({"low": 3, "high": 1})
    assert len(metrics) == len(RISK_ORDER)
    assert metrics[0][1] == 3
    assert metrics[2][1] == 1


def test_risk_table_sorts_worst_first():
    import pandas as pd

    frame = pd.DataFrame({
        "district": ["A", "B", "C"],
        "risk_level": ["low", "critical", "medium"],
        "risk_score": [0.1, 0.95, 0.5],
        "predicted_cases": [10, 900, 200],
    })
    table = risk_table(frame, top_n=3)
    assert list(table["district"]) == ["B", "C", "A"]
