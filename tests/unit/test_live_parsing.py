"""The live upstream parsing paths, driven by captured payloads.

These paths only execute when credentials are present, so nothing else in the
suite reaches them — which makes them exactly the code most likely to break
unnoticed between a working demo and a real deployment.

Fixtures are captured response shapes, not synthetic data; see
`tests/fixtures/README.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def analytics_payload():
    return json.loads((FIXTURES / "dhis2_analytics.json").read_text())


# ------------------------------------------------------------------- DHIS2
def test_dhis2_period_ids_convert_to_iso_weeks():
    from src.data_ingestion.adapters.dhis2_surveillance import _dhis2_period_to_epi_week

    assert _dhis2_period_to_epi_week("2024W10") == "2024-W10"
    assert _dhis2_period_to_epi_week("2024W7") == "2024-W07"      # single digit padded
    assert _dhis2_period_to_epi_week("2024-W10") == "2024-W10"    # already normalised


def test_dhis2_analytics_response_maps_onto_the_tidy_grid(analytics_payload, small_region, monkeypatch):
    """Header order is data, not layout: the parser must not assume positions."""
    from src.data_ingestion.adapters.dhis2_surveillance import DHIS2SurveillanceAdapter

    adapter = DHIS2SurveillanceAdapter(region=small_region, diseases=["malaria", "cholera"])

    headers = [h["name"] for h in analytics_payload["headers"]]
    frame = pd.DataFrame(analytics_payload["rows"], columns=headers)
    reverse = {"MAL_CONF": "malaria", "CHO_SUSP": "cholera"}

    from src.data_ingestion.adapters.dhis2_surveillance import (
        CASE_VARIABLES, _dhis2_period_to_epi_week,
    )

    records = []
    for _, row in frame.iterrows():
        slug = reverse.get(row["dx"])
        if slug is None:
            continue
        records.append({
            "district": str(row["ou"]),
            "week": _dhis2_period_to_epi_week(str(row["pe"])),
            "variable": CASE_VARIABLES[slug],
            "value": pd.to_numeric(row["value"], errors="coerce"),
            "quality": 1.0,
        })
    tidy = adapter.tidy(records)

    assert set(tidy.columns) == {"district", "week", "variable", "value", "quality", "source"}
    assert tidy["week"].str.match(r"^\d{4}-W\d{2}$").all()
    assert set(tidy["variable"]) == {"cases_malaria", "cases_cholera"}
    assert tidy.loc[tidy["district"] == "Kinondoni", "value"].max() == 1902


def test_unknown_districts_are_dropped_by_normalisation(analytics_payload, small_region):
    """The fixture contains 'Atlantis' on purpose."""
    from src.data_ingestion.adapters.dhis2_surveillance import (
        CASE_VARIABLES, DHIS2SurveillanceAdapter, _dhis2_period_to_epi_week,
    )
    from src.data_ingestion.base_adapter import AdapterResult, FetchMode

    adapter = DHIS2SurveillanceAdapter(region=small_region)
    headers = [h["name"] for h in analytics_payload["headers"]]
    frame = pd.DataFrame(analytics_payload["rows"], columns=headers)
    records = [
        {"district": str(r["ou"]), "week": _dhis2_period_to_epi_week(str(r["pe"])),
         "variable": CASE_VARIABLES["malaria"], "value": float(r["value"]), "quality": 1.0}
        for _, r in frame.iterrows() if r["dx"] == "MAL_CONF"
    ]
    result = AdapterResult(source="dhis2", frame=adapter.tidy(records), mode=FetchMode.LIVE)
    normalized = adapter.normalize(result)

    assert "Atlantis" not in set(normalized.frame["district"])
    assert any(f.code == "unknown_districts" for f in normalized.flags)


def test_data_element_search_shape_is_understood():
    payload = json.loads((FIXTURES / "dhis2_data_elements.json").read_text())
    elements = payload.get("dataElements", [])
    assert elements and elements[0]["id"] == "MAL_CONF"
    # The adapter takes the first match, which is why the search filter matters.
    assert all({"id", "name"} <= set(e) for e in elements)


# ------------------------------------------------------------------- CDR
def test_cdr_od_file_is_read_and_reduced(small_region, tmp_path, monkeypatch):
    from config.settings import get_settings
    from src.data_ingestion.adapters.cdr_mobility import CDRMobilityAdapter

    drop = tmp_path / "cdr"
    drop.mkdir()
    (drop / "week10.csv").write_text((FIXTURES / "cdr_origin_destination.csv").read_text())

    settings = get_settings().model_copy(update={"cdr_data_dir": drop})
    adapter = CDRMobilityAdapter(region=small_region, settings=settings)

    assert adapter.is_configured() is True
    od = adapter.read_od_files(["2024-W10"])
    assert len(od) == 8                       # week 11 rows filtered out
    assert set(od.columns) == {"origin", "destination", "week", "trips"}

    tidy = adapter._od_to_tidy(od, ["2024-W10"], quality=1.0)
    inbound = tidy[(tidy["variable"] == "mobility_inbound") & (tidy["district"] == "Kinondoni")]
    # 51,100 from Ilala + 1,180 from Mwanza City. The Atlantis row is dropped as
    # an unknown district, and the internal Kinondoni->Kinondoni trip is not inbound.
    assert inbound["value"].iloc[0] == pytest.approx(52280.0)

    internal = tidy[(tidy["variable"] == "mobility_internal") & (tidy["district"] == "Kinondoni")]
    assert internal["value"].iloc[0] == pytest.approx(412000.0)


def test_empirical_travel_matrix_beats_the_gravity_fallback(small_region, tmp_path):
    """Rule #8: use real mobility when it exists, the model only when it does not."""
    import numpy as np

    from config.settings import get_settings
    from src.data_ingestion.adapters.cdr_mobility import CDRMobilityAdapter

    drop = tmp_path / "cdr"
    drop.mkdir()
    (drop / "od.csv").write_text((FIXTURES / "cdr_origin_destination.csv").read_text())
    settings = get_settings().model_copy(update={"cdr_data_dir": drop})

    empirical = CDRMobilityAdapter(region=small_region, settings=settings).travel_matrix("2024-W10")
    fallback = CDRMobilityAdapter(region=small_region).travel_matrix()

    assert np.allclose(np.diag(empirical.to_numpy()), 0.0)
    assert empirical.loc["Kinondoni", "Ilala"] > 0.9        # the fixture's dominant flow
    assert not np.allclose(empirical.to_numpy(), fallback.to_numpy())


def test_malformed_cdr_files_are_skipped_not_fatal(small_region, tmp_path):
    from config.settings import get_settings
    from src.data_ingestion.adapters.cdr_mobility import CDRMobilityAdapter

    drop = tmp_path / "cdr"
    drop.mkdir()
    (drop / "good.csv").write_text((FIXTURES / "cdr_origin_destination.csv").read_text())
    (drop / "bad.csv").write_text("something,entirely,different\n1,2,3\n")

    settings = get_settings().model_copy(update={"cdr_data_dir": drop})
    od = CDRMobilityAdapter(region=small_region, settings=settings).read_od_files()
    assert not od.empty                       # the good file still loads


# ------------------------------------------------------------------- WASH
def test_jmp_extract_is_parsed(small_region, tmp_path):
    from config.settings import get_settings
    from src.data_ingestion.adapters.wash_indicators import WASHIndicatorsAdapter

    raw = tmp_path / "raw" / "wash"
    raw.mkdir(parents=True)
    (raw / "jmp_subnational.csv").write_text((FIXTURES / "jmp_subnational.csv").read_text())

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    adapter = WASHIndicatorsAdapter(region=small_region, settings=settings)

    assert adapter.is_configured() is True
    tidy = adapter.fetch_live(["2023-W10", "2024-W10"])
    assert not tidy.empty
    assert set(tidy["variable"]) == {"wash_access", "improved_sanitation"}
    kinondoni = tidy[(tidy["district"] == "Kinondoni") & (tidy["week"] == "2023-W10")
                     & (tidy["variable"] == "wash_access")]
    assert kinondoni["value"].iloc[0] == pytest.approx(0.83)


def test_jmp_extract_rejects_missing_columns(small_region, tmp_path):
    from config.settings import get_settings
    from src.data_ingestion.adapters.wash_indicators import WASHIndicatorsAdapter

    raw = tmp_path / "raw" / "wash"
    raw.mkdir(parents=True)
    (raw / "jmp_subnational.csv").write_text("district,year\nKinondoni,2023\n")

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    adapter = WASHIndicatorsAdapter(region=small_region, settings=settings)
    with pytest.raises(ValueError, match="must contain columns"):
        adapter.fetch_live(["2023-W10"])


def test_a_broken_live_path_degrades_to_synthetic(small_region, tmp_path):
    """Rule #1: a malformed upstream file must not stop the pipeline."""
    from config.settings import get_settings
    from src.data_ingestion.adapters.wash_indicators import WASHIndicatorsAdapter

    raw = tmp_path / "raw" / "wash"
    raw.mkdir(parents=True)
    (raw / "jmp_subnational.csv").write_text("district,year\nKinondoni,2023\n")

    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    result = WASHIndicatorsAdapter(region=small_region, settings=settings).run(
        "2024-W01", "2024-W04"
    )
    assert not result.is_empty
    assert any(f.code == "live_fetch_failed" for f in result.flags)
