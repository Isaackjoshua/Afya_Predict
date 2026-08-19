"""Data quality validation and flagging (shortcoming #7).

The rule the whole platform obeys: **flag, never silently drop or zero-fill**.
A missing DHIS2 week is `NaN` with `quality = 0`, not `0` cases. Quality scores
travel with every row and are aggregated per prediction, where they widen
confidence intervals and can stamp an alert LOW DATA CONFIDENCE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.core.timeutils import sort_weeks
from src.core.types import QualityFlag

if TYPE_CHECKING:  # pragma: no cover
    from src.data_ingestion.base_adapter import AdapterResult, BaseAdapter

log = get_logger("ingest.quality")

#: Physically plausible ranges. Values outside are flagged and quality-penalised,
#: never deleted — a real 55 degrees C reading matters even if it is suspect.
PLAUSIBLE_RANGES: Dict[str, tuple] = {
    "rainfall_mm": (0.0, 800.0),
    "temperature_c": (-15.0, 55.0),
    "humidity_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, 60.0),
    "ndvi": (-0.2, 1.0),
    "no2_mol_m2": (0.0, 1e-2),
    "pm25_ug_m3": (0.0, 1000.0),
    "aerosol_index": (-3.0, 10.0),
    "wash_access": (0.0, 1.0),
    "improved_sanitation": (0.0, 1.0),
    "surface_water_index": (0.0, 1.0),
    "population": (0.0, 5e7),
    "population_density_km2": (0.0, 5e4),
    "urban_share": (0.0, 1.0),
    "reporting_completeness": (0.0, 1.0),
}

#: Any `cases_*` variable must be a non-negative count.
CASE_PREFIX = "cases_"

#: z-score above which a value is treated as a spike worth flagging.
SPIKE_Z = 6.0
#: A district repeating one value this many weeks is probably a stuck sensor.
STUCK_RUN = 8


def validate_frame(
    frame: pd.DataFrame, source: str = "unknown"
) -> tuple[pd.DataFrame, List[QualityFlag]]:
    """Run every check over a tidy frame, returning it with adjusted quality."""
    flags: List[QualityFlag] = []
    if frame is None or frame.empty:
        return frame, [
            QualityFlag(code="empty_frame", severity="error", message="no rows produced", source=source)
        ]

    frame = frame.copy()
    frame["quality"] = pd.to_numeric(frame["quality"], errors="coerce").fillna(0.5)

    frame, missing_flags = _check_missing(frame, source)
    flags.extend(missing_flags)
    frame, range_flags = _check_ranges(frame, source)
    flags.extend(range_flags)
    frame, spike_flags = _check_spikes(frame, source)
    flags.extend(spike_flags)
    flags.extend(_check_gaps(frame, source))
    flags.extend(_check_stuck(frame, source))
    flags.extend(_check_coverage(frame, source))

    frame["quality"] = frame["quality"].clip(0.0, 1.0)
    return frame, flags


def _check_missing(frame: pd.DataFrame, source: str):
    """NaN values keep their row but drop to zero quality (never zero-filled)."""
    missing = frame["value"].isna()
    count = int(missing.sum())
    flags: List[QualityFlag] = []
    if count:
        frame.loc[missing, "quality"] = 0.0
        share = count / len(frame)
        flags.append(
            QualityFlag(
                code="missing_values",
                severity="error" if share > 0.25 else "warning",
                message=f"{count} missing value(s) ({share:.1%}) retained as NaN for explicit modelling",
                source=source,
                affected_rows=count,
            )
        )
    return frame, flags


def _check_ranges(frame: pd.DataFrame, source: str):
    """Flag physically implausible values and halve their quality."""
    flags: List[QualityFlag] = []
    for variable, (low, high) in PLAUSIBLE_RANGES.items():
        mask = (frame["variable"] == variable) & frame["value"].notna()
        if not mask.any():
            continue
        bad = mask & ((frame["value"] < low) | (frame["value"] > high))
        count = int(bad.sum())
        if count:
            frame.loc[bad, "quality"] *= 0.5
            flags.append(
                QualityFlag(
                    code="out_of_range",
                    severity="warning",
                    message=f"{count} {variable} value(s) outside plausible range [{low}, {high}]",
                    source=source,
                    affected_rows=count,
                )
            )
    case_mask = frame["variable"].str.startswith(CASE_PREFIX) & frame["value"].notna()
    negative = case_mask & (frame["value"] < 0)
    if negative.any():
        count = int(negative.sum())
        frame.loc[negative, "quality"] = 0.0
        flags.append(
            QualityFlag(
                code="negative_cases",
                severity="error",
                message=f"{count} negative case count(s) — upstream data-entry error",
                source=source,
                affected_rows=count,
            )
        )
    return frame, flags


def _check_spikes(frame: pd.DataFrame, source: str):
    """Robust (median/MAD) outlier detection per district x variable series."""
    flags: List[QualityFlag] = []
    values = frame["value"]
    grouped = frame.groupby(["district", "variable"])["value"]
    median = grouped.transform("median")
    mad = grouped.transform(lambda s: (s - s.median()).abs().median())
    scale = mad.replace(0, np.nan) * 1.4826
    z = (values - median).abs() / scale
    spikes = z > SPIKE_Z
    count = int(spikes.fillna(False).sum())
    if count:
        frame.loc[spikes.fillna(False), "quality"] *= 0.6
        flags.append(
            QualityFlag(
                code="value_spike",
                severity="warning",
                message=f"{count} value(s) exceed {SPIKE_Z} robust z-scores — possible reporting artefact",
                source=source,
                affected_rows=count,
            )
        )
    return frame, flags


def _check_gaps(frame: pd.DataFrame, source: str) -> List[QualityFlag]:
    """Detect districts missing whole weeks from the requested range."""
    weeks = sort_weeks(frame["week"].tolist())
    expected = len(weeks)
    if expected < 2:
        return []
    counts = frame.groupby("district")["week"].nunique()
    incomplete = counts[counts < expected * 0.9]
    if incomplete.empty:
        return []
    worst = ", ".join(incomplete.sort_values().head(5).index)
    return [
        QualityFlag(
            code="temporal_gaps",
            severity="warning",
            message=(
                f"{len(incomplete)} district(s) report fewer than 90% of the {expected} "
                f"requested weeks (e.g. {worst})"
            ),
            source=source,
            affected_rows=int(incomplete.sum()),
        )
    ]


def _check_stuck(frame: pd.DataFrame, source: str) -> List[QualityFlag]:
    """Detect a constant run in a series, which usually means a dead sensor."""
    suspects: List[str] = []
    for (district, variable), group in frame.groupby(["district", "variable"]):
        if variable in {"population", "urban_share", "wash_access", "improved_sanitation"}:
            continue  # legitimately near-constant
        series = group.sort_values("week")["value"].dropna()
        if len(series) < STUCK_RUN:
            continue
        runs = (series != series.shift()).cumsum()
        if runs.value_counts().max() >= STUCK_RUN:
            suspects.append(f"{district}/{variable}")
    if not suspects:
        return []
    return [
        QualityFlag(
            code="stuck_series",
            severity="warning",
            message=(
                f"{len(suspects)} series constant for >= {STUCK_RUN} weeks "
                f"(e.g. {', '.join(suspects[:3])}) — check the upstream feed"
            ),
            source=source,
            affected_rows=len(suspects),
        )
    ]


def _check_coverage(frame: pd.DataFrame, source: str) -> List[QualityFlag]:
    """Flag when very few districts are represented at all."""
    districts = frame["district"].nunique()
    if districts >= 5:
        return []
    return [
        QualityFlag(
            code="sparse_coverage",
            severity="warning",
            message=f"only {districts} district(s) present in this source",
            source=source,
            affected_rows=districts,
        )
    ]


def validate_adapter_result(result: "AdapterResult", adapter: "BaseAdapter") -> "AdapterResult":
    """Adapter-facing entry point: validate in place and append the flags."""
    if result.is_empty:
        result.flag("empty_frame", "adapter produced no rows", "error")
        return result
    frame, flags = validate_frame(result.frame, source=adapter.source_name)
    result.frame = frame
    result.flags.extend(flags)
    return result


def quality_report(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-source completeness/quality summary for the data-quality dashboard."""
    rows = []
    for source, frame in frames.items():
        if frame is None or frame.empty:
            rows.append({"source": source, "rows": 0, "districts": 0, "weeks": 0,
                         "completeness": 0.0, "mean_quality": 0.0, "latest_week": None})
            continue
        rows.append(
            {
                "source": source,
                "rows": len(frame),
                "districts": frame["district"].nunique(),
                "weeks": frame["week"].nunique(),
                "completeness": round(float(frame["value"].notna().mean()), 4),
                "mean_quality": round(float(frame["quality"].mean()), 4),
                "latest_week": sort_weeks(frame["week"].tolist())[-1],
            }
        )
    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


def aggregate_quality(
    frame: pd.DataFrame, district: Optional[str] = None, week: Optional[str] = None
) -> float:
    """Mean quality for a district/week slice — feeds interval widening."""
    subset = frame
    if district is not None:
        subset = subset[subset["district"] == district]
    if week is not None:
        subset = subset[subset["week"] == week]
    if subset.empty:
        return 0.0
    return float(subset["quality"].mean())
