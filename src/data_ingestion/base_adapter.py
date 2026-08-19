"""Abstract base class for every data source adapter.

Contract
--------
`fetch()`     -> pull the raw payload for a week range (live, cache or synthetic)
`validate()`  -> attach quality flags without dropping rows
`normalize()` -> return a tidy `district x week x variable` frame
`run()`       -> the orchestrated fetch → validate → normalize pipeline

Design rules honoured here:

* **#1 no hard failure** — a live fetch error degrades to cache, then to the
  deterministic synthetic generator, and the result is flagged, never dropped.
* **#7 quality propagates** — every row carries a 0–1 `quality` score and the
  result carries structured `QualityFlag`s that widen prediction intervals
  downstream.
* **#6 interoperability** — every adapter, whatever its wire format, returns
  the same five columns on the same spatiotemporal grid.
"""

from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings, get_settings
from src.core.config_loader import cached_region_config
from src.core.logging import get_logger
from src.core.timeutils import epi_week_start, sort_weeks, to_epi_week, week_range
from src.core.types import QualityFlag, RegionConfig

TIDY_COLUMNS = ["district", "week", "variable", "value", "quality", "source"]


class FetchMode(str, Enum):
    """Where the data in an :class:`AdapterResult` actually came from."""

    LIVE = "live"          # fetched from the upstream API/archive
    CACHE = "cache"        # replayed from the local parquet/csv cache
    SYNTHETIC = "synthetic"  # deterministic climatological simulation
    EMPTY = "empty"        # nothing available at all


@dataclass
class AdapterResult:
    """Tidy data plus the provenance and quality metadata that travels with it."""

    source: str
    frame: pd.DataFrame
    mode: FetchMode = FetchMode.EMPTY
    flags: List[QualityFlag] = field(default_factory=list)
    fetched_at: Optional[date] = None
    latest_data_date: Optional[date] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.frame is None or self.frame.empty

    @property
    def variables(self) -> List[str]:
        if self.is_empty:
            return []
        return sorted(self.frame["variable"].unique().tolist())

    @property
    def mean_quality(self) -> float:
        if self.is_empty:
            return 0.0
        return float(self.frame["quality"].mean())

    def flag(self, code: str, message: str, severity: str = "warning", **kwargs: Any) -> None:
        self.flags.append(
            QualityFlag(code=code, message=message, severity=severity, source=self.source, **kwargs)
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "mode": self.mode.value,
            "rows": 0 if self.is_empty else len(self.frame),
            "variables": self.variables,
            "mean_quality": round(self.mean_quality, 3),
            "latest_data_date": self.latest_data_date,
            "flags": [f.model_dump() for f in self.flags],
        }


class BaseAdapter(abc.ABC):
    """Base class every data source adapter inherits from."""

    #: registry key, matches `source:` in the disease YAML configs
    source_name: str = "base"
    #: variables this adapter produces on the tidy grid
    variables: Sequence[str] = ()
    #: how often upstream publishes, in days (used by the scheduler)
    update_frequency_days: int = 7
    #: whether the source is optional (rule #14: never a primary predictor)
    optional: bool = False
    #: nominal spatial resolution, for the data-status endpoint
    native_resolution: str = "unknown"

    def __init__(
        self,
        region: Optional[RegionConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.region = region or cached_region_config(self.settings.default_region)
        self.log = get_logger(f"ingest.{self.source_name}")

    # -- capability -------------------------------------------------------
    def is_configured(self) -> bool:
        """True when credentials/paths for a live fetch are present."""
        return False

    # -- the three-step contract -----------------------------------------
    @abc.abstractmethod
    def fetch_live(self, weeks: List[str]) -> pd.DataFrame:
        """Pull real data from upstream. Raise on failure; `run()` degrades."""

    @abc.abstractmethod
    def synthesize(self, weeks: List[str]) -> pd.DataFrame:
        """Deterministic fallback consistent with the region's climatology."""

    def fetch(self, weeks: List[str]) -> AdapterResult:
        """Live → cache → synthetic, in that order, never raising."""
        result = AdapterResult(source=self.source_name, frame=pd.DataFrame(columns=TIDY_COLUMNS))
        result.fetched_at = date.today()

        if self.settings.offline_mode:
            self.log.debug("offline mode: skipping live fetch for %s", self.source_name)
        elif self.is_configured():
            try:
                frame = self.fetch_live(weeks)
                if frame is not None and not frame.empty:
                    result.frame = self._coerce(frame)
                    result.mode = FetchMode.LIVE
                    self.write_cache(result.frame)
                    return self._finalise(result)
                result.flag("empty_live_response", "upstream returned no rows", "warning")
            except Exception as exc:  # noqa: BLE001 - degradation is the point
                self.log.warning("live fetch failed for %s: %s", self.source_name, exc)
                result.flag("live_fetch_failed", f"live fetch failed: {exc}", "warning")
        else:
            result.flag(
                "not_configured",
                "no credentials configured; using cache or synthetic fallback",
                "info",
            )

        cached = self.read_cache(weeks)
        if cached is not None and not cached.empty:
            result.frame = self._coerce(cached)
            result.mode = FetchMode.CACHE
            result.flag("served_from_cache", "served from local cache", "info")
            return self._finalise(result)

        frame = self.synthesize(weeks)
        result.frame = self._coerce(frame)
        result.mode = FetchMode.SYNTHETIC
        result.flag(
            "synthetic_fallback",
            "no live or cached data; deterministic synthetic climatology used — "
            "predictions from this source carry reduced confidence",
            "warning",
        )
        return self._finalise(result)

    def validate(self, result: AdapterResult) -> AdapterResult:
        """Attach quality flags and downgrade `quality` scores in place."""
        from src.data_ingestion.quality_checks import validate_adapter_result

        return validate_adapter_result(result, self)

    def normalize(self, result: AdapterResult) -> AdapterResult:
        """Ensure the frame sits exactly on the district x week grid."""
        if result.is_empty:
            return result
        frame = result.frame.copy()
        known = set(self.region.district_names)
        unknown = sorted(set(frame["district"]) - known)
        if unknown:
            result.flag(
                "unknown_districts",
                f"dropped {len(unknown)} district(s) absent from the region grid: "
                f"{', '.join(unknown[:5])}{'...' if len(unknown) > 5 else ''}",
                "warning",
                affected_rows=int(frame["district"].isin(unknown).sum()),
            )
            frame = frame[frame["district"].isin(known)]
        frame = (
            frame.groupby(["district", "week", "variable"], as_index=False)
            .agg(value=("value", "mean"), quality=("quality", "min"))
            .assign(source=self.source_name)
        )
        result.frame = frame[TIDY_COLUMNS].sort_values(["variable", "district", "week"]).reset_index(drop=True)
        return result

    def run(self, start_week: str, end_week: str) -> AdapterResult:
        """Full pipeline for a week range: fetch → validate → normalize."""
        weeks = week_range(start_week, end_week)
        if not weeks:
            return AdapterResult(source=self.source_name, frame=pd.DataFrame(columns=TIDY_COLUMNS))
        self.log.info(
            "%s: ingesting %d week(s) %s..%s", self.source_name, len(weeks), weeks[0], weeks[-1]
        )
        result = self.fetch(weeks)
        result = self.validate(result)
        result = self.normalize(result)
        self.log.info(
            "%s: %d rows (%s), mean quality %.2f",
            self.source_name,
            0 if result.is_empty else len(result.frame),
            result.mode.value,
            result.mean_quality,
        )
        return result

    # -- caching ----------------------------------------------------------
    @property
    def cache_path(self) -> Path:
        return Path(self.settings.cache_dir) / f"{self.source_name}.parquet"

    def write_cache(self, frame: pd.DataFrame) -> None:
        """Merge `frame` into the on-disk cache (offline-first, rule #6)."""
        if frame is None or frame.empty:
            return
        try:
            existing = self.read_cache(None)
            merged = (
                pd.concat([existing, frame], ignore_index=True)
                if existing is not None and not existing.empty
                else frame
            )
            merged = merged.drop_duplicates(subset=["district", "week", "variable"], keep="last")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                merged.to_parquet(self.cache_path, index=False)
            except Exception:  # pyarrow missing — CSV keeps offline mode alive
                merged.to_csv(self.cache_path.with_suffix(".csv"), index=False)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("cache write failed for %s: %s", self.source_name, exc)

    def read_cache(self, weeks: Optional[List[str]]) -> Optional[pd.DataFrame]:
        """Read the cache, optionally restricted to `weeks`."""
        path = self.cache_path if self.cache_path.exists() else self.cache_path.with_suffix(".csv")
        if not path.exists():
            return None
        try:
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("cache read failed for %s: %s", self.source_name, exc)
            return None
        if weeks is not None:
            frame = frame[frame["week"].isin(set(weeks))]
            missing = set(weeks) - set(frame["week"].unique())
            # A cache that covers less than 80% of the request is not usable.
            if len(missing) > 0.2 * len(weeks):
                return None
        return frame

    # -- synthetic helpers -----------------------------------------------
    def _rng(self, *tokens: str) -> np.random.Generator:
        """Deterministic RNG keyed by source + tokens (reproducible fixtures)."""
        digest = hashlib.sha256("|".join((self.source_name,) + tokens).encode()).hexdigest()
        return np.random.default_rng(int(digest[:16], 16) % (2**32))

    def _season_phase(self, week: str) -> float:
        """Position within the year in radians (0 at ISO week 1)."""
        start = epi_week_start(week)
        day_of_year = start.timetuple().tm_yday
        return 2 * np.pi * day_of_year / 365.25

    def tidy(
        self,
        records: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Build a tidy frame from `{district, week, variable, value, quality}` dicts."""
        if not records:
            return pd.DataFrame(columns=TIDY_COLUMNS)
        frame = pd.DataFrame.from_records(records)
        frame["source"] = self.source_name
        if "quality" not in frame:
            frame["quality"] = 1.0
        return frame[TIDY_COLUMNS]

    # -- internals --------------------------------------------------------
    def _coerce(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Force the tidy schema and dtypes onto whatever the adapter produced."""
        if frame is None or frame.empty:
            return pd.DataFrame(columns=TIDY_COLUMNS)
        frame = frame.copy()
        if "source" not in frame:
            frame["source"] = self.source_name
        if "quality" not in frame:
            frame["quality"] = 1.0
        missing = [c for c in TIDY_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{self.source_name}: adapter frame missing columns {missing}")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["quality"] = pd.to_numeric(frame["quality"], errors="coerce").fillna(0.5).clip(0, 1)
        frame["district"] = frame["district"].astype(str)
        frame["week"] = frame["week"].astype(str)
        frame["variable"] = frame["variable"].astype(str)
        return frame[TIDY_COLUMNS]

    def _finalise(self, result: AdapterResult) -> AdapterResult:
        if not result.is_empty:
            weeks = sort_weeks(result.frame["week"].tolist())
            if weeks:
                result.latest_data_date = epi_week_start(weeks[-1])
        result.meta.setdefault("update_frequency_days", self.update_frequency_days)
        result.meta.setdefault("native_resolution", self.native_resolution)
        return result

    # -- misc -------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "source": self.source_name,
            "variables": list(self.variables),
            "configured": self.is_configured(),
            "optional": self.optional,
            "update_frequency_days": self.update_frequency_days,
            "native_resolution": self.native_resolution,
        }

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return f"<{type(self).__name__} source={self.source_name} configured={self.is_configured()}>"
