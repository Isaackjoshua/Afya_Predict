"""Assemble the model-ready design matrix from a fused panel.

`FeatureBuilder` is the single place that decides, for one disease, which
columns exist and what each one means. It:

1. maps each declared digital proxy to the panel variable that carries it,
2. fits the optimal lag per district (rule #3) and materialises those lags,
3. adds rolling/anomaly, seasonal, spatial, mobility, interaction, shaped and
   One Health features,
4. records provenance for every column so the explainer can say *why* a feature
   mattered in mechanistic language, not just "feature_37 = 0.42".

Provenance is what makes shortcoming #4 solvable end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.core.types import DiseaseConfig, FeatureSpec, RegionConfig
from src.data_ingestion.normalizer import IMPUTED_SUFFIX, QUALITY_SUFFIX, Panel
from src.feature_engineering.interaction_terms import add_interactions, add_shaped_features
from src.feature_engineering.lag_features import (
    LagFit,
    add_lag_features,
    fit_optimal_lags,
    lag_column,
)
from src.feature_engineering.mobility_features import add_mobility_features, get_travel_matrix
from src.feature_engineering.one_health_features import add_one_health_features
from src.feature_engineering.rolling_stats import add_anomalies, add_deltas, add_rolling_means
from src.feature_engineering.seasonality import add_seasonality
from src.feature_engineering.spatial_features import (
    add_connectivity_features,
    add_neighbour_features,
    neighbour_case_pressure,
)

log = get_logger("features.builder")

#: Panel variable that carries each declared proxy name. A proxy is a *concept*
#: ("rainfall"); a variable is the column an adapter actually produced
#: ("rainfall_mm"). Keeping them separate is what lets a new adapter supply an
#: existing proxy without touching any disease config.
PROXY_TO_VARIABLE: Dict[str, str] = {
    "rainfall": "rainfall_mm",
    "temperature": "temperature_c",
    "humidity": "humidity_pct",
    "wind_speed": "wind_speed_ms",
    "ndvi": "ndvi",
    "air_quality_pm25": "pm25_ug_m3",
    "air_quality_no2": "no2_mol_m2",
    "aerosol": "aerosol_index",
    "mobility_inbound": "mobility_inbound",
    "mobility_outbound": "mobility_outbound",
    "mobility_internal": "mobility_internal",
    "population_density": "population_density_km2",
    "wash_access": "wash_access",
    "sanitation": "improved_sanitation",
    "surface_water": "surface_water_index",
    "livestock_events": "livestock_outbreak_events",
    "livestock_mortality": "livestock_mortality",
}


@dataclass
class FeatureMatrix:
    """Design matrix plus everything needed to explain and reproduce it."""

    X: pd.DataFrame
    y: pd.Series
    disease: str
    feature_names: List[str] = field(default_factory=list)
    provenance: Dict[str, dict] = field(default_factory=dict)
    lag_fits: Dict[str, Dict[str, LagFit]] = field(default_factory=dict)
    row_quality: Optional[pd.Series] = None
    target_column: str = ""
    population: Optional[pd.Series] = None

    @property
    def districts(self) -> List[str]:
        return sorted(self.X.index.get_level_values("district").unique())

    @property
    def weeks(self) -> List[str]:
        from src.core.timeutils import sort_weeks

        return sort_weeks(self.X.index.get_level_values("week").unique().tolist())

    def for_district(self, district: str) -> "FeatureMatrix":
        mask = self.X.index.get_level_values("district") == district
        return FeatureMatrix(
            X=self.X[mask],
            y=self.y[mask],
            disease=self.disease,
            feature_names=list(self.feature_names),
            provenance=self.provenance,
            lag_fits={district: self.lag_fits.get(district, {})},
            row_quality=None if self.row_quality is None else self.row_quality[mask],
            target_column=self.target_column,
            population=self.population,
        )

    def dropna_rows(self, min_feature_coverage: float = 0.6) -> "FeatureMatrix":
        """Drop rows with a missing target or too many missing features."""
        coverage = self.X.notna().mean(axis=1)
        keep = self.y.notna() & (coverage >= min_feature_coverage)
        return FeatureMatrix(
            X=self.X[keep],
            y=self.y[keep],
            disease=self.disease,
            feature_names=list(self.feature_names),
            provenance=self.provenance,
            lag_fits=self.lag_fits,
            row_quality=None if self.row_quality is None else self.row_quality[keep],
            target_column=self.target_column,
            population=self.population,
        )

    def describe_feature(self, name: str) -> dict:
        return self.provenance.get(name, {"feature": name, "kind": "unknown"})


class FeatureBuilder:
    """Turn a fused `Panel` into a `FeatureMatrix` for one disease."""

    def __init__(
        self,
        config: DiseaseConfig,
        region: RegionConfig,
        include_optional_sources: bool = False,
        fourier_order: int = 3,
        neighbour_k: int = 5,
    ) -> None:
        self.config = config
        self.region = region
        self.include_optional_sources = include_optional_sources
        self.fourier_order = fourier_order
        self.neighbour_k = neighbour_k
        self.log = log

    # -- proxy resolution --------------------------------------------------
    @property
    def target_column(self) -> str:
        from src.data_ingestion.adapters.dhis2_surveillance import CASE_VARIABLES

        return CASE_VARIABLES.get(self.config.slug, f"cases_{self.config.slug}")

    def usable_specs(self, panel: Panel) -> List[FeatureSpec]:
        """Proxies whose backing variable actually made it into the panel."""
        specs = []
        for spec in self.config.digital_proxies:
            if spec.optional and not self.include_optional_sources:
                continue
            variable = self.variable_for(spec)
            if variable in panel.frame.columns:
                specs.append(spec)
            else:
                self.log.warning(
                    "proxy %r for %s has no data (%s missing from panel); skipped",
                    spec.name, self.config.name, variable,
                )
        return specs

    def variable_for(self, spec: FeatureSpec) -> str:
        if spec.name in PROXY_TO_VARIABLE:
            return PROXY_TO_VARIABLE[spec.name]
        if spec.name.startswith("search_") or spec.source == "search_trends":
            return f"search_{self.config.slug}"
        return spec.name

    # -- build -------------------------------------------------------------
    def build(
        self,
        panel: Panel,
        horizon_weeks: Optional[int] = None,
        lag_fits: Optional[Dict[str, Dict[str, LagFit]]] = None,
    ) -> FeatureMatrix:
        """Construct the design matrix and the forward-shifted target.

        `y` is the case count `horizon_weeks` *ahead* of the feature row, which
        is what makes this a forecasting system rather than a nowcast: at
        prediction time only data available today enters `X`.
        """
        horizon = horizon_weeks if horizon_weeks is not None else self.config.model.forecast_horizon_weeks
        values = panel.values().sort_index()
        target_column = self.target_column
        if target_column not in values.columns:
            raise KeyError(
                f"panel has no target column {target_column!r}; "
                f"available: {', '.join(sorted(values.columns))}"
            )

        specs = self.usable_specs(panel)
        if len({s.source for s in specs}) < 3:
            self.log.warning(
                "%s: only %d source(s) available — rule #1 wants >= 3; intervals will widen",
                self.config.name, len({s.source for s in specs}),
            )

        variable_for_proxy = {spec.name: self.variable_for(spec) for spec in specs}

        # 1. Fit the lags rather than trusting the YAML's prior (rule #3).
        fits = lag_fits or fit_optimal_lags(
            values, target_column, specs, variable_for_proxy
        )

        frame = values.copy()
        provenance: Dict[str, dict] = {}

        # 2. Materialise the fitted lag per proxy, plus neighbouring lags so the
        #    model can refine the choice the correlation scan made.
        selected_lags: Dict[str, List[int]] = {}
        for spec in specs:
            variable = variable_for_proxy[spec.name]
            lags = {
                fit.lag_weeks for fit in (f.get(spec.name) for f in fits.values()) if fit
            } or {spec.optimal_lag_weeks}
            low, high = spec.lag_weeks_range
            extra = {l for lag in list(lags) for l in (lag - 1, lag + 1)}
            lags = sorted({l for l in lags | extra if low <= l <= high} or {spec.optimal_lag_weeks})
            selected_lags[variable] = lags
            frame = add_lag_features(frame, [variable], lags)
            for lag in lags:
                name = lag_column(variable, lag)
                provenance[name] = {
                    "feature": name,
                    "kind": "lagged_proxy",
                    "proxy": spec.name,
                    "variable": variable,
                    "lag_weeks": lag,
                    "source": spec.source,
                    "mechanism": spec.mechanism,
                    "relationship": spec.relationship.value,
                }

        # 3. Mechanism-shaped versions of the fitted-lag columns.
        primary_column = {
            spec.name: lag_column(
                variable_for_proxy[spec.name],
                _modal_lag(fits, spec.name, spec.optimal_lag_weeks),
            )
            for spec in specs
        }
        frame = add_shaped_features(frame, specs, primary_column)
        for spec in specs:
            name = f"{spec.name}_shaped"
            if name in frame.columns:
                provenance[name] = {
                    "feature": name,
                    "kind": "response_shape",
                    "proxy": spec.name,
                    "variable": variable_for_proxy[spec.name],
                    "lag_weeks": _modal_lag(fits, spec.name, spec.optimal_lag_weeks),
                    "source": spec.source,
                    "mechanism": spec.mechanism,
                    "relationship": spec.relationship.value,
                }

        # 4. Rolling context and anomalies on the raw drivers.
        driver_variables = sorted(set(variable_for_proxy.values()))
        frame = add_rolling_means(frame, driver_variables, windows=(4, 8, 12))
        frame = add_anomalies(frame, driver_variables)
        frame = add_deltas(frame, driver_variables, periods=(1, 4))
        for variable in driver_variables:
            proxy = next((p for p, v in variable_for_proxy.items() if v == variable), variable)
            spec = next((s for s in specs if s.name == proxy), None)
            for suffix, kind, note in (
                ("_roll4", "rolling", "4-week trailing mean"),
                ("_roll8", "rolling", "8-week trailing mean"),
                ("_roll12", "rolling", "12-week trailing mean"),
                ("_anomaly", "anomaly", "deviation from the district's own trailing baseline"),
                ("_zscore", "anomaly", "standardised deviation from the district's baseline"),
                ("_delta1", "trend", "week-on-week change"),
                ("_delta4", "trend", "4-week change"),
            ):
                name = f"{variable}{suffix}"
                if name in frame.columns:
                    provenance[name] = {
                        "feature": name, "kind": kind, "proxy": proxy, "variable": variable,
                        "lag_weeks": 0, "source": spec.source if spec else "derived",
                        "mechanism": note if spec is None else f"{note} of {spec.mechanism}",
                    }

        # 5. Target history — autoregression is a legitimate signal, and its
        #    absence is what makes a model lose to the naive baseline.
        frame = _add_target_history(frame, target_column)
        for name, note in (
            (f"{target_column}_lag1", "last week's reported cases"),
            (f"{target_column}_lag2", "cases two weeks ago"),
            (f"{target_column}_lag4", "cases four weeks ago"),
            (f"{target_column}_roll4", "4-week mean of reported cases"),
            (f"{target_column}_roll8", "8-week mean of reported cases"),
            (f"{target_column}_trend4", "4-week change in reported cases"),
            (f"{target_column}_yoy", "same week last year"),
        ):
            if name in frame.columns:
                provenance[name] = {
                    "feature": name, "kind": "autoregressive", "proxy": "case_history",
                    "variable": target_column, "lag_weeks": 1, "source": "dhis2",
                    "mechanism": note,
                }

        # 6. Seasonality.
        frame = add_seasonality(frame, fourier_order=self.fourier_order)
        for name in frame.columns:
            if name.startswith(("fourier_", "week_", "season_", "time_index")) and name not in provenance:
                provenance[name] = {
                    "feature": name, "kind": "seasonality", "proxy": "season",
                    "variable": name, "lag_weeks": 0, "source": "calendar",
                    "mechanism": "annual transmission cycle",
                }

        # 7. Spatial context and importation pressure.
        frame = add_connectivity_features(frame, self.region)
        frame = add_neighbour_features(
            frame, self.region, [target_column] + driver_variables[:3], k=self.neighbour_k
        )
        frame["neighbour_case_pressure"] = neighbour_case_pressure(
            frame, self.region, target_column, k=self.neighbour_k
        )
        travel = get_travel_matrix(self.region) if self.config.spatial.enabled else None
        frame = add_mobility_features(
            frame, self.region, case_column=target_column, travel_matrix=travel
        )
        for name in frame.columns:
            if name in provenance:
                continue
            if name.startswith("importation_pressure"):
                provenance[name] = {
                    "feature": name, "kind": "spatial", "proxy": "importation",
                    "variable": name, "lag_weeks": _trailing_int(name),
                    "source": "cdr_mobility",
                    "mechanism": "incidence in connected districts weighted by travel flow",
                }
            elif name.endswith(("_nbr_mean", "_nbr_max")) or name == "neighbour_case_pressure":
                provenance[name] = {
                    "feature": name, "kind": "spatial", "proxy": "neighbourhood",
                    "variable": name, "lag_weeks": 0, "source": "derived",
                    "mechanism": "conditions in the surrounding districts",
                }
            elif name.startswith("mobility_") or name in {"accessibility_index", "remoteness_km"}:
                provenance[name] = {
                    "feature": name, "kind": "connectivity", "proxy": "mobility",
                    "variable": name, "lag_weeks": 0, "source": "cdr_mobility",
                    "mechanism": "how strongly this district is connected to the rest of the network",
                }

        # 8. Interactions.
        frame, interaction_mechanisms = add_interactions(frame, primary_column)
        for name, mechanism in interaction_mechanisms.items():
            provenance[name] = {
                "feature": name, "kind": "interaction", "proxy": name,
                "variable": name, "lag_weeks": 0, "source": "derived", "mechanism": mechanism,
            }

        # 9. One Health.
        frame = add_one_health_features(frame, case_column=target_column)
        for name in ("one_health_animal_anomaly", "one_health_environment_anomaly",
                     "zoonotic_spillover_score", "animal_leads_human"):
            if name in frame.columns:
                provenance[name] = {
                    "feature": name, "kind": "one_health", "proxy": "one_health",
                    "variable": name, "lag_weeks": 0, "source": "livestock_disease",
                    "mechanism": "convergence of animal, environmental and human anomalies",
                }

        # 10. Target: shift the case count *backwards* so row t predicts t+h.
        y = (
            frame.groupby(level="district")[target_column]
            .shift(-horizon)
            .rename(f"{target_column}_h{horizon}")
        )

        feature_names = [
            c for c in frame.columns
            if c in provenance and not c.endswith((QUALITY_SUFFIX, IMPUTED_SUFFIX))
        ]
        # Never let the contemporaneous target leak into its own prediction.
        feature_names = [c for c in feature_names if c != target_column]

        row_quality = self._row_quality(panel, frame.index)
        populations = pd.Series({d.name: float(d.population) for d in self.region.districts})

        matrix = FeatureMatrix(
            X=frame[feature_names],
            y=y,
            disease=self.config.slug,
            feature_names=feature_names,
            provenance=provenance,
            lag_fits=fits,
            row_quality=row_quality,
            target_column=target_column,
            population=populations,
        )
        self.log.info(
            "%s: built %d features over %d district-weeks (horizon %dw)",
            self.config.name, len(feature_names), len(matrix.X), horizon,
        )
        return matrix

    # -- helpers -----------------------------------------------------------
    def _row_quality(self, panel: Panel, index: pd.MultiIndex) -> pd.Series:
        """Mean input quality per district-week — drives interval widening."""
        quality = panel.quality()
        if quality.empty:
            return pd.Series(1.0, index=index, name="row_quality")
        return quality.mean(axis=1).reindex(index).fillna(0.0).rename("row_quality")


def _add_target_history(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    grouped = frame.groupby(level="district")[target_column]
    new = {
        f"{target_column}_lag1": grouped.shift(1),
        f"{target_column}_lag2": grouped.shift(2),
        f"{target_column}_lag4": grouped.shift(4),
        f"{target_column}_roll4": grouped.transform(
            lambda s: s.shift(1).rolling(4, min_periods=2).mean()
        ),
        f"{target_column}_roll8": grouped.transform(
            lambda s: s.shift(1).rolling(8, min_periods=3).mean()
        ),
        f"{target_column}_trend4": grouped.shift(1) - grouped.shift(5),
        f"{target_column}_yoy": grouped.shift(52),
    }
    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)


def _modal_lag(fits: Dict[str, Dict[str, LagFit]], proxy: str, default: int) -> int:
    """The lag most districts agreed on, used for the single shaped column."""
    lags = [f[proxy].lag_weeks for f in fits.values() if proxy in f]
    if not lags:
        return default
    return int(pd.Series(lags).mode().iloc[0])


def _trailing_int(name: str) -> int:
    tail = name.rsplit("lag", 1)[-1]
    return int(tail) if tail.isdigit() else 0
