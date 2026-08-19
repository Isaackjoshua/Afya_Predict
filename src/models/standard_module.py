"""The reference implementation of `BaseDiseaseModule`.

Every shipped disease inherits from this. It implements the full contract —
feature building, per-district training, explained prediction, outbreak
detection, spatial risk and recommendations — so a concrete disease module only
overrides what is genuinely disease-specific.

That asymmetry is the point of shortcoming #11: adding dengue should be a
config file and a short class, not a new system.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.core.timeutils import shift_week, to_epi_week
from src.core.types import (
    Alert,
    DriverExplanation,
    FeatureSpec,
    PredictionResult,
    Recommendation,
    RiskLevel,
    SourceRisk,
)
from src.data_ingestion.normalizer import Panel
from src.explainability.counterfactuals import headline_counterfactual
from src.explainability.natural_language import explain_prediction
from src.explainability.shap_explainer import ShapExplainer
from src.feature_engineering.builder import FeatureMatrix
from src.feature_engineering.mobility_features import get_travel_matrix, top_source_districts
from src.models.base_model import BaseDiseaseModule

#: Default response timeframe per severity, in days.
TIMEFRAME_DAYS: Dict[str, int] = {"low": 30, "medium": 21, "high": 7, "critical": 3}

#: Who owns the response at each level.
RESPONSIBLE: Dict[str, str] = {
    "low": "District Health Management Team",
    "medium": "District Health Management Team",
    "high": "Regional Health Management Team",
    "critical": "National Emergency Operations Centre",
}


class StandardDiseaseModule(BaseDiseaseModule):
    """Config-driven disease module covering the common case."""

    # -- interface: configuration -----------------------------------------
    def get_feature_config(self) -> List[FeatureSpec]:
        """The digital proxies and lag ranges declared for this disease."""
        return list(self.config.digital_proxies)

    def build_feature_matrix(
        self, panel: Panel, horizon_weeks: Optional[int] = None
    ) -> FeatureMatrix:
        matrix = self.builder.build(panel, horizon_weeks=horizon_weeks)
        self.feature_matrix = matrix
        return matrix

    # -- interface: training ----------------------------------------------
    def train(self, matrix: FeatureMatrix, districts: Optional[Sequence[str]] = None) -> None:
        self.fit_models(matrix, districts=districts)

    # -- interface: prediction --------------------------------------------
    def predict(
        self,
        matrix: FeatureMatrix,
        district: str,
        horizon_weeks: Optional[int] = None,
        n_weeks: int = 1,
        panel: Optional[Panel] = None,
        travel_matrix: Optional[pd.DataFrame] = None,
    ) -> List[PredictionResult]:
        """Forecast `district`, with SHAP, counterfactual and recommendations.

        Critical rule #2 is enforced structurally: the explanation is built in
        the same call that builds the prediction, so a `PredictionResult`
        without drivers cannot be constructed by this path.
        """
        horizon = horizon_weeks if horizon_weeks is not None else self.horizon
        model = self.model_for(district)
        if model is None:
            raise RuntimeError(f"{self.config.name}: no model trained for {district}")

        rows = self.latest_feature_rows(matrix, district, n_weeks=n_weeks)
        if rows.empty:
            return []

        bundle = self._predict_rows(matrix, district, rows)

        background = matrix.for_district(district).X.dropna(how="all")
        if len(background) < 20:
            background = matrix.X.dropna(how="all")
        explainer = ShapExplainer(
            model, background=background, provenance=matrix.provenance
        )
        explanation = explainer.explain(rows)

        importation_risk, source_districts = (0.0, [])
        if self.config.spatial.enabled:
            importation_risk, source_districts = self.get_spatial_risk(
                district, travel_matrix if travel_matrix is not None else get_travel_matrix(self.region),
                panel=panel, week=str(rows.index[-1][1]),
            )

        district_meta = self.region.get(district)
        quality_flags = self._quality_flags(matrix, panel, rows)
        low_confidence = self._is_low_confidence(bundle, quality_flags)

        results: List[PredictionResult] = []
        for position, (index, _) in enumerate(rows.iterrows()):
            feature_week = str(index[1])
            target_week = shift_week(feature_week, horizon)
            point = float(bundle.point.iloc[position])
            lower = float(bundle.lower.iloc[position])
            upper = float(bundle.upper.iloc[position])

            incidence = self.incidence_per_1000(point, district)
            level = self.adjust_risk_level(
                self.classify_risk(incidence), incidence, importation_risk, low_confidence
            )
            drivers = explanation.top_drivers(position=position)

            counterfactual = headline_counterfactual(
                model,
                rows.iloc[position],
                drivers,
                classify_risk=self.classify_risk,
                incidence_fn=lambda cases: self.incidence_per_1000(cases, district),
                provenance=matrix.provenance,
            )

            narrative = explain_prediction(
                disease=self.config.name,
                district=district,
                target_week=target_week,
                predicted_cases=point,
                incidence_per_1000=incidence,
                risk_level=level,
                drivers=drivers,
                lead_time_weeks=horizon,
                ci=(lower, upper),
                importation_risk=importation_risk,
                source_districts=source_districts,
                data_quality_flags=quality_flags,
                low_confidence=low_confidence,
            )

            result = PredictionResult(
                prediction_id=self.new_prediction_id(),
                disease=self.config.name,
                district=district,
                region=district_meta.region,
                forecast_date=date.today(),
                target_week=target_week,
                predicted_cases=round(point, 2),
                confidence_interval_lower=round(lower, 2),
                confidence_interval_upper=round(upper, 2),
                risk_level=level,
                risk_score=round(self.risk_score(incidence), 4),
                shap_values=explanation.as_dict(position=position),
                top_drivers=drivers,
                natural_language_explanation=narrative,
                counterfactual=counterfactual,
                importation_risk=round(importation_risk, 4),
                source_districts=source_districts,
                model_version=bundle.model_version,
                data_freshness=self._freshness(panel),
                data_quality_flags=quality_flags,
            )
            result.recommendations = self.generate_recommendations(
                self._provisional_alert(result)
            )
            results.append(result)
        return results

    def predict_all(
        self,
        matrix: FeatureMatrix,
        districts: Optional[Sequence[str]] = None,
        panel: Optional[Panel] = None,
        **kwargs,
    ) -> List[PredictionResult]:
        """Forecast every district, sharing one travel matrix across the run."""
        travel = get_travel_matrix(self.region) if self.config.spatial.enabled else None
        out: List[PredictionResult] = []
        for district in districts or matrix.districts:
            try:
                out.extend(
                    self.predict(matrix, district, panel=panel, travel_matrix=travel, **kwargs)
                )
            except Exception as exc:  # noqa: BLE001 - one district must not stop the run
                self.log.warning("prediction failed for %s: %s", district, exc)
        return out

    # -- interface: alerting ----------------------------------------------
    def detect_outbreak(
        self,
        predictions: Sequence[PredictionResult],
        actuals: Optional[pd.Series] = None,
    ) -> List[Alert]:
        """Raise an alert for every prediction at or above the `low` threshold."""
        from src.alerting.alert_generator import build_alert

        alerts: List[Alert] = []
        for prediction in predictions:
            incidence = self.incidence_per_1000(prediction.predicted_cases, prediction.district)
            if incidence < self.config.alerts.low:
                continue
            alert = build_alert(
                prediction,
                incidence_per_1000=incidence,
                threshold_crossed=self._threshold_for(prediction.risk_level),
                lead_time_weeks=self.horizon,
                recommendations=self.generate_recommendations(self._provisional_alert(prediction)),
            )
            alerts.append(alert)
        return alerts

    # -- interface: spatial ------------------------------------------------
    def get_spatial_risk(
        self,
        district: str,
        travel_matrix: pd.DataFrame,
        panel: Optional[Panel] = None,
        week: Optional[str] = None,
    ) -> Tuple[float, List[SourceRisk]]:
        """Importation risk 0-1 and the districts contributing it."""
        if not self.config.spatial.enabled:
            return 0.0, []
        source = panel.values() if panel is not None else (
            self.feature_matrix.X if self.feature_matrix is not None else None
        )
        if source is None or self.target_column not in getattr(source, "columns", []):
            return 0.0, []

        weeks = sorted({str(w) for w in source.index.get_level_values("week")})
        target_week = week if week in weeks else (weeks[-1] if weeks else None)
        if target_week is None:
            return 0.0, []

        ranked = top_source_districts(
            source, self.region, district, target_week, self.target_column,
            travel_matrix=travel_matrix,
        )
        if ranked.empty:
            return 0.0, []

        sources = [
            SourceRisk(
                district=row["district"],
                flow_weight=float(row["flow_weight"]),
                active_cases=float(row["active_cases"]),
                contributed_risk=float(row["contributed_risk"]),
            )
            for _, row in ranked.iterrows()
        ]

        # Absolute pressure: flow-weighted incidence arriving from elsewhere,
        # scaled against this disease's own `high` threshold so the number is
        # comparable across diseases.
        populations = {d.name: float(d.population) for d in self.region.districts}
        pressure = 0.0
        for item in sources:
            population = populations.get(item.district, 1.0)
            incidence = item.active_cases / max(population, 1.0) * 1000.0
            pressure += item.flow_weight * incidence
        reference = max(self.config.alerts.high, 1e-6)
        normalised = float(np.tanh(pressure / reference))
        return round(normalised * self.config.spatial.importation_weight * 2.0, 4), sources

    # -- interface: recommendations ---------------------------------------
    def generate_recommendations(self, alert: Alert) -> List[Recommendation]:
        """Config-driven response actions, escalating with severity.

        Rule #9: these come from YAML, not code, so health officials can adapt
        them to their own IDSR guidelines without a release.
        """
        from src.alerting.recommendation_engine import RecommendationEngine

        return RecommendationEngine(self.config, self.region).build(alert)

    # -- hooks a concrete disease may override -----------------------------
    def adjust_risk_level(
        self,
        level: RiskLevel,
        incidence: float,
        importation_risk: float,
        low_confidence: bool,
    ) -> RiskLevel:
        """Post-process the threshold classification.

        Default behaviour: a district facing strong importation pressure is
        escalated one level even if its own counts are still flat (that is the
        whole point of predicting spread), while low input confidence pulls the
        level back one step so the system does not over-claim on weak data.
        """
        from src.core.types import RISK_ORDER

        index = RISK_ORDER.index(level)
        if importation_risk >= 0.7 and index < len(RISK_ORDER) - 1:
            index += 1
        if low_confidence and index > 0:
            index -= 1
        return RISK_ORDER[index]  # type: ignore[return-value]

    # -- internals ---------------------------------------------------------
    def _threshold_for(self, level: RiskLevel) -> float:
        return {
            "low": self.config.alerts.low,
            "medium": self.config.alerts.medium,
            "high": self.config.alerts.high,
            "critical": self.config.alerts.critical,
        }[level]

    def _provisional_alert(self, prediction: PredictionResult) -> Alert:
        """A lightweight alert used only to select recommendation templates."""
        incidence = self.incidence_per_1000(prediction.predicted_cases, prediction.district)
        return Alert(
            alert_id="provisional",
            disease=prediction.disease,
            district=prediction.district,
            region=prediction.region,
            issued_at=datetime.utcnow(),
            target_week=prediction.target_week,
            risk_level=prediction.risk_level,
            risk_score=prediction.risk_score,
            predicted_cases=prediction.predicted_cases,
            predicted_incidence_per_1000=incidence,
            threshold_crossed=self._threshold_for(prediction.risk_level),
            lead_time_weeks=self.horizon,
            prediction_id=prediction.prediction_id,
            top_drivers=prediction.top_drivers,
            importation_risk=prediction.importation_risk,
            source_districts=prediction.source_districts,
            low_data_confidence=bool(
                prediction.data_quality_flags
                and any("LOW DATA CONFIDENCE" in f for f in prediction.data_quality_flags)
            ),
            data_quality_flags=prediction.data_quality_flags,
        )

    def _quality_flags(
        self, matrix: FeatureMatrix, panel: Optional[Panel], rows: pd.DataFrame
    ) -> List[str]:
        flags: List[str] = []
        if panel is not None:
            for flag in panel.flags:
                if flag.severity in ("warning", "error"):
                    flags.append(str(flag))
        if matrix.row_quality is not None:
            quality = float(matrix.row_quality.reindex(rows.index).mean())
            if quality < 0.6:
                flags.append(
                    f"mean input quality for these weeks is {quality:.2f} (below 0.6)"
                )
        model = self.model_for(rows.index[0][0]) if len(rows) else None
        if model is not None and model.scope == "pooled":
            flags.append(
                "no district-specific model: this forecast uses the pooled model "
                "because local history is too short"
            )
        return flags[:8]

    def _is_low_confidence(self, bundle, flags: Sequence[str]) -> bool:
        """Rule #7: wide intervals or poor inputs must be stated, not hidden."""
        point = float(bundle.point.iloc[-1])
        width = float(bundle.upper.iloc[-1] - bundle.lower.iloc[-1])
        wide = point > 0 and width / point > 1.5
        poor = float(bundle.quality.iloc[-1]) < 0.6
        errors = any("error" in f.lower() or "insufficient" in f.lower() for f in flags)
        return bool(wide or poor or errors)

    def _freshness(self, panel: Optional[Panel]) -> Dict[str, date]:
        if panel is None:
            return {}
        return {
            source: value
            for source, value in panel.freshness.items()
            if isinstance(value, date)
        }
