"""Build, deduplicate and persist structured alerts."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from config.settings import get_settings
from src.core.config_loader import load_alert_rules
from src.core.logging import get_logger
from src.core.types import Alert, DiseaseConfig, PredictionResult, RegionConfig, Recommendation

log = get_logger("alerting.generator")


def build_alert(
    prediction: PredictionResult,
    incidence_per_1000: float,
    threshold_crossed: float,
    lead_time_weeks: int,
    recommendations: Optional[Sequence[Recommendation]] = None,
    escalated: bool = False,
    escalation_reason: Optional[str] = None,
) -> Alert:
    """Assemble an `Alert` from a finished prediction."""
    low_confidence = any(
        "LOW DATA CONFIDENCE" in flag or "low data confidence" in flag.lower()
        for flag in prediction.data_quality_flags
    ) or "LOW DATA CONFIDENCE" in prediction.natural_language_explanation

    return Alert(
        alert_id=str(uuid.uuid4()),
        disease=prediction.disease,
        disease_name=prediction.disease_name,
        district=prediction.district,
        region=prediction.region,
        issued_at=datetime.utcnow(),
        target_week=prediction.target_week,
        risk_level=prediction.risk_level,
        risk_score=prediction.risk_score,
        predicted_cases=prediction.predicted_cases,
        predicted_incidence_per_1000=round(incidence_per_1000, 5),
        threshold_crossed=threshold_crossed,
        lead_time_weeks=lead_time_weeks,
        prediction_id=prediction.prediction_id,
        explanation=prediction.natural_language_explanation,
        top_drivers=list(prediction.top_drivers),
        recommendations=list(recommendations or prediction.recommendations),
        importation_risk=prediction.importation_risk,
        source_districts=list(prediction.source_districts),
        low_data_confidence=low_confidence,
        data_quality_flags=list(prediction.data_quality_flags),
        escalated=escalated,
        escalation_reason=escalation_reason,
    )


class AlertGenerator:
    """Convert predictions into alerts, applying dedup and escalation rules."""

    def __init__(
        self,
        config: DiseaseConfig,
        region: RegionConfig,
        rules: Optional[dict] = None,
        settings=None,
    ) -> None:
        self.config = config
        self.region = region
        self.rules = rules if rules is not None else load_alert_rules()
        self.settings = settings or get_settings()

        from src.alerting.recommendation_engine import RecommendationEngine
        from src.alerting.risk_classifier import RiskClassifier

        self.classifier = RiskClassifier(config.alerts, self.rules)
        self.recommender = RecommendationEngine(config, region, self.rules)

    # -- generation --------------------------------------------------------
    def generate(
        self,
        predictions: Sequence[PredictionResult],
        history: Optional[Dict[str, Sequence[float]]] = None,
        min_level: str = "medium",
    ) -> List[Alert]:
        """Classify each prediction and emit alerts at or above `min_level`."""
        from src.core.types import RISK_ORDER
        from src.alerting.escalation_rules import EscalationEngine

        floor = RISK_ORDER.index(min_level)
        escalation = EscalationEngine(self.rules)
        alerts: List[Alert] = []

        for prediction in predictions:
            try:
                population = self.region.population_of(prediction.district)
            except KeyError:
                log.warning("prediction for unknown district %s ignored", prediction.district)
                continue
            incidence = prediction.predicted_cases / max(population, 1) * 1000.0

            recent = (history or {}).get(prediction.district)
            classification = self.classifier.classify(
                incidence,
                recent_incidence=recent,
                importation_risk=prediction.importation_risk,
                ci_width_ratio=prediction.ci_width_ratio,
                input_quality=None,
            )
            if RISK_ORDER.index(classification.level) < floor:
                continue

            alert = build_alert(
                prediction,
                incidence_per_1000=incidence,
                threshold_crossed=classification.threshold_crossed,
                lead_time_weeks=prediction.top_drivers[0].lag_weeks if False else 0,
            )
            alert.risk_level = classification.level
            alert.risk_score = classification.score
            alert.low_data_confidence = alert.low_data_confidence or classification.low_confidence
            if classification.adjustments:
                alert.escalated = classification.level != classification.base_level
                alert.escalation_reason = "; ".join(classification.adjustments)
            if alert.low_data_confidence:
                banner = self.rules.get("data_confidence", {}).get(
                    "low_confidence_banner",
                    "LOW DATA CONFIDENCE — verify with field reports.",
                )
                if banner not in alert.explanation:
                    alert.explanation = f"{banner} {alert.explanation}"

            alert = escalation.apply(alert, self.load_history(alert.district, alert.disease))
            alert.recommendations = self.recommender.build(alert)
            alerts.append(alert)

        return self.deduplicate(alerts)

    # -- dedup and persistence --------------------------------------------
    def deduplicate(self, alerts: Sequence[Alert]) -> List[Alert]:
        """Suppress a repeat of an identical alert inside the configured window."""
        window_days = int(self.rules.get("deduplication", {}).get("window_days", 7))
        recent = self._recent_keys(window_days)
        out: List[Alert] = []
        seen = set()
        for alert in alerts:
            key = (alert.disease, alert.district, alert.target_week, alert.risk_level)
            if key in recent or key in seen:
                log.debug("suppressed duplicate alert %s", key)
                continue
            seen.add(key)
            out.append(alert)
        return out

    @property
    def store_path(self) -> Path:
        path = Path(self.settings.data_dir) / "alerts"
        path.mkdir(parents=True, exist_ok=True)
        return path / "alerts.jsonl"

    def persist(self, alerts: Iterable[Alert]) -> int:
        """Append alerts to the local JSONL store (offline-safe)."""
        count = 0
        with open(self.store_path, "a", encoding="utf-8") as handle:
            for alert in alerts:
                handle.write(json.dumps(alert.model_dump(), default=str) + "\n")
                count += 1
        return count

    def load_all(self, limit: Optional[int] = None) -> List[Alert]:
        if not self.store_path.exists():
            return []
        rows = []
        with open(self.store_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(Alert.model_validate(json.loads(line)))
                except Exception:  # noqa: BLE001 - a corrupt line must not break the feed
                    continue
        rows.sort(key=lambda a: a.issued_at, reverse=True)
        return rows[:limit] if limit else rows

    def load_history(self, district: str, disease: str, weeks: int = 8) -> List[Alert]:
        cutoff = datetime.utcnow() - timedelta(weeks=weeks)
        return [
            a
            for a in self.load_all()
            if a.district == district and a.disease == disease and a.issued_at >= cutoff
        ]

    def _recent_keys(self, window_days: int) -> set:
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        return {
            (a.disease, a.district, a.target_week, a.risk_level)
            for a in self.load_all()
            if a.issued_at >= cutoff
        }

    def acknowledge(self, alert_id: str, by: str) -> bool:
        """Mark an alert acknowledged, rewriting the store."""
        alerts = self.load_all()
        found = False
        for alert in alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = by
                found = True
        if found:
            with open(self.store_path, "w", encoding="utf-8") as handle:
                for alert in reversed(alerts):
                    handle.write(json.dumps(alert.model_dump(), default=str) + "\n")
        return found
