"""Record what was actually done in response to an alert.

Without this record, impact estimation is impossible and the platform is just
another alert generator. The logger deliberately captures **coverage and
timing**, not just "an intervention happened": 5,000 nets delivered to 12% of a
district three weeks late is a different exposure from 50,000 delivered to 80%
on time, and the two should not be scored as the same event.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from src.core.logging import get_logger
from src.core.timeutils import to_epi_week, weeks_between
from src.core.types import Alert, Intervention

log = get_logger("intervention.logger")

#: Standard intervention types with the lag before an effect is plausible.
#: Used to place the evaluation window; an estimate taken before the lag has
#: elapsed would score the intervention on weeks it could not have influenced.
INTERVENTION_TYPES: Dict[str, Dict[str, object]] = {
    "llin_distribution": {"label": "LLIN (bed net) distribution", "effect_lag_weeks": 3, "disease": "malaria"},
    "irs_spraying": {"label": "Indoor residual spraying", "effect_lag_weeks": 2, "disease": "malaria"},
    "act_prepositioning": {"label": "ACT antimalarial pre-positioning", "effect_lag_weeks": 1, "disease": "malaria"},
    "ors_distribution": {"label": "ORS distribution", "effect_lag_weeks": 1, "disease": "cholera"},
    "water_chlorination": {"label": "Water point chlorination", "effect_lag_weeks": 1, "disease": "cholera"},
    "ocv_campaign": {"label": "Oral cholera vaccination campaign", "effect_lag_weeks": 2, "disease": "cholera"},
    "treatment_centre": {"label": "Cholera treatment centre established", "effect_lag_weeks": 1, "disease": "cholera"},
    "active_case_finding": {"label": "Active TB case finding", "effect_lag_weeks": 4, "disease": "tuberculosis"},
    "oxygen_prepositioning": {"label": "Oxygen pre-positioning", "effect_lag_weeks": 1, "disease": "respiratory"},
    "mobile_hts": {"label": "Mobile HIV testing services", "effect_lag_weeks": 4, "disease": "hiv"},
    "risk_communication": {"label": "Community risk communication", "effect_lag_weeks": 1, "disease": None},
    "other": {"label": "Other response action", "effect_lag_weeks": 2, "disease": None},
}


class InterventionLogger:
    """Create and retrieve intervention records."""

    def __init__(self, cache=None, settings=None) -> None:
        from offline.local_cache import LocalCache

        self.cache = cache or LocalCache(settings=settings)

    # -- writing -----------------------------------------------------------
    def log(
        self,
        disease: str,
        district: str,
        intervention_type: str,
        started_week: Optional[str] = None,
        ended_week: Optional[str] = None,
        coverage: float = 0.0,
        quantity: Optional[float] = None,
        unit: Optional[str] = None,
        alert_id: Optional[str] = None,
        notes: str = "",
        logged_by: str = "system",
    ) -> Intervention:
        """Record one response action."""
        if intervention_type not in INTERVENTION_TYPES:
            log.info(
                "unknown intervention type %r recorded as 'other'; add it to "
                "INTERVENTION_TYPES if it becomes routine", intervention_type,
            )
        intervention = Intervention(
            intervention_id=str(uuid.uuid4()),
            disease=disease,
            district=district,
            alert_id=alert_id,
            intervention_type=intervention_type,
            started_week=started_week or to_epi_week(datetime.utcnow().date()),
            ended_week=ended_week,
            coverage=float(min(max(coverage, 0.0), 1.0)),
            quantity=quantity,
            unit=unit,
            notes=notes,
            logged_by=logged_by,
        )
        self.cache.save_intervention(intervention)
        log.info(
            "logged %s in %s (%s, coverage %.0f%%)",
            intervention_type, district, intervention.started_week, intervention.coverage * 100,
        )
        return intervention

    def log_from_alert(
        self,
        alert: Alert,
        intervention_type: str,
        coverage: float = 0.0,
        quantity: Optional[float] = None,
        unit: Optional[str] = None,
        logged_by: str = "system",
        notes: str = "",
    ) -> Intervention:
        """Log a response and keep the link back to the alert that prompted it."""
        return self.log(
            disease=alert.disease,
            district=alert.district,
            intervention_type=intervention_type,
            started_week=to_epi_week(datetime.utcnow().date()),
            coverage=coverage,
            quantity=quantity,
            unit=unit,
            alert_id=alert.alert_id,
            notes=notes or f"Response to {alert.risk_level} alert for week {alert.target_week}",
            logged_by=logged_by,
        )

    # -- reading -----------------------------------------------------------
    def list(
        self, disease: Optional[str] = None, district: Optional[str] = None, limit: int = 200
    ) -> List[Intervention]:
        return self.cache.get_interventions(disease=disease, district=district, limit=limit)

    def for_alert(self, alert_id: str) -> List[Intervention]:
        return [i for i in self.cache.get_interventions(limit=1000) if i.alert_id == alert_id]

    # -- analysis helpers --------------------------------------------------
    @staticmethod
    def effect_lag(intervention_type: str) -> int:
        record = INTERVENTION_TYPES.get(intervention_type, INTERVENTION_TYPES["other"])
        return int(record["effect_lag_weeks"])  # type: ignore[index]

    def response_time(self, alert: Alert, intervention: Intervention) -> int:
        """Weeks between an alert being issued and the response starting.

        The core operational metric: a system with 8 weeks of lead time that
        gets a 9-week response has delivered nothing.
        """
        return weeks_between(to_epi_week(alert.issued_at.date()), intervention.started_week)

    def coverage_summary(self, disease: str) -> Dict[str, object]:
        """How much of the recommended response actually happened."""
        interventions = self.list(disease=disease, limit=1000)
        if not interventions:
            return {"disease": disease, "n": 0, "mean_coverage": None, "by_type": {}}
        by_type: Dict[str, List[float]] = {}
        for item in interventions:
            by_type.setdefault(item.intervention_type, []).append(item.coverage)
        return {
            "disease": disease,
            "n": len(interventions),
            "mean_coverage": round(
                sum(i.coverage for i in interventions) / len(interventions), 4
            ),
            "districts": sorted({i.district for i in interventions}),
            "by_type": {
                key: {"n": len(values), "mean_coverage": round(sum(values) / len(values), 4)}
                for key, values in by_type.items()
            },
        }
