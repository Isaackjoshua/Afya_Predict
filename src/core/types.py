"""Core domain types shared across every layer of AFYA-PREDICT.

These are the contracts that keep the disease modules pluggable: an adapter
produces `TidyFrame`-shaped data, a disease module consumes `FeatureSpec`s and
emits `PredictionResult`s, and the alerting layer turns those into `Alert`s
with `Recommendation`s attached.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["low", "medium", "high", "critical"]
RISK_ORDER: Tuple[str, ...] = ("low", "medium", "high", "critical")


class Relationship(str, Enum):
    """Shape of the dose-response between a digital proxy and transmission."""

    POSITIVE_LINEAR = "positive_linear"
    NEGATIVE_LINEAR = "negative_linear"
    BELL_CURVE = "bell_curve"
    POSITIVE_WITH_SATURATION = "positive_with_saturation"
    THRESHOLD = "threshold"


class TransmissionMode(str, Enum):
    VECTOR_BORNE = "vector_borne"
    WATERBORNE = "waterborne"
    AIRBORNE = "airborne"
    SEXUAL = "sexual"
    ZOONOTIC = "zoonotic"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Data-quality
# ---------------------------------------------------------------------------
class QualityFlag(BaseModel):
    """A single data-quality finding attached to a source/district/week."""

    code: str
    severity: Literal["info", "warning", "error"] = "warning"
    message: str
    source: Optional[str] = None
    district: Optional[str] = None
    affected_rows: int = 0

    def __str__(self) -> str:  # pragma: no cover - display helper
        scope = self.source or "pipeline"
        return f"[{self.severity.upper()}] {scope}: {self.message}"


# ---------------------------------------------------------------------------
# Disease configuration
# ---------------------------------------------------------------------------
class FeatureSpec(BaseModel):
    """One digital proxy and the lag structure to search for it."""

    model_config = ConfigDict(extra="allow")

    name: str
    source: str
    relationship: Relationship = Relationship.POSITIVE_LINEAR
    lag_weeks_range: Tuple[int, int] = (0, 0)
    optimal_lag_weeks: int = 0
    mechanism: str = ""
    spatial: bool = False
    optional: bool = False

    @property
    def lag_candidates(self) -> List[int]:
        """Every lag the fitter is allowed to try for this proxy."""
        low, high = self.lag_weeks_range
        low, high = int(min(low, high)), int(max(low, high))
        return list(range(low, high + 1))

    def params(self) -> Dict[str, Any]:
        """Proxy-specific extras from YAML (thresholds, optimal ranges, ...)."""
        known = set(type(self).model_fields)
        extra = self.model_extra or {}
        return {k: v for k, v in extra.items() if k not in known}


class SpatialConfig(BaseModel):
    enabled: bool = False
    diffusion_model: Literal["gravity", "radiation", "cdr_empirical"] = "gravity"
    importation_weight: float = 0.0


class ModelConfig(BaseModel):
    primary: str = "xgboost"
    ensemble_members: List[str] = Field(default_factory=list)
    forecast_horizon_weeks: int = 4
    retrain_frequency: str = "monthly"
    min_training_months: int = 24

    @property
    def min_training_weeks(self) -> int:
        return int(round(self.min_training_months * 4.345))


class AlertThresholds(BaseModel):
    """Cases per 1,000 population per week at which each level triggers."""

    low: float = 0.0
    medium: float = 0.0
    high: float = 0.0
    critical: float = 0.0

    def as_ordered(self) -> List[Tuple[str, float]]:
        return [
            ("critical", self.critical),
            ("high", self.high),
            ("medium", self.medium),
            ("low", self.low),
        ]


class DiseaseConfig(BaseModel):
    """Parsed `config/diseases/<name>.yaml`."""

    #: The config filename stem, set by the loader. This is the disease's
    #: identity everywhere — registry key, module slug, `cases_*` column — and
    #: it is NOT derivable from `name`: "Acute Respiratory Infection" lives in
    #: `respiratory.yaml` and its surveillance column is `cases_respiratory`.
    config_slug: Optional[str] = None
    name: str
    code: str
    transmission_mode: TransmissionMode = TransmissionMode.OTHER
    vector: str = ""
    pathogen: str = ""
    digital_proxies: List[FeatureSpec] = Field(default_factory=list)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    alerts: AlertThresholds = Field(default_factory=AlertThresholds)
    recommendations: Dict[str, List[str]] = Field(default_factory=dict)

    @property
    def slug(self) -> str:
        """The disease's canonical identifier.

        Prefers the config filename, falling back to a name-derived slug only
        for configs built in memory (tests, scaffolding).
        """
        return self.config_slug or self.name.lower().replace(" ", "_")

    @property
    def required_sources(self) -> List[str]:
        """Distinct adapter names this disease needs (optional ones excluded)."""
        seen: List[str] = []
        for proxy in self.digital_proxies:
            if proxy.optional:
                continue
            if proxy.source not in seen:
                seen.append(proxy.source)
        return seen

    @property
    def all_sources(self) -> List[str]:
        seen: List[str] = []
        for proxy in self.digital_proxies:
            if proxy.source not in seen:
                seen.append(proxy.source)
        return seen


# ---------------------------------------------------------------------------
# Region configuration
# ---------------------------------------------------------------------------
class District(BaseModel):
    name: str
    region: str
    lat: float
    lon: float
    population: int
    urban: bool = False
    wash_access: float = 0.5
    density_km2: float = 100.0


class BoundingBox(BaseModel):
    west: float
    south: float
    east: float
    north: float


class RegionConfig(BaseModel):
    """Parsed `config/regions/<region>.yaml`."""

    name: str
    code: str
    iso3: str = ""
    timezone: str = "UTC"
    epi_week_system: str = "iso8601"
    crs: str = "EPSG:4326"
    admin_level: str = "district"
    geojson_path: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    reporting_completeness: float = 0.9
    weather_stations: int = 0
    districts: List[District] = Field(default_factory=list)

    @property
    def district_names(self) -> List[str]:
        return [d.name for d in self.districts]

    def get(self, name: str) -> District:
        for d in self.districts:
            if d.name == name:
                return d
        raise KeyError(f"Unknown district: {name!r}")

    def population_of(self, name: str) -> int:
        return self.get(name).population


# ---------------------------------------------------------------------------
# Predictions, explanations, alerts
# ---------------------------------------------------------------------------
class DriverExplanation(BaseModel):
    """One feature's contribution to a prediction, in plain language."""

    feature: str
    proxy: str = ""
    lag_weeks: int = 0
    value: Optional[float] = None
    shap_value: float = 0.0
    contribution_share: float = 0.0  # |shap| / sum(|shap|)
    direction: Literal["increases", "decreases", "neutral"] = "neutral"
    mechanism: str = ""
    narrative: str = ""


class SourceRisk(BaseModel):
    """A district contributing importation pressure to the target district."""

    district: str
    flow_weight: float = 0.0
    active_cases: float = 0.0
    contributed_risk: float = 0.0


class Recommendation(BaseModel):
    action: str
    priority: RiskLevel = "medium"
    timeframe_days: int = 14
    responsible: str = "District Health Management Team"
    quantity: Optional[str] = None
    rationale: str = ""


class PredictionResult(BaseModel):
    """The full, explainable, actionable output of a single forecast."""

    prediction_id: str
    disease: str
    district: str
    region: str
    forecast_date: date
    target_week: str
    predicted_cases: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    risk_level: RiskLevel
    risk_score: float

    # Explainability (shortcoming #4)
    shap_values: Dict[str, float] = Field(default_factory=dict)
    top_drivers: List[DriverExplanation] = Field(default_factory=list)
    natural_language_explanation: str = ""
    counterfactual: Optional[str] = None

    # Spatial (shortcoming #10)
    importation_risk: float = 0.0
    source_districts: List[SourceRisk] = Field(default_factory=list)

    # Actionable output (shortcoming #12)
    recommendations: List[Recommendation] = Field(default_factory=list)

    # Metadata
    model_version: str = "unknown"
    data_freshness: Dict[str, date] = Field(default_factory=dict)
    data_quality_flags: List[str] = Field(default_factory=list)

    model_config = ConfigDict(protected_namespaces=())

    @property
    def incidence_per_1000(self) -> Optional[float]:
        return getattr(self, "_incidence", None)

    @property
    def ci_width_ratio(self) -> float:
        if self.predicted_cases <= 0:
            return float("inf") if self.confidence_interval_upper > 0 else 0.0
        return (self.confidence_interval_upper - self.confidence_interval_lower) / self.predicted_cases


class Alert(BaseModel):
    """A dispatched, actionable warning derived from one or more predictions."""

    alert_id: str
    disease: str
    district: str
    region: str
    issued_at: datetime
    target_week: str
    risk_level: RiskLevel
    risk_score: float
    predicted_cases: float
    predicted_incidence_per_1000: float
    threshold_crossed: float
    lead_time_weeks: int = 0
    prediction_id: Optional[str] = None
    explanation: str = ""
    top_drivers: List[DriverExplanation] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    importation_risk: float = 0.0
    source_districts: List[SourceRisk] = Field(default_factory=list)
    low_data_confidence: bool = False
    data_quality_flags: List[str] = Field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    delivery_status: Dict[str, str] = Field(default_factory=dict)


class Intervention(BaseModel):
    """A logged response action, used to close the feedback loop (#15)."""

    intervention_id: str
    disease: str
    district: str
    alert_id: Optional[str] = None
    intervention_type: str
    started_week: str
    ended_week: Optional[str] = None
    coverage: float = 0.0  # 0-1 share of target population reached
    quantity: Optional[float] = None
    unit: Optional[str] = None
    notes: str = ""
    logged_by: str = "system"
    logged_at: datetime = Field(default_factory=datetime.utcnow)
