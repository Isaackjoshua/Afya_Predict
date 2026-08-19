"""Request and response models for the HTTP API.

Deliberately thin wrappers around the domain types in `src.core.types`: the API
contract is the domain contract, so a prediction served over HTTP is exactly the
object the model produced, explanation and recommendations included. There is no
"summary" endpoint that strips the reasoning — rule #2 applies at the API
boundary too.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.core.types import (
    Alert,
    DriverExplanation,
    Intervention,
    PredictionResult,
    Recommendation,
    RiskLevel,
    SourceRisk,
)


# ------------------------------------------------------------------ generic
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    environment: str
    offline_mode: bool
    diseases: List[str]
    districts: int
    backends: Dict[str, Any]
    cache: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None


# -------------------------------------------------------------- predictions
class PredictionResponse(BaseModel):
    """A forecast, with the full explanation and action package attached."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: PredictionResult


class PredictionListResponse(BaseModel):
    count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    predictions: List[PredictionResult]
    warnings: List[str] = Field(default_factory=list)


class ForecastRequest(BaseModel):
    disease: str
    districts: Optional[List[str]] = None
    horizon_weeks: Optional[int] = Field(default=None, ge=1, le=26)
    end_week: Optional[str] = None
    persist: bool = True


class ForecastRunResponse(BaseModel):
    disease: str
    districts: int
    predictions_generated: int
    alerts_generated: int
    duration_seconds: float
    warnings: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------- alerts
class AlertListResponse(BaseModel):
    count: int
    alerts: List[Alert]


class AlertAcknowledgeRequest(BaseModel):
    acknowledged_by: str = Field(min_length=1, max_length=120)
    note: Optional[str] = None


class AlertAcknowledgeResponse(BaseModel):
    alert_id: str
    acknowledged: bool
    acknowledged_by: str


# ------------------------------------------------------------- explainability
class ExplanationResponse(BaseModel):
    prediction_id: str
    disease: str
    district: str
    target_week: str
    risk_level: RiskLevel
    natural_language_explanation: str
    counterfactual: Optional[str] = None
    top_drivers: List[DriverExplanation]
    shap_values: Dict[str, float]
    source_contributions: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality_flags: List[str] = Field(default_factory=list)


# ------------------------------------------------------------- data status
class SourceStatus(BaseModel):
    source: str
    in_use: bool
    optional: bool
    configured: bool = False
    update_frequency_days: int
    last_run: Optional[str] = None
    age_hours: Optional[float] = None
    stale: bool = True
    mode: Optional[str] = None
    rows: int = 0
    mean_quality: float = 0.0
    latest_data_date: Optional[str] = None
    error: Optional[str] = None


class DataStatusResponse(BaseModel):
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    sources: List[SourceStatus]
    fused_sources: int
    meets_fusion_rule: bool
    warnings: List[str] = Field(default_factory=list)


# ----------------------------------------------------------------- diseases
class DiseaseSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    disease: str
    slug: str
    code: str
    transmission_mode: str
    proxies: List[str]
    sources: List[str]
    horizon_weeks: int
    spatial_enabled: bool
    trained: bool
    trained_at: Optional[str] = None
    district_models: int = 0
    thresholds: Dict[str, float]


class DiseaseListResponse(BaseModel):
    count: int
    diseases: List[DiseaseSummary]


# ------------------------------------------------------------- interventions
class InterventionRequest(BaseModel):
    disease: str
    district: str
    intervention_type: str
    started_week: Optional[str] = None
    ended_week: Optional[str] = None
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    quantity: Optional[float] = None
    unit: Optional[str] = None
    alert_id: Optional[str] = None
    notes: str = ""
    logged_by: str = "api"


class InterventionResponse(BaseModel):
    intervention: Intervention


class InterventionListResponse(BaseModel):
    count: int
    interventions: List[Intervention]


class ImpactResponse(BaseModel):
    intervention_id: str
    estimates: Dict[str, Optional[float]]
    confidence: str
    narrative: str
    caveats: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------- admin
class RetrainRequest(BaseModel):
    disease: Optional[str] = None
    force: bool = False
    history_weeks: int = Field(default=208, ge=52, le=1040)


class RetrainResponse(BaseModel):
    results: List[Dict[str, Any]]


class IngestRequest(BaseModel):
    sources: Optional[List[str]] = None
    start_week: Optional[str] = None
    end_week: Optional[str] = None


class BacktestResponse(BaseModel):
    disease: str
    report: Dict[str, Any]
