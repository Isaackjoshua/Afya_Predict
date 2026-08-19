"""Central configuration for AFYA-PREDICT.

All runtime knobs live here. Credentials are optional by design: an adapter
without credentials degrades to cache or synthetic data rather than crashing
the pipeline (critical implementation rule #1).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import List, Optional

try:  # pydantic-settings is the documented dependency
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # pragma: no cover - graceful degradation for slim installs
    from pydantic import BaseModel as BaseSettings  # type: ignore

    SettingsConfigDict = dict  # type: ignore
    _HAS_PYDANTIC_SETTINGS = False

from pydantic import Field, field_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DISEASE_CONFIG_DIR = CONFIG_DIR / "diseases"
REGION_CONFIG_DIR = CONFIG_DIR / "regions"
ALERT_RULES_DIR = CONFIG_DIR / "alert_rules"


class Settings(BaseSettings):
    """Application settings, populated from environment or `.env`."""

    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

    # --- General ---------------------------------------------------------
    app_env: str = "development"
    log_level: str = "INFO"
    default_region: str = "tanzania"
    offline_mode: bool = False

    # --- Storage ---------------------------------------------------------
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    data_dir: Path = REPO_ROOT / "data"
    artifact_dir: Path = REPO_ROOT / "artifacts"

    # --- DHIS2 -----------------------------------------------------------
    dhis2_base_url: Optional[str] = None
    dhis2_username: Optional[str] = None
    dhis2_password: Optional[str] = None

    # --- Climate sources -------------------------------------------------
    cds_api_url: str = "https://cds.climate.copernicus.eu/api"
    cds_api_key: Optional[str] = None
    chirps_base_url: str = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"
    earthdata_username: Optional[str] = None
    earthdata_password: Optional[str] = None
    cdse_client_id: Optional[str] = None
    cdse_client_secret: Optional[str] = None

    # --- Mobility --------------------------------------------------------
    cdr_data_dir: Path = REPO_ROOT / "data" / "raw" / "cdr"

    # --- Notifications ---------------------------------------------------
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    africastalking_username: Optional[str] = None
    africastalking_api_key: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email_from: str = "alerts@afya-predict.org"

    # --- API -------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: Optional[str] = None
    cors_origins: str = "*"
    rate_limit_per_minute: int = 120

    # --- Modelling defaults ---------------------------------------------
    random_seed: int = 42
    default_forecast_horizon_weeks: int = 8
    min_training_weeks: int = 104  # 24 months
    drift_check_min_residuals: int = 20
    synthetic_history_weeks: int = 260  # 5 years of fallback history

    @field_validator("data_dir", "artifact_dir", "cdr_data_dir", mode="before")
    @classmethod
    def _expand(cls, value):
        if value in (None, ""):
            return value
        return Path(str(value)).expanduser()

    # --- Derived helpers -------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "afya.sqlite"

    @property
    def effective_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.sqlite_path}"

    @property
    def cors_origin_list(self) -> List[str]:
        if self.cors_origins.strip() in ("*", ""):
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def has_dhis2(self) -> bool:
        return bool(self.dhis2_base_url and self.dhis2_username and self.dhis2_password)

    def has_cds(self) -> bool:
        return bool(self.cds_api_key)

    def has_earthdata(self) -> bool:
        return bool(self.earthdata_username and self.earthdata_password)

    def has_cdse(self) -> bool:
        return bool(self.cdse_client_id and self.cdse_client_secret)

    def ensure_dirs(self) -> None:
        """Create the directories the pipeline writes to."""
        for path in (
            self.data_dir,
            self.cache_dir,
            self.raw_dir,
            self.processed_dir,
            self.artifact_dir,
            self.cdr_data_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
