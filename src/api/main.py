"""FastAPI application entry point.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Design notes worth knowing before extending this:

* **Reads are cache-backed.** `GET /predictions/...` reads precomputed forecasts
  from the local SQLite store, so it stays well inside the 2-second budget
  (acceptance criterion #10). Anything that trains or ingests lives under an
  explicit `POST` and says so.
* **Explanations are never stripped.** Every prediction served carries its SHAP
  drivers, natural-language summary and recommendations (critical rule #2).
* **It starts without data.** A fresh install with no credentials, no cache and
  no trained models still boots, serves `/health` and reports honestly what is
  missing, rather than failing at import time.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from config.settings import get_settings
from src import __version__
from src.api.middleware import install_middleware
from src.api.routes import (
    admin,
    alerts,
    data_status,
    diseases,
    explainability,
    interventions,
    predictions,
)
from src.api.schemas import HealthResponse
from src.core.logging import configure_logging, get_logger

log = get_logger("api")

DESCRIPTION = """
**AFYA-PREDICT** — open, explainable outbreak forecasting for East Africa and beyond.

Fuses satellite climate, mobility, population, air-quality, WASH and routine
health-surveillance data onto a `district x epidemiological week` grid and
forecasts where and when disease risk will rise, typically 4-12 weeks ahead.

Every prediction returned by this API includes:

* a 95% confidence interval,
* SHAP attributions naming the digital proxies that drove it, with their fitted
  lags and the mechanism each represents,
* a plain-language explanation and at least one counterfactual,
* importation risk and the source districts it is arriving from,
* costed, timeboxed response recommendations with an owner.

MIT licensed. No vendor lock-in, no per-seat cost.
"""

app = FastAPI(
    title="AFYA-PREDICT API",
    description=DESCRIPTION,
    version=__version__,
    contact={"name": "AFYA-PREDICT", "url": "https://github.com/Isaackjoshua/Afya_Predict"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)

install_middleware(app)
app.include_router(predictions.router)
app.include_router(alerts.router)
app.include_router(data_status.router)
app.include_router(explainability.router)
app.include_router(diseases.router)
app.include_router(interventions.router)
app.include_router(admin.router)


@app.on_event("startup")
def on_startup() -> None:
    configure_logging()
    settings = get_settings()
    log.info(
        "AFYA-PREDICT %s starting (env=%s, offline=%s, region=%s)",
        __version__, settings.app_env, settings.offline_mode, settings.default_region,
    )


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "AFYA-PREDICT",
        "version": __version__,
        "description": "Explainable, multi-source outbreak forecasting",
        "docs": "/docs",
        "health": "/health",
        "license": "MIT",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness plus an honest account of what this instance can currently do."""
    from src.api.dependencies import get_cache, get_region
    from src.models.backends import backend_report
    from src.models.registry import list_modules

    settings = get_settings()
    warnings: List[str] = []

    try:
        region = get_region()
        district_count = len(region.districts)
    except Exception as exc:  # noqa: BLE001
        district_count = 0
        warnings.append(f"region config unavailable: {exc}")

    try:
        cache_status = get_cache().status()
        if not cache_status.get("offline_ready"):
            warnings.append(
                "fewer than 2 weeks of forecasts cached; this node is not yet ready to run offline"
            )
    except Exception as exc:  # noqa: BLE001
        cache_status = {"error": str(exc)}
        warnings.append(f"local cache unavailable: {exc}")

    backends = backend_report()
    if backends["default_resolved"] == "numpy_gbm":
        warnings.append(
            "no XGBoost/LightGBM/scikit-learn installed; using the bundled NumPy gradient "
            "booster. Training works but is slower."
        )

    return HealthResponse(
        status="ok" if not warnings else "degraded",
        version=__version__,
        environment=settings.app_env,
        offline_mode=settings.offline_mode,
        diseases=list_modules(),
        districts=district_count,
        backends=backends,
        cache=cache_status,
        warnings=warnings,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """Never leak a stack trace to a client; log it and return a clean error."""
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "internal_error"},
    )


def main() -> None:  # pragma: no cover - process entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app", host=settings.api_host, port=settings.api_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
