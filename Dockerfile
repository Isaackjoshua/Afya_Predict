# AFYA-PREDICT application image.
#
# One image serves all four roles (api, dashboard, scheduler, one-off jobs); the
# compose file selects the role by overriding the command. That keeps the build
# cache warm and means a district deployment pulls a single artefact.
FROM python:3.11-slim-bookworm AS base

# Runtime deps only. The geospatial extras (rasterio/geopandas) are deliberately
# NOT installed here: they pull in ~400 MB of GDAL, and the platform is designed
# to run without them (see src/data_ingestion/raster_utils.py). Build the `geo`
# target below if your deployment does live raster ingestion.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # This platform is deployed over links that drop mid-download. Without
    # generous retries a build fails with `error: incomplete-download` on a
    # 10 MB wheel, which is exactly the condition the target deployments live in.
    PIP_RETRIES=10 \
    PIP_TIMEOUT=120

WORKDIR /app

# Dependency layer first, so editing source does not re-download every wheel.
# `--no-deps .` on the second install is what makes this correct: the first pass
# resolves and caches the dependency tree from an empty package skeleton, and the
# second installs the real code without re-resolving. Installing the project in
# the first pass instead would silently ship a package missing config/, dashboard/
# and offline/, which only shows up at runtime.
# EXTRAS controls how heavy the image is. The default keeps it light and
# buildable on a poor link; `--build-arg EXTRAS=ml,dashboard` adds XGBoost,
# LightGBM, SHAP and scikit-learn (roughly 400 MB more, mostly llvmlite) for a
# server deployment where the faster backends are worth it.
ARG EXTRAS=dashboard
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src && touch src/__init__.py \
    && pip install --upgrade pip \
    && pip install ".[${EXTRAS}]" \
    && rm -rf src

COPY config/ config/
COPY src/ src/
COPY dashboard/ dashboard/
COPY offline/ offline/
COPY scripts/ scripts/
RUN pip install --no-deps .

# Data and artefacts live on volumes so a container restart never loses the
# local cache an offline node depends on.
RUN mkdir -p /app/data /app/artifacts \
    && useradd --create-home --uid 10001 afya \
    && chown -R afya:afya /app
USER afya

ENV DATA_DIR=/app/data \
    ARTIFACT_DIR=/app/artifacts \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# tini reaps the child processes APScheduler and uvicorn workers leave behind.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# --- optional geospatial variant -------------------------------------------
# Build with:  docker build --target geo -t afya-predict:geo .
# Only needed when ingesting CHIRPS/WorldPop rasters directly rather than
# through pre-extracted district tables.
FROM base AS geo
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gdal-bin libgdal-dev g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install ".[geo]" \
    && apt-get purge -y g++ && apt-get autoremove -y
USER afya


# --- default target ---------------------------------------------------------
# Docker builds the LAST stage when none is named, so this alias has to come
# after `geo`. Without it, a plain `docker build .` (and `docker compose build`)
# would silently produce the ~400 MB GDAL image instead of the light one.
FROM base AS runtime
