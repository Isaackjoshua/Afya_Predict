# Deployment

Three shapes, depending on what you are running.

| Shape | Runs | Storage | Use |
|---|---|---|---|
| **Single node** | api + dashboard | SQLite | one district or a pilot |
| **Central** | api + dashboard + scheduler + db + redis | PostgreSQL/TimescaleDB | national instance |
| **Offline edge** | api only, or the distilled model | SQLite | intermittent connectivity |

## Quick start

```bash
git clone https://github.com/Isaackjoshua/Afya_Predict.git
cd Afya_Predict
cp .env.example .env          # optional — the stack runs without credentials
docker compose up --build
```

> **The build needs a working link.** The default image is roughly 1.2 GB built
> (Streamlit pulls PyArrow, and SciPy/pandas/NumPy are not small). Build with
> `--build-arg EXTRAS=` for an API-only image without the dashboard stack, or
> `EXTRAS=ml,dashboard` for the faster model backends. `PIP_RETRIES`/`PIP_TIMEOUT` are set
> generously in the Dockerfile because the deployments this platform targets
> live on links that drop mid-download, but a link that cannot complete a
> 3 MB wheel at all will surface as `ResolutionImpossible` rather than as a
> network error — pip walks back through every candidate version before giving
> up. If you see that, the fix is a mirror or a pre-pulled base image, not a
> dependency pin. Build once somewhere with bandwidth and
> `docker save`/`docker load` the result onto the district machine.

- API: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>

The stack comes up with **no credentials at all**: every adapter falls back to
cached or synthetic data and `/health` and `/data/status` report that honestly.
Do not mistake a green dashboard for operational data — check
`meets_fusion_rule` first.

Populate a fresh volume:

```bash
docker compose run --rm seed
```

Bring up the database and cache too:

```bash
docker compose --profile full up
```

## Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard]"

python scripts/setup_database.py
python scripts/seed_historical_data.py

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_URL=http://localhost:8000 streamlit run dashboard/app.py &
python -m src.data_ingestion.scheduler &
```

### Install size and the `ml` extra

The **core install is deliberately light** — no XGBoost, LightGBM, SHAP or
scikit-learn. Those are all optional, because the platform ships working
fallbacks: a NumPy histogram gradient booster, exact TreeSHAP, and a
seasonal-naive stand-in for SARIMA.

```bash
pip install -e .                    # ~40 MB of wheels, works everywhere
pip install -e ".[ml]"              # + faster backends and reference SHAP
pip install -e ".[ml,dashboard]"    # + Streamlit and Plotly
```

`shap` alone pulls numba and llvmlite, roughly 60 MB of transitive
dependencies. A district node on an intermittent link should not have to fetch
that to produce a forecast, so it does not.

Training is slower without the extra; the outputs are the same shape, and the
platform picks up the faster backends automatically once they are installed
(`GET /health` reports which one resolved). CI runs **both** configurations on
every push precisely so the light path keeps working.

## Configuration

Everything is environment-driven — see `.env.example`. The settings that change
behaviour rather than credentials:

| Variable | Default | Effect |
|---|---|---|
| `DATABASE_URL` | empty → SQLite | central Postgres instance |
| `OFFLINE_MODE` | `false` | skip live fetches entirely; serve cache only |
| `API_KEY` | empty → **open** | require `X-API-Key` on every non-public route |
| `CORS_ORIGINS` | `*` | restrict browser origins |
| `RATE_LIMIT_PER_MINUTE` | `120` | per-client request cap |
| `DEFAULT_REGION` | `tanzania` | which `config/regions/*.yaml` to load |

## Before exposing this beyond a private network

The API defaults to **open**, deliberately: an agency that has not yet set up
secrets management must still be able to run it on an internal network. That
default is wrong on a public interface.

1. **Set `API_KEY`.** Without it, `/admin/retrain` and `/interventions/log` are
   unauthenticated.
2. **Terminate TLS at a reverse proxy** (nginx, Caddy, a cloud load balancer).
   The app speaks plain HTTP by design.
3. **Restrict `CORS_ORIGINS`** to the dashboard's real origin.
4. **Change `DB_PASSWORD`.** The compose default is a placeholder.
5. **Do not expose `db` or `redis` ports publicly.** Remove those `ports:`
   entries on a production host.
6. **Review the surveillance data you are ingesting** against your own
   information-governance rules before it leaves the ministry network.

## Scheduling

The scheduler polls each source on its own cadence — CHIRPS every 5 days, DHIS2
weekly, WorldPop annually — rather than hammering everything on one cron line.

```bash
python -m src.data_ingestion.scheduler            # daemon
python -m src.data_ingestion.scheduler --once     # run what is due, exit
python -m src.data_ingestion.scheduler --status   # freshness report
```

Retraining is separate and is triggered by cadence **or** detected drift:

```bash
curl -X POST localhost:8000/admin/retrain -H 'Content-Type: application/json' -d '{}'
```

A refit is promoted only if it beats the incumbent on a holdout window, so
running this repeatedly cannot degrade the deployed model.

Weekly forecasting, via cron:

```cron
0 6 * * 1  cd /opt/afya-predict && .venv/bin/python scripts/seed_historical_data.py --quiet
```

## Offline districts

An offline node is a normal install with `OFFLINE_MODE=true` and a populated
cache. Before it disconnects:

```bash
python scripts/seed_historical_data.py --forecast-weeks 6
curl localhost:8000/admin/offline/status
```

`ready: true` means at least two weeks of forecasts are cached and the dashboard
and API will keep serving through the outage. Alerts raised while offline are
queued to disk and delivered on reconnect:

```bash
curl -X POST "localhost:8000/admin/offline/sync?central_url=https://afya.moh.go.tz"
```

The SQLite file is a single portable artefact — it can be copied to a USB stick
and carried to a district office.

## Sizing

| Deployment | CPU | RAM | Disk |
|---|---|---|---|
| Single district (6 councils) | 2 cores | 2 GB | 5 GB |
| Regional (30 councils) | 4 cores | 4 GB | 20 GB |
| National (184 councils, 5 diseases) | 8 cores | 16 GB | 100 GB |

Training dominates CPU; serving is cache reads and is comfortably inside the
2-second budget on modest hardware. Reduce cost by lowering
`--forecast-weeks`, retraining monthly rather than weekly, and pruning the cache
(`POST /admin/cache/prune`).

## What the image actually contains

Verified on a fresh build of the default target:

```
$ docker build -t afya-predict .          # target: runtime (the light image)
$ docker run -d -p 8000:8000 afya-predict
$ curl -s localhost:8000/health
{"status":"degraded", "diseases":["cholera","hiv","malaria","respiratory","tuberculosis"],
 "districts":110, "backends":{"default_resolved":"numpy_gbm", "xgboost":false, ...},
 "cache":{"predictions":0, "offline_ready":false},
 "warnings":["fewer than 2 weeks of forecasts cached; this node is not yet ready to run offline",
             "no XGBoost/LightGBM/scikit-learn installed; using the bundled NumPy gradient booster ..."]}
```

Three things worth reading in that response:

- **All five diseases and 110 districts load from the packaged YAML**, so the
  configuration ships with the image rather than needing a mounted volume.
- **`default_resolved` is `numpy_gbm`.** The default image has no ML wheels, and
  the platform silently picked its bundled backend. Add `EXTRAS=ml` and the same
  endpoint will report `xgboost`.
- **`status` is `degraded`, not `ok`,** on a fresh volume — with warnings naming
  exactly what is missing. That is the intended behaviour: the service is up, the
  cache is empty, and it says so instead of implying it has forecasts to serve.
  Run `docker compose run --rm seed` and it becomes `ok`.

The container runs as a non-root user (`uid=10001 afya`) with `tini` as PID 1.

## Health and monitoring

```bash
curl localhost:8000/health           # capability + honest warnings
curl localhost:8000/data/status      # per-source freshness, fusion rule
curl localhost:8000/admin/drift/malaria
```

`/health` returns `degraded` with populated `warnings` rather than pretending to
be fine. Alert your monitoring on:

- `status != "ok"` for more than a few hours;
- `meets_fusion_rule == false` (fewer than three live sources);
- any source `stale == true`;
- `X-Response-Time-Ms` above 2000 on `/predictions`.

## Upgrading

```bash
git pull && docker compose build && docker compose up -d
docker compose run --rm backtest       # re-verify against the acceptance criteria
```

Model artefacts in `/app/artifacts` survive an upgrade. If the feature pipeline
changed, retrain — a stale model paired with new features will misalign columns
and produce nonsense.
