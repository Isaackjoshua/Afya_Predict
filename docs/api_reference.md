# API reference

Interactive docs are served at `/docs` (Swagger) and `/redoc`. The OpenAPI
schema is at `/openapi.json`.

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Conventions

- **Reads are cache-backed.** `GET /predictions/...` reads precomputed forecasts
  from the local store, which is what keeps it inside the 2-second budget. Any
  endpoint that trains or ingests is a `POST` and says so.
- **Explanations are never stripped.** Every prediction returned carries its SHAP
  drivers, natural-language summary, counterfactual and recommendations. There is
  deliberately no "summary" endpoint that removes the reasoning.
- **Auth:** open by default. When `API_KEY` is set, send `X-API-Key` on every
  route except `/`, `/health`, `/docs`, `/redoc`, `/openapi.json`.
- **Rate limit:** `RATE_LIMIT_PER_MINUTE` per client, `429` with `Retry-After`.
- **Timing:** every response carries `X-Response-Time-Ms`.

## Meta

### `GET /health`

Liveness plus an honest account of what this instance can currently do.

```json
{
  "status": "degraded",
  "version": "0.1.0",
  "diseases": ["cholera", "hiv", "malaria", "respiratory", "tuberculosis"],
  "districts": 110,
  "backends": {"default_resolved": "numpy_gbm", "xgboost": false},
  "cache": {"predictions": 12, "weeks_cached": 4, "offline_ready": true},
  "warnings": ["no XGBoost/LightGBM/scikit-learn installed; using the bundled ..."]
}
```

`degraded` means the service is up but something is missing, and `warnings` says
what. Monitor on `status` and on the specific warnings, not on HTTP status alone.

## Predictions

### `GET /predictions`

Query: `disease`, `district`, `target_week`, `risk_level`, `limit` (≤1000).

Returns `{count, generated_at, predictions[], warnings[]}`. An empty result
includes a warning explaining how to populate the cache rather than returning a
bare empty list.

Each prediction contains:

| Field | Meaning |
|---|---|
| `predicted_cases` | point forecast for `target_week` |
| `confidence_interval_lower/upper` | 95% interval from out-of-sample residuals |
| `risk_level`, `risk_score` | threshold band and continuous 0–1 severity |
| `shap_values` | per-feature contribution (sums to prediction − base) |
| `top_drivers[]` | ranked drivers with `proxy`, `lag_weeks`, `mechanism`, `contribution_share` |
| `natural_language_explanation` | the paragraph a health officer reads |
| `counterfactual` | "if rainfall had been 20% lower, …" |
| `importation_risk`, `source_districts[]` | where the risk is arriving from |
| `recommendations[]` | action, owner, deadline, quantity |
| `data_quality_flags[]` | why confidence may be reduced |

### `GET /predictions/{disease}/{district}`

Forecast history for one district, newest first. `404` if nothing is cached.

### `GET /predictions/id/{prediction_id}`

One prediction by id.

### `GET /predictions/{disease}/map/national`

District rows with `lat`/`lon` and risk, for the heatmap.

### `POST /predictions/run`

The slow path — ingests, rebuilds features, predicts. Trains first if no model
is persisted.

```json
{"disease": "malaria", "districts": ["Kinondoni"], "horizon_weeks": 8, "persist": true}
```

## Alerts

| Route | Purpose |
|---|---|
| `GET /alerts` | recent alerts; filter by disease, district, level, acknowledged, days |
| `GET /alerts/active` | unacknowledged, medium and above, severity-ordered — the working queue |
| `GET /alerts/{alert_id}` | one alert |
| `POST /alerts/{alert_id}/acknowledge` | `{"acknowledged_by": "dho@..."}` |
| `POST /alerts/bulk` | receive alerts pushed by an offline node |
| `GET /alerts/summary/by-district` | counts per district and level |

Acknowledgement is part of the feedback loop, not bookkeeping: an alert never
acknowledged is evidence the warning did not reach a decision-maker, and
`/interventions/audit/responses` reports exactly that.

## Explainability

### `GET /explain/{prediction_id}`

Full explanation plus `source_contributions[]` — SHAP rolled up to the data
source that produced each feature. If one source dominates, the platform has
recreated the single-source dependence that broke Google Flu Trends, and this is
where you see it.

### `GET /explain/{prediction_id}/waterfall`

Ordered contributions for the waterfall chart, labelled by proxy and fitted lag
rather than by engineered column name.

## Data status

| Route | Purpose |
|---|---|
| `GET /data/status` | per-source freshness, mode (`live`/`cache`/`synthetic`), `meets_fusion_rule` |
| `GET /data/sources` | adapter capability report |
| `GET /data/quality` | sampled completeness and quality flags |

Check `meets_fusion_rule` before quoting any forecast. `false` means fewer than
three sources returned live data; predictions are still produced, with widened
intervals, but they are not operational.

## Diseases

| Route | Purpose |
|---|---|
| `GET /diseases` | every registered module and its state |
| `GET /diseases/{slug}` | one disease |
| `GET /diseases/{slug}/config` | full config **including every proxy's mechanism** |
| `GET /diseases/{slug}/lags` | the lag actually fitted per district |

`/config` and `/lags` are published deliberately: an agency should be able to see
which proxies drive a disease, at what lags, and why, without reading the code.
`/lags` is the evidence that coefficients were learned locally rather than
transplanted — a wide spread across districts is the expected result.

## Interventions

| Route | Purpose |
|---|---|
| `POST /interventions/log` | record a response (201) |
| `GET /interventions` | list |
| `GET /interventions/types` | known types and their effect lags |
| `GET /interventions/{id}/impact` | triangulated effect estimate |
| `GET /interventions/audit/responses` | did alerts produce responses, fast enough? |
| `POST /interventions/bulk` | receive from an offline node |

```bash
curl -X POST localhost:8000/interventions/log -H 'Content-Type: application/json' -d '{
  "disease": "cholera", "district": "Sengerema",
  "intervention_type": "water_chlorination",
  "coverage": 0.62, "quantity": 48000, "unit": "households",
  "alert_id": "…", "logged_by": "dhmt.mwanza"
}'
```

`coverage` is a 0–1 share and is validated (`422` outside that range). It matters:
5,000 nets reaching 12% of a district three weeks late is a different exposure
from 50,000 reaching 80% on time.

`/impact` returns three estimates — forecast counterfactual,
difference-in-differences, pre/post — with a `confidence` label capped at
`moderate`, because this is observational data, not a trial. When the estimates
disagree in sign it returns `unresolved` rather than picking one.

## Admin

Protect these with `API_KEY` before exposing the service.

| Route | Purpose |
|---|---|
| `POST /admin/retrain` | drift check and gated refit |
| `GET /admin/drift/{disease}` | current drift verdict |
| `POST /admin/ingest` | background data refresh |
| `POST /admin/backtest/{disease}` | walk-forward validation (slow) |
| `GET /admin/offline/status` | offline readiness |
| `POST /admin/offline/sync` | push local records, pull forecasts |
| `POST /admin/cache/prune` | drop stale rows |
| `GET /admin/registry/validate` | config + interface check for every disease |

## Errors

| Status | Meaning |
|---|---|
| `401` | missing or invalid `X-API-Key` |
| `404` | unknown disease, district, prediction or alert |
| `409` | the request is valid but the data is not there yet (e.g. no fitted lags in memory) |
| `422` | request-body validation failed |
| `429` | rate limited; see `Retry-After` |
| `500` | unhandled error — logged server-side, never leaked to the client |

## Python client

No SDK is needed; the API is plain JSON.

```python
import requests

base = "http://localhost:8000"
alerts = requests.get(f"{base}/alerts/active").json()["alerts"]

for alert in alerts:
    print(f"[{alert['risk_level'].upper()}] {alert['disease']} — {alert['district']}")
    print(alert["explanation"])
    for rec in alert["recommendations"][:3]:
        print(f"  → {rec['action']} ({rec['responsible']}, {rec['timeframe_days']}d)")
```
