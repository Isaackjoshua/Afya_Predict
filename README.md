# AFYA-PREDICT

**AI-Powered Fusion Yield Analysis for Predictive Disease Intelligence in East Africa & Beyond**

*Afya* is Swahili for "health". AFYA-PREDICT is a free, open-source, explainable
outbreak-**forecasting** platform. It fuses satellite climate, mobility,
population, air-quality, WASH and routine health-surveillance streams onto a
common `district × epidemiological week` grid and predicts **where** and **when**
disease risk will rise — typically 4–12 weeks before cases reach facility reports.

Built for Tanzania first, designed to scale across Africa. MIT licensed.

---

## Why another system?

| Existing gap | AFYA-PREDICT |
|---|---|
| Reactive surveillance (DHIS2/eIDSR, HealthMap, ProMED) | Pre-outbreak forecasting from upstream climate and mobility drivers |
| Single-source fragility (Google Flu Trends) | ≥3 fused sources per disease; one feed failing degrades quality, not availability |
| Never retrained (GFT 2008–2015) | Page-Hinkley + ADWIN drift detection with automatic, holdout-gated refits |
| Black box (BlueDot) | SHAP attribution and a plain-language explanation on **every** prediction |
| Siloed human/animal/environment data | One Health schema feeding a single engine |
| Costly and proprietary | Free data sources, commodity hardware, no vendor lock-in |
| No spatial spread | Mobility-weighted diffusion naming the source districts, gravity-model fallback |
| Single disease | Plugin architecture — a new disease is a YAML file and a short class |
| Alerts without actions | Every alert carries costed, timeboxed, owned recommendations |
| Rarely externally validated | Walk-forward CV; three naive baselines a model must beat to deploy |
| Digital-divide bias | Primary signals are satellite + DHIS2, never search or social |
| No feedback loop | Interventions logged, impact triangulated, contaminated weeks down-weighted |

---

## Quick start

```bash
git clone https://github.com/Isaackjoshua/Afya_Predict.git
cd Afya_Predict
cp .env.example .env          # optional — it runs without any credentials
docker compose up --build
```

- API — <http://localhost:8000/docs>
- Dashboard — <http://localhost:8501>

Populate a fresh install with history, models and forecasts:

```bash
docker compose run --rm seed
```

Or without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dashboard]"   # drop "ml," for a light install
python scripts/setup_database.py
python scripts/seed_historical_data.py
uvicorn src.api.main:app --port 8000
```

**No credentials?** It still runs. Every adapter falls back to cached files and
then to a deterministic synthetic climatology, so you can exercise the full
pipeline offline on day one — and `/health` and `/data/status` tell you honestly
which sources are live and which are standing in.

---

## What a prediction looks like

```
Risk is high for cholera in Sengerema in 6 weeks (week 2026-W12): about 480
cases, 0.72 per 1,000 people (95% interval 380–610).

What is driving this: Rainfall 3 weeks ago (measured 118 mm) is pushing risk up,
accounting for 34% of the signal — heavy rain floods latrines and contaminates
drinking water sources. Safe water and sanitation coverage is pushing risk up,
accounting for 19% of the signal — low improved-water coverage sustains
faecal-oral transmission.

Roughly 42% of this risk is imported rather than local, arriving mainly from
Mwanza City, Magu.

If rainfall 3 weeks ago had been 30% lower (118 → 83 mm), predicted cases would
fall by 190 (40%), moving risk from HIGH to MEDIUM.

RECOMMENDED ACTIONS
 1. Deploy 5,000 ORS kits and 2 rehydration tents within 14 days
    within 7 days | owner: Regional Health Management Team
 2. Pre-position 2,900 ORS sachets in Sengerema ahead of week 2026-W12
    quantity: 2,900 (6.0 sachets per predicted case)
 3. Test and chlorinate approximately 1,300 community water points in Sengerema;
    52% of the population lacks improved water access
```

---

## Repository layout

```
config/          settings, disease YAMLs, region grids, alert rules
src/core/        domain types, config loading, epi-week and geo helpers
src/data_ingestion/    10 source adapters, normaliser, quality checks, scheduler
src/feature_engineering/  fitted lags, rolling stats, seasonality, spatial, mobility
src/models/      base module, 5 disease modules, ensemble, diffusion, drift, retrain
src/explainability/    SHAP, feature importance, natural language, counterfactuals
src/evaluation/  walk-forward CV, metrics, outbreak detection, baseline gate
src/alerting/    risk classification, alerts, recommendations, delivery
src/intervention_tracking/  response logging, impact estimation, feedback loops
src/api/         FastAPI service
dashboard/       Streamlit dashboard (6 pages)
offline/         SQLite cache, sync manager, distilled edge model
notebooks/       7 executable notebooks — see notebooks/README.md
scripts/         setup, seeding, backtesting, add_new_disease
docs/            architecture, data sources, adding diseases, deployment, API
```

---

## Adding a disease

```bash
python scripts/add_new_disease.py --name "Dengue Fever" --code DEN \
    --transmission vector_borne
python scripts/add_new_disease.py --validate dengue_fever
python scripts/run_backtest.py --disease dengue_fever
```

The core pipeline does not change. See [docs/adding_diseases.md](docs/adding_diseases.md).

---

## Validation

```bash
pytest                                     # 249 test functions
python scripts/check_acceptance.py --full  # all 15 acceptance criteria
python scripts/run_backtest.py --all       # walk-forward, against the baseline gate
```

`check_acceptance.py` evaluates every criterion programmatically and reports
`NOT VERIFIED` rather than assuming a pass for anything it could not measure.
Last run on 4 districts over 2019–2024, 3 walk-forward folds:

| # | Criterion | Result |
|---|---|---|
| 1 | ≥5 disease modules registered and functional | 5 registered, registry validates |
| 2 | ≥3 digital proxy sources per disease | 4–6 per disease |
| 3 | Predictions include 95% confidence intervals | coverage 0.945 measured out of sample |
| 4 | SHAP + natural-language explanation on every prediction | built in the same call as the forecast |
| 5 | Outbreak-detection AUC ≥ 0.75 | **0.933** |
| 6 | Beats all three naive baselines | narrowest margin **+67.5%** |
| 7 | District-level importation risk | produced, with source districts named |
| 8 | Drift detection triggers retraining | step change caught, no false alarm on a stable stream |
| 9 | Alerts carry actionable recommendations | 10 actions, all owned and timeboxed |
| 10 | API returns predictions in < 2 s | **19 ms** for 50 predictions |
| 11 | Dashboard heatmap, district detail, SHAP waterfall | 6 pages, 4 components |
| 12 | Offline mode caches ≥2 weeks | 4 weeks cached |
| 13 | `add_new_disease.py` scaffolds in < 5 min | config + module + registry + notebook |
| 14 | `docker compose` brings up the stack | all 5 services healthy; API, dashboard, scheduler, TimescaleDB (2 hypertables), Redis verified live |
| 15 | Test coverage ≥ 80% | **81%** |

Numbers 3, 5 and 6 come from synthetic data, so they demonstrate that the
machinery recovers a signal it was not given — not that the platform achieves
this accuracy on Tanzanian surveillance data. Re-run against live feeds before
quoting them.

Every model must beat three naive baselines — seasonal naive, 4-week rolling
mean, and smoothed persistence — before it is fit to deploy. `run_backtest.py`
exits non-zero when it does not, so it can gate a release.

CI runs the suite on Python 3.11 and 3.12, in **both** a full install and a
minimal one with no ML wheels at all, because that minimal configuration is what
a low-spec district deployment actually runs.

---

## Documentation

- [Architecture](docs/architecture.md) — how the layers fit together and what each guarantees
- [Data sources](docs/data_sources.md) — the ten adapters, credentials, quality rules
- [Adding diseases](docs/adding_diseases.md) — the extension path
- [Deployment](docs/deployment.md) — Docker, offline nodes, hardening, sizing
- [API reference](docs/api_reference.md) — endpoints and response shapes
- [Notebooks](notebooks/README.md) — executable documentation of every layer

---

## Honest limitations

- **The bundled data is synthetic** until you configure credentials. It encodes
  the causal lags the disease configs declare, which makes an end-to-end run a
  genuine test of signal recovery — but no figure produced from it is an
  epidemiological finding.
- **Impact estimates are observational.** Confidence is capped at *moderate*;
  the platform reports three counterfactuals and says `unresolved` when they
  disagree, rather than picking one.
- **The HIV module is a targeting tool, not a forecast.** It refuses
  individual-level inference and works only on district aggregates.
- **District coordinates and populations are approximate.** Point
  `boundaries.geojson_path` at an authoritative shapefile for operational use.
- **Alert thresholds ship as defaults.** They should be re-derived from local
  incidence distributions and response capacity.

---

## License

MIT — see [LICENSE](LICENSE). This platform must stay free for the health
agencies that need it most.
