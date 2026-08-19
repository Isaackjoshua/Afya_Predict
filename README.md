# AFYA-PREDICT

**AI-Powered Fusion Yield Analysis for Predictive Disease Intelligence in East Africa & Beyond**

*Afya* is Swahili for "health". AFYA-PREDICT is a free, open-source, explainable
outbreak-**forecasting** platform. It fuses satellite climate, mobility, population,
air-quality, WASH and routine health-surveillance streams onto a common
`district x epidemiological week` grid and predicts **where** and **when** disease
risk will rise — typically 4–12 weeks before cases appear in facility reports.

It is built for Tanzania first and designed to scale across Africa.

---

## Why another system?

| Existing gap | AFYA-PREDICT |
|---|---|
| Reactive surveillance (DHIS2/eIDSR, HealthMap, ProMED) | Pre-outbreak forecasting from upstream climate/mobility drivers |
| Single-source fragility (Google Flu Trends) | ≥3 fused sources per disease, degrades gracefully when one fails |
| Never retrained (GFT 2008–2015) | Drift detection (Page-Hinkley / ADWIN) + automatic rolling refit |
| Black box (BlueDot) | SHAP attribution + natural-language explanation on **every** prediction |
| Siloed human/animal/environment data | One Health schema feeding a single engine |
| Costly and proprietary | MIT-licensed, free data sources, commodity hardware |
| No spatial spread | Mobility-weighted diffusion, gravity-model fallback |
| Single disease | Plugin `DiseaseModule` architecture — a new disease is a YAML + a class |
| Alerts without actions | Every alert carries costed, timeboxed response recommendations |
| Rarely externally validated | Walk-forward CV, three naive baselines a model must beat to deploy |
| Digital-divide bias | Primary signals are satellite + DHIS2, never search/social |
| No feedback loop | Interventions are logged and their impact estimated |

---

## Quick start

```bash
git clone https://github.com/Isaackjoshua/Afya_Predict.git
cd Afya_Predict

python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

cp .env.example .env        # optional: fill in DHIS2 / CDS / telco credentials
```

No credentials? The platform still runs. Every adapter falls back to cached
files and then to a deterministic synthetic generator, so you can exercise the
full pipeline — ingestion, training, explanation, alerting, dashboard — offline
on day one.

---

## Repository layout

```
config/         settings, disease YAMLs, region grids, alert rules
src/core/       shared domain types, config loading, geo + epi-week helpers
src/data_ingestion/    source adapters, normaliser, quality checks, scheduler
src/feature_engineering/  lags, rolling stats, seasonality, spatial, mobility
src/models/     base model, disease modules, ensemble, diffusion, drift, retrain
src/explainability/   SHAP, feature importance, natural language, counterfactuals
src/evaluation/ walk-forward CV, metrics, outbreak detection, baselines
src/alerting/   risk classification, alert generation, recommendations, delivery
src/intervention_tracking/  response logging and impact estimation
src/api/        FastAPI service
dashboard/      Streamlit dashboard
offline/        SQLite cache, sync manager, lightweight edge model
scripts/        setup, seeding, backtesting, add_new_disease
docs/           architecture, data sources, deployment, API reference
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/data_sources.md`](docs/data_sources.md)
- [`docs/adding_diseases.md`](docs/adding_diseases.md)
- [`docs/deployment.md`](docs/deployment.md)
- [`docs/api_reference.md`](docs/api_reference.md)

## License

MIT — see [LICENSE](LICENSE). This platform must stay free for the health
agencies that need it most.
