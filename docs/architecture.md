# Architecture

## The shape of the problem

Most disease-surveillance systems answer *"has an outbreak started?"*. AFYA-PREDICT
answers *"where and when will risk rise?"* — which is a different computation with
a different failure mode.

The forecast is possible because transmission has **upstream drivers with known
lags**. Rainfall creates mosquito breeding sites weeks before malaria cases reach
a clinic. Reservoir warming precedes cholera by roughly four months. Those lags
are the lead time; everything in this system exists to extract them, quantify the
uncertainty around them, and turn the result into something a district health
team can act on.

```
   satellite · mobility · surveillance · census · WASH · animal health
                              │
                        ┌─────▼─────┐
                        │  adapters │  live → cache → synthetic, never fail
                        └─────┬─────┘
                              │  tidy: district, week, variable, value, quality
                        ┌─────▼─────┐
                        │ normaliser│  one district × epi-week panel
                        └─────┬─────┘
                              │  + quality and imputation flags
                     ┌────────▼────────┐
                     │ feature builder │  lags FITTED per district
                     └────────┬────────┘
                              │  + provenance and mechanism per column
        ┌─────────────────────▼─────────────────────┐
        │            disease module                 │
        │  pooled model + per-district models       │
        │  transfer learning for sparse districts   │
        └──────┬──────────────────────┬─────────────┘
               │                      │
        ┌──────▼──────┐        ┌──────▼───────┐
        │ explainer   │        │  diffusion   │  where it spreads next
        │ SHAP + text │        └──────┬───────┘
        └──────┬──────┘               │
               └──────────┬───────────┘
                     ┌────▼─────┐
                     │ alerting │  classify → recommend → deliver
                     └────┬─────┘
                          │
                ┌─────────▼──────────┐
                │ intervention loop  │  log → estimate → reweight
                └────────────────────┘
```

## Layer by layer

### 1. Ingestion — `src/data_ingestion/`

Every source implements one contract: `fetch() → validate() → normalize()`.
Whatever the wire format — GeoTIFF, GRIB, DHIS2 JSON, telco Parquet — the output
is the same five columns: `district, week, variable, value, quality, source`.

Two behaviours matter more than the adapter list:

**Degradation, not failure.** A fetch tries live, then the on-disk cache, then a
deterministic synthetic climatology. A missing credential produces a flagged,
lower-quality row — never an exception that stops the run. This is why the
platform is usable on day one and why one broken feed cannot take down a
national forecast.

**Quality propagates.** Every cell carries a 0–1 quality score. A missing DHIS2
week is `NaN` with `quality = 0`, *not* zero cases: roughly 70% of Tanzanian
deaths occur outside facilities, so absence of a report is not absence of
disease. Those scores flow into sample weights during training and into interval
width at prediction time.

### 2. Normalisation — `src/data_ingestion/normalizer.py`

Fuses everything onto `district × ISO epidemiological week`. Gaps in *driver*
series are filled in a recorded order — short interpolation, then
neighbour-district borrowing, then the district's own seasonal climatology, then
the national median — and every filled cell is marked in a `__imputed` column.

Case counts are never imputed. Fabricating a target poisons the model.

### 3. Feature engineering — `src/feature_engineering/`

The defining choice: **lags are fitted, not assumed.** The disease YAML supplies
a search *range* and a prior; `fit_optimal_lags` scans the range against each
district's own history and keeps what the data supports. Districts with too
little history inherit the pooled national fit, recorded as `pooled` rather than
passed off as a local result.

Every generated column carries provenance — which proxy, which lag, which source,
and the causal mechanism. That is what lets the explainer say "rainfall six weeks
ago, which floods breeding sites" instead of "feature 37".

### 4. Models — `src/models/`

`BaseDiseaseModule` is the contract; `StandardDiseaseModule` implements all of it
from configuration. A concrete disease overrides only what configuration cannot
express — malaria's 18 °C transmission floor, cholera's WASH gate, TB's smoothing.

A **pooled** model is always fitted; districts with enough history also get their
own. Backends resolve XGBoost → LightGBM → scikit-learn → a bundled NumPy
histogram GBM, so training works on a machine that cannot install a 57 MB wheel.

Prediction intervals come from residuals measured on a chronological internal
holdout, split by week. An earlier version used in-sample residuals and covered
37% of outcomes at a nominal 95%; walk-forward validation caught it.

### 5. Explainability — `src/explainability/`

Every prediction is constructed *together with* its explanation, so a
`PredictionResult` without SHAP drivers cannot come out of the normal path.
SHAP resolves to the `shap` package, exact TreeSHAP for the bundled ensemble, or
permutation Shapley values — all three satisfy local accuracy, which is what
makes "34% of this elevation" a real share rather than a decoration.

### 6. Evaluation — `src/evaluation/`

Walk-forward validation with a purge gap equal to the forecast horizon. Three
naive baselines a model must beat before deployment. Outbreak detection scored
at the operational threshold, with timeliness — the actual lead time delivered —
reported alongside it.

### 7. Alerting — `src/alerting/`

Threshold classification, adjusted for trend, importation pressure and input
uncertainty. Recommendations sized to district population and forecast burden,
each with an owner and a deadline. Delivery across log/email/SMS/DHIS2, queued to
disk when a channel is unreachable.

### 8. Intervention feedback — `src/intervention_tracking/`

Logs what was done, estimates what it changed against three counterfactuals, and
feeds the result back: response latency is audited against forecast lead time,
and intervention-affected weeks are down-weighted in retraining so the model is
not penalised for outbreaks it helped avert.

### 9. Offline — `offline/`

SQLite store, sync manager, and a ridge-distilled edge model. The districts with
the worst connectivity carry the highest burden, so this is a first-class path,
not a degraded one.

## Cross-cutting invariants

| Invariant | Enforced in |
|---|---|
| ≥3 fused sources per disease | `validate_disease_config`, `Normalizer._fusion_flags` |
| No prediction without an explanation | `StandardDiseaseModule.predict` |
| Lags fitted, never transplanted | `fit_optimal_lags` |
| Retraining automatic, gated on holdout | `AutoRetrainer.retrain` |
| Quality widens intervals rather than hiding | `_predict_rows`, `RiskClassifier` |
| Search/social never primary | `SearchTrendsAdapter.optional = True` |
| Alerts carry costed actions | `RecommendationEngine` |
| Models beat three naive baselines | `benchmark_against_naive` |

## Extension points

- **New data source** — subclass `BaseAdapter`, `register_adapter`.
- **New disease** — `scripts/add_new_disease.py`; see [adding_diseases.md](adding_diseases.md).
- **New region** — copy `config/regions/_template.yaml`.
- **New backend** — add to `_BACKENDS` in `src/models/backends.py`.
