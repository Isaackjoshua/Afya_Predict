# Adding a disease

Adding dengue, Rift Valley fever or Marburg should be a **configuration
exercise**, not an engineering project. The ingestion, feature, training,
explanation and alerting pipeline does not change.

## The five-minute path

```bash
python scripts/add_new_disease.py --name "Dengue Fever" --code DEN \
    --transmission vector_borne
```

That writes four things:

| File | What it is |
|---|---|
| `config/diseases/dengue_fever.yaml` | proxies, lags, mechanisms, thresholds, recommendations |
| `src/models/disease_modules/dengue_fever_module.py` | a subclass of `StandardDiseaseModule` |
| `src/models/registry.py` (edited) | registration, so it appears everywhere at once |
| `notebooks/explore_dengue_fever.ipynb` | a guided validation notebook |

`--transmission` selects a starting proxy set (`vector_borne`, `waterborne`,
`airborne`, `zoonotic`). Those presets are **starting points for expert review**,
not validated parameters.

## What you must fill in

### 1. Digital proxies — the core of the work

For each proxy, state *why* it relates to transmission. The mechanism is not
documentation; it is carried into the SHAP explanation an official reads.

```yaml
- name: "rainfall"
  source: "chirps"
  relationship: "positive_with_saturation"
  lag_weeks_range: [1, 12]      # the range the system SEARCHES
  optimal_lag_weeks: 4          # a prior, not an assumption
  mechanism: "rainfall fills the containers Aedes aegypti breeds in; above
              ~200 mm/month the containers overflow and larvae wash out"
  saturation_threshold_mm: 200
```

Rules the validator enforces:

- **at least three non-optional sources** (critical rule #1);
- **every proxy states a mechanism**;
- `optimal_lag_weeks` falls inside `lag_weeks_range`.

Make the range **wide enough to contain the truth**. The prior barely matters —
the system refits the lag per district — but a range that excludes the real lag
cannot be recovered from.

Available `relationship` values:

| Value | Shape | Example |
|---|---|---|
| `positive_linear` | monotone increasing | humidity → vector survival |
| `negative_linear` | monotone decreasing | WASH coverage → cholera |
| `bell_curve` | optimum in a range | temperature → parasite development |
| `positive_with_saturation` | rises then declines | rainfall → breeding sites |
| `threshold` | step above a value | heavy rain → flooding |

### 2. Alert thresholds

Cases per 1,000 population per week. Derive these from **your own** historical
incidence distribution, not from another country's. The generated notebook prints
the percentiles; a workable starting point is the 80th/95th/99th percentile for
medium/high/critical — then adjust against what your response capacity can
absorb. A threshold that fires weekly gets ignored.

The validator rejects all-equal or zero thresholds, because they would classify
every forecast into the same band.

### 3. Response recommendations

Text lives in YAML, not code (critical rule #9), so a health office can adapt it
to national IDSR guidance without a release. Higher levels **inherit** lower-level
actions, so a critical alert never silently drops a step the medium level
specified.

```yaml
recommendations:
  medium:
    - "Increase RDT stock at health facilities in {district}"
  high:
    - "Deploy vector-control teams to {district} within 7 days"
```

Placeholders `{district}`, `{region}`, `{week}` and `{cases}` are substituted at
alert time, and hard-coded quantities are rescaled to the district's population.

### 4. Wire up surveillance

Add the case variable so the target exists:

```python
# src/data_ingestion/adapters/dhis2_surveillance.py
CASE_VARIABLES = {
    ...,
    "dengue_fever": "cases_dengue_fever",
}
```

On a real DHIS2 instance, map it to the data element your ministry actually uses.

## Validate before you trust it

```bash
python scripts/add_new_disease.py --validate dengue_fever
```

Checks the config contract and the module interface, and lists unresolved TODOs.
Then run the exploratory notebook, then the deployment gate:

```bash
python scripts/run_backtest.py --disease dengue_fever
```

**The model must beat all three naive baselines** — seasonal naive, 4-week
rolling mean, smoothed persistence — before it is fit to deploy (critical rule
#10). `run_backtest.py` exits non-zero when it does not, so this can gate CI.

## When configuration is not enough

Override a method on the generated class. What the shipped modules do:

| Module | Override | Why |
|---|---|---|
| `MalariaModule` | `adjust_risk_level` | caps risk below an 18 °C transmission floor — the parasite's incubation exceeds the vector's lifespan |
| `CholeraModule` | `generate_recommendations` | adds a water-point chlorination action sized to the district's uncovered population |
| `TuberculosisModule` | `predict` | smooths over 4 weeks; TB notification noise is case-finding effort, not transmission |
| `RespiratoryModule` | `generate_recommendations` | adds an air-quality advisory during pollution episodes |
| `HIVModule` | `predict`, `detect_outbreak` | enforces aggregate-only framing and appends the targeting-not-forecasting disclaimer |

Keep these narrow. Anything that would help several diseases belongs in
`StandardDiseaseModule` or in configuration.

## A note on responsibility

A disease module is a claim that these proxies, at these lags, predict this
disease in this ecology. Where that claim is weak, say so in the module —
`HIVModule` is the worked example: it refuses individual-level inference,
constrains itself to district aggregates, and states its own limits in every
alert it produces. Copy that pattern when the epidemiology does not support a
confident forecast.
