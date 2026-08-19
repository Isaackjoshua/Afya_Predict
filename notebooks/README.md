# Notebooks

Executable documentation for AFYA-PREDICT. Each notebook runs against the real
codebase — no notebook re-implements platform logic, they import it — so if a
notebook stops running, something in `src/` has changed behaviour.

Run them in order the first time; after that they are independent.

| # | Notebook | What it answers | Runtime |
|---|---|---|---|
| 01 | [`01_data_exploration.ipynb`](01_data_exploration.ipynb) | What does each of the 10 adapters actually produce, how complete is it, and do the fused series behave epidemiologically? | ~1 min |
| 02 | [`02_lag_analysis.ipynb`](02_lag_analysis.ipynb) | How many weeks after a rainfall anomaly do cases rise — and do districts agree? (They do not, which is why lags are fitted per district.) | ~1 min |
| 03 | [`03_model_training.ipynb`](03_model_training.ipynb) | End-to-end training: features → model → explained, actionable forecast. Includes leakage and interval-coverage checks. | ~3 min |
| 04 | [`04_model_comparison.ipynb`](04_model_comparison.ipynb) | Backend and ensemble comparison, walk-forward validation, and the three naive baselines a model must beat before deployment. | ~6 min |
| 05 | [`05_spatial_validation.ipynb`](05_spatial_validation.ipynb) | Did the disease spread *where* we said it would? Travel matrices, diffusion, hotspot hit rate and displacement error in km. | ~5 min |
| 06 | [`06_drift_and_retraining.ipynb`](06_drift_and_retraining.ipynb) | The Google Flu Trends failure mode, caught: drift detection, the retraining decision, and the promotion gate. | ~4 min |
| 07 | [`07_alerting_and_interventions.ipynb`](07_alerting_and_interventions.ipynb) | Forecast → alert → recommendation → delivery → logged response → impact estimate → back into the model. | ~3 min |

## Running them

```bash
pip install -e ".[dev]"
pip install jupyter matplotlib      # neither is a runtime dependency
jupyter lab notebooks/
```

They also execute unattended, which is how they are checked:

```bash
jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb \
    --output /tmp/out.ipynb
```

**Two things make that work and are worth preserving when editing:**

- The bootstrap cell locates the repo root by walking up from the working
  directory, so notebooks run from any cwd.
- The plotting cell falls back to `Agg` when no display is attached, and to
  printed tables when matplotlib is absent entirely. Without that, an unattended
  run blocks forever on a GUI window that never opens.

## About the numbers in these notebooks

With no credentials in `.env`, every adapter falls back to a deterministic
synthetic climatology (see `src/data_ingestion/base_adapter.py`). That is
deliberate — the whole pipeline is runnable on day one — but it means **the
figures produced here are demonstrative, not epidemiological findings**.

Each notebook prints the fetch `mode` per source. On a real deployment, check
that it says `live` before quoting any number from these pages.

The synthetic generator is not noise: it encodes the same causal lags the disease
configs declare (rainfall → malaria at ~6 weeks, heavy rain → cholera at ~3
weeks, reservoir warming → cholera at ~16 weeks), plus the reporting pathologies
the real system has to survive — missing weeks, sub-100% completeness, and
occasional reporting spikes. That is what makes an end-to-end run a genuine test
of signal recovery rather than an exercise in fitting noise.
