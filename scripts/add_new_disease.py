#!/usr/bin/env python3
"""Scaffold a new disease module (shortcoming #11, acceptance criterion #13).

Adding a disease must be a config exercise, not an engineering project:

    python scripts/add_new_disease.py --name "Dengue Fever" --code DEN \
        --transmission vector_borne

This writes the YAML config from the template, writes a module class from the
module template, registers it, and then *validates* the result — so a half-wired
disease fails loudly here rather than silently producing nothing downstream.

It deliberately does not invent epidemiology. The generated config is a skeleton
with `TODO` markers; a domain expert fills in the proxies, lags and mechanisms.
For a few well-documented diseases a starting proxy set is offered via
`--preset`, clearly marked as a starting point to be reviewed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.settings import DISEASE_CONFIG_DIR  # noqa: E402
from src.core.logging import get_logger  # noqa: E402

log = get_logger("scripts.add_disease")

MODULE_DIR = REPO_ROOT / "src" / "models" / "disease_modules"
REGISTRY_PATH = REPO_ROOT / "src" / "models" / "registry.py"
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

TRANSMISSION_MODES = ["vector_borne", "waterborne", "airborne", "sexual", "zoonotic", "other"]

#: Starting proxy sets for diseases with a well-documented transmission chain.
#: These are *starting points for expert review*, not validated parameters.
PRESETS: Dict[str, dict] = {
    "vector_borne": {
        "note": "Aedes/Anopheles-style vector chain: breeding sites, development rate, survival.",
        "proxies": [
            ("rainfall", "chirps", "positive_with_saturation", [1, 12], 4,
             "rainfall creates and then washes out vector breeding sites",
             {"saturation_threshold_mm": 200}),
            ("temperature", "era5", "bell_curve", [2, 16], 6,
             "temperature controls vector development rate and pathogen incubation",
             {"optimal_range_celsius": [26, 31]}),
            ("humidity", "era5", "positive_linear", [1, 8], 3,
             "humidity governs adult vector survival and biting activity", {}),
            ("population_density", "population_density", "positive_linear", [0, 0], 0,
             "urban container-breeding vectors track human density closely", {}),
            ("mobility_inbound", "cdr_mobility", "positive_linear", [1, 4], 2,
             "viraemic travellers seed new districts", {"spatial": True}),
        ],
        "spatial": {"enabled": True, "importation_weight": 0.35},
        "horizon": 6,
    },
    "waterborne": {
        "note": "Faecal-oral chain: contamination pressure gated by sanitation coverage.",
        "proxies": [
            ("rainfall", "chirps", "threshold", [1, 8], 2,
             "heavy rain floods sanitation and contaminates drinking water",
             {"threshold_mm": 80}),
            ("temperature", "era5", "positive_linear", [4, 20], 12,
             "warm water accelerates pathogen survival in aquatic reservoirs", {}),
            ("wash_access", "wash_indicators", "negative_linear", [0, 0], 0,
             "low improved-water and sanitation coverage sustains transmission", {}),
            ("population_density", "population_density", "positive_linear", [0, 0], 0,
             "crowding raises exposure to shared contaminated water points", {}),
            ("mobility_inbound", "cdr_mobility", "positive_linear", [1, 3], 1,
             "infected travellers seed districts along transport corridors",
             {"spatial": True}),
        ],
        "spatial": {"enabled": True, "importation_weight": 0.45},
        "horizon": 6,
    },
    "airborne": {
        "note": "Respiratory chain: contact rate, indoor crowding, airway irritation.",
        "proxies": [
            ("air_quality_pm25", "sentinel5p", "positive_linear", [0, 6], 1,
             "particulate exposure inflames airways and precipitates infection", {}),
            ("temperature", "era5", "negative_linear", [1, 6], 2,
             "cooler spells increase indoor crowding and pathogen stability", {}),
            ("humidity", "era5", "bell_curve", [1, 6], 2,
             "very low and very high humidity both favour airborne transmission",
             {"optimal_range_percent": [40, 60]}),
            ("population_density", "population_density", "positive_linear", [0, 0], 0,
             "contact rate scales with density", {}),
            ("mobility_inbound", "cdr_mobility", "positive_linear", [1, 3], 1,
             "travel imports circulating strains between districts", {"spatial": True}),
        ],
        "spatial": {"enabled": True, "importation_weight": 0.4},
        "horizon": 4,
    },
    "zoonotic": {
        "note": "Spillover chain: animal reservoir activity plus environmental trigger.",
        "proxies": [
            ("livestock_events", "livestock_disease", "positive_linear", [1, 8], 2,
             "animal outbreaks precede human spillover (One Health signal)", {}),
            ("rainfall", "chirps", "positive_with_saturation", [2, 12], 6,
             "flooding concentrates animal and human contact around water",
             {"saturation_threshold_mm": 180}),
            ("temperature", "era5", "positive_linear", [2, 12], 6,
             "temperature governs reservoir and vector activity", {}),
            ("ndvi", "modis", "positive_linear", [2, 12], 6,
             "vegetation flush drives herd movement and vector habitat", {}),
            ("population_density", "population_density", "positive_linear", [0, 0], 0,
             "human-animal interface density governs spillover opportunity", {}),
        ],
        "spatial": {"enabled": True, "importation_weight": 0.25},
        "horizon": 6,
    },
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def class_name(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", name) if part) + "Module"


def render_config(name: str, code: str, transmission: str, preset: Optional[str],
                  vector: str, pathogen: str) -> str:
    chosen = PRESETS.get(preset or transmission)
    lines = [
        f"# {name} ({code}) - generated by scripts/add_new_disease.py",
        "#",
        "# REVIEW EVERY VALUE BELOW BEFORE USE. Lag ranges and thresholds are starting",
        "# points for a domain expert, not validated parameters. The system fits the",
        "# lag within the range you give it (critical rule #3), so the range matters",
        "# more than the prior - make it wide enough to contain the truth.",
    ]
    if chosen:
        lines.append(f"# Preset applied: {chosen['note']}")
    lines += [
        "disease:",
        f'  name: "{name}"',
        f'  code: "{code}"',
        f'  transmission_mode: "{transmission}"',
        f'  vector: "{vector}"',
        f'  pathogen: "{pathogen}"',
        "",
        "  # At least three non-optional sources are required (critical rule #1).",
        "  digital_proxies:",
    ]

    proxies = chosen["proxies"] if chosen else [
        ("", "", "positive_linear", [0, 0], 0, "TODO: why does this proxy relate to transmission?", {}),
    ]
    for proxy, source, relationship, lag_range, prior, mechanism, extra in proxies:
        lines += [
            f'    - name: "{proxy}"',
            f'      source: "{source}"',
            f'      relationship: "{relationship}"',
            f"      lag_weeks_range: [{lag_range[0]}, {lag_range[1]}]",
            f"      optimal_lag_weeks: {prior}",
            f'      mechanism: "{mechanism}"',
        ]
        for key, value in extra.items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            lines.append(f"      {key}: {rendered}")
        lines.append("")

    spatial = chosen["spatial"] if chosen else {"enabled": False, "importation_weight": 0.0}
    horizon = chosen["horizon"] if chosen else 4
    lines += [
        "  spatial:",
        f"    enabled: {str(spatial['enabled']).lower()}",
        '    diffusion_model: "gravity"',
        f"    importation_weight: {spatial['importation_weight']}",
        "",
        "  model:",
        '    primary: "xgboost"',
        '    ensemble_members: ["xgboost", "lightgbm", "sarima"]',
        f"    forecast_horizon_weeks: {horizon}",
        '    retrain_frequency: "monthly"',
        "    min_training_months: 24",
        "",
        "  # TODO: cases per 1,000 population per week at which each level triggers.",
        "  # Derive these from your own historical distribution, not from another",
        "  # country's - see notebooks/04_model_comparison.ipynb section 6.",
        "  alerts:",
        "    low: 0.0",
        "    medium: 0.0",
        "    high: 0.0",
        "    critical: 0.0",
        "",
        "  # TODO: response actions per level, from your national IDSR guidance.",
        "  # These are configuration, not code (critical rule #9) - edit freely.",
        "  recommendations:",
        "    medium:",
        f'      - "TODO: first {name.lower()} response step at MEDIUM risk"',
        "    high:",
        f'      - "TODO: first {name.lower()} response step at HIGH risk"',
        "    critical:",
        f'      - "TODO: first {name.lower()} response step at CRITICAL risk"',
        "",
    ]
    return "\n".join(lines)


def render_module(name: str, slug: str, cls: str) -> str:
    return f'''"""{name} disease module.

Generated by `scripts/add_new_disease.py`. `StandardDiseaseModule` already
implements the whole `BaseDiseaseModule` contract from the YAML config, so this
subclass only needs to exist. Override a method below when {name.lower()} needs
behaviour the shared engine cannot express through configuration.
"""

from __future__ import annotations

from typing import List

from src.core.types import Alert, Recommendation, RiskLevel
from src.models.standard_module import StandardDiseaseModule


class {cls}(StandardDiseaseModule):
    """Config-driven {name.lower()} module."""

    slug = "{slug}"

    # --- Optional overrides -------------------------------------------------
    #
    # def adjust_risk_level(self, level, incidence, importation_risk, low_confidence) -> RiskLevel:
    #     """Apply a biological or programmatic constraint the thresholds miss.
    #
    #     Examples from the shipped modules: malaria caps risk below an 18 C
    #     transmission floor; cholera escalates in districts with weak WASH
    #     coverage.
    #     """
    #     return super().adjust_risk_level(level, incidence, importation_risk, low_confidence)
    #
    # def generate_recommendations(self, alert: Alert) -> List[Recommendation]:
    #     """Add actions the YAML templates cannot express."""
    #     recommendations = super().generate_recommendations(alert)
    #     return recommendations
    #
    # def predict(self, matrix, district: str, **kwargs):
    #     """Post-process the forecast - smoothing, caps, extra disclaimers."""
    #     return super().predict(matrix, district, **kwargs)
'''


def render_notebook(name: str, slug: str) -> str:
    import json

    def code(src):
        lines = src.split("\n")
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}

    def md(src):
        lines = src.split("\n")
        return {"cell_type": "markdown", "metadata": {},
                "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}

    cells = [
        md(f"""# Exploratory analysis: {name}

Generated by `scripts/add_new_disease.py`.

Before this module is deployed it must clear three gates:

1. the config validates (at least three fused sources, mechanisms stated,
   monotonic thresholds, recommendations present);
2. the fitted lags are epidemiologically plausible for {name.lower()};
3. the model beats all three naive baselines (critical rule #10).

Work through the sections below, then run
`python scripts/run_backtest.py --disease {slug}`."""),
        code("""import sys, pathlib, warnings
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
            if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import logging; logging.getLogger("afya").setLevel(logging.WARNING)
import numpy as np, pandas as pd"""),
        md("## 1. Does the config validate?"),
        code(f"""from src.core.config_loader import load_disease_config, validate_disease_config

config = load_disease_config("{slug}")
problems = validate_disease_config(config)
print(f"{{config.name}} ({{config.code}}) - {{len(config.digital_proxies)}} proxies, "
      f"{{len(config.required_sources)}} non-optional sources")
if problems:
    print("\\nPROBLEMS TO FIX:")
    for problem in problems:
        print(f"  - {{problem}}")
else:
    print("\\nconfig is valid")"""),
        md("## 2. Are the declared mechanisms defensible?\n\nEvery proxy needs a causal story, not just a correlation."),
        code("""pd.DataFrame([{
    "proxy": p.name, "source": p.source, "relationship": p.relationship.value,
    "lag_range": f"{p.lag_weeks_range[0]}-{p.lag_weeks_range[1]}w",
    "prior": p.optimal_lag_weeks, "mechanism": p.mechanism,
} for p in config.digital_proxies])"""),
        md("## 3. Ingest and inspect the drivers"),
        code(f"""from src.core.config_loader import load_region_config
from src.core.geo import subset_region
from src.data_ingestion.normalizer import ingest

REGION = subset_region(load_region_config(),
                       ["Kinondoni", "Mwanza City", "Sengerema", "Dodoma City"])
SOURCES = sorted(set(config.required_sources) | {{"dhis2"}})
panel = ingest(SOURCES, "2020-W01", "2024-W52", region=REGION)
print(panel.summary())"""),
        md("## 4. Fit the lags\n\nDo they land inside the ranges you declared, and are they biologically plausible?"),
        code(f"""from src.feature_engineering.builder import PROXY_TO_VARIABLE
from src.feature_engineering.lag_features import fit_optimal_lags, lag_dispersion

specs = [p for p in config.digital_proxies if not p.optional]
variable_for_proxy = {{p.name: PROXY_TO_VARIABLE.get(p.name, p.name) for p in specs}}
target = f"cases_{slug}"

if target in panel.value_columns:
    fits = fit_optimal_lags(panel.values(), target, specs, variable_for_proxy)
    display(lag_dispersion(fits))
else:
    print(f"No surveillance column {{target!r}} yet.")
    print("Add this disease to CASE_VARIABLES in")
    print("src/data_ingestion/adapters/dhis2_surveillance.py, and map it to the")
    print("DHIS2 data element your instance uses.")"""),
        md("## 5. Train and check against the baselines\n\nThis is the gate that decides deployment."),
        code(f"""from src.models.registry import build_module
from src.evaluation.walk_forward_cv import WalkForwardCV

module = build_module("{slug}", region=REGION)
matrix = module.build_feature_matrix(panel)
cv = WalkForwardCV(module, initial_train_weeks=120, step_weeks=26, test_weeks=26, max_folds=3)
result = cv.run(matrix)

print(result.summary_line())
for note in result.acceptance_notes:
    print(f"  - {{note}}")"""),
        md(f"""## 6. Set the alert thresholds

Thresholds belong in `config/diseases/{slug}.yaml` and should come from *your*
historical incidence distribution. A useful starting point: set `medium` near the
80th percentile of observed weekly incidence, `high` near the 95th, and `critical`
near the 99th — then review against what your response capacity can actually
absorb."""),
        code(f"""populations = pd.Series({{d.name: float(d.population) for d in REGION.districts}})
if target in panel.value_columns:
    cases = panel.values()[target]
    incidence = cases / cases.index.get_level_values("district").map(populations) * 1000
    display(incidence.describe(percentiles=[0.5, 0.8, 0.95, 0.99]).round(4))
    print("\\nSuggested starting thresholds (review against response capacity):")
    for level, q in [("low", 0.5), ("medium", 0.8), ("high", 0.95), ("critical", 0.99)]:
        print(f"  {{level:9}}: {{incidence.quantile(q):.4f}}")"""),
    ]
    return json.dumps({
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }, indent=1) + "\n"


def register(slug: str, cls: str) -> bool:
    """Add the new module to the registry, idempotently."""
    text = REGISTRY_PATH.read_text()
    if cls in text:
        log.info("%s is already registered", cls)
        return False

    import_anchor = "from src.models.disease_modules import ("
    start = text.index(import_anchor) + len(import_anchor)
    end = text.index(")", start)
    imports = [n.strip().rstrip(",") for n in text[start:end].split("\n") if n.strip()]
    imports.append(cls)
    text = text[:start] + "\n    " + ",\n    ".join(sorted(imports)) + ",\n" + text[end:]

    entry_anchor = "DISEASE_MODULE_REGISTRY: Dict[str, Type[BaseDiseaseModule]] = {"
    position = text.index(entry_anchor) + len(entry_anchor)
    text = text[:position] + f"\n    {cls}.slug: {cls}," + text[position:]

    REGISTRY_PATH.write_text(text)

    init_path = MODULE_DIR / "__init__.py"
    init_text = init_path.read_text()
    module_name = f"src.models.disease_modules.{slug}_module"
    if cls not in init_text:
        init_text = init_text.replace(
            "__all__ = [", f"from {module_name} import {cls}  # noqa: F401\n\n__all__ = ["
        ).replace("__all__ = [", f'__all__ = [\n    "{cls}",', 1)
        init_path.write_text(init_text)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold a new AFYA-PREDICT disease module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  python scripts/add_new_disease.py --name "Dengue Fever" --code DEN '
            "--transmission vector_borne\n"
            '  python scripts/add_new_disease.py --name "Rift Valley Fever" --code RVF '
            "--transmission zoonotic\n"
            "  python scripts/add_new_disease.py --validate dengue_fever\n"
        ),
    )
    parser.add_argument("--name", help='Display name, e.g. "Dengue Fever"')
    parser.add_argument("--code", help="Short code, e.g. DEN")
    parser.add_argument("--transmission", choices=TRANSMISSION_MODES, default="other")
    parser.add_argument("--vector", default="", help="Vector species, if applicable")
    parser.add_argument("--pathogen", default="", help="Causative organism")
    parser.add_argument("--preset", choices=sorted(PRESETS),
                        help="Starting proxy set (defaults to the transmission mode)")
    parser.add_argument("--validate", metavar="SLUG",
                        help="Validate an existing disease instead of creating one")
    parser.add_argument("--no-notebook", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    if args.validate:
        return validate_only(args.validate)

    if not args.name or not args.code:
        parser.error("--name and --code are required (or use --validate SLUG)")

    slug = slugify(args.name)
    cls = class_name(args.name)
    config_path = Path(DISEASE_CONFIG_DIR) / f"{slug}.yaml"
    module_path = MODULE_DIR / f"{slug}_module.py"

    for path in (config_path, module_path):
        if path.exists() and not args.force:
            print(f"ERROR: {path} already exists. Use --force to overwrite.")
            return 1

    config_path.write_text(
        render_config(args.name, args.code.upper(), args.transmission, args.preset,
                      args.vector, args.pathogen)
    )
    print(f"  wrote {config_path.relative_to(REPO_ROOT)}")

    module_path.write_text(render_module(args.name, slug, cls))
    print(f"  wrote {module_path.relative_to(REPO_ROOT)}")

    if register(slug, cls):
        print(f"  registered {cls} in {REGISTRY_PATH.relative_to(REPO_ROOT)}")

    if not args.no_notebook:
        notebook_path = NOTEBOOK_DIR / f"explore_{slug}.ipynb"
        notebook_path.write_text(render_notebook(args.name, slug))
        print(f"  wrote {notebook_path.relative_to(REPO_ROOT)}")

    print(f"\n{args.name} scaffolded. Now:\n")
    print(f"  1. Fill in the TODOs in config/diseases/{slug}.yaml -")
    print("     proxies with mechanisms, alert thresholds, response recommendations.")
    print(f"  2. Add 'cases_{slug}' to CASE_VARIABLES in")
    print("     src/data_ingestion/adapters/dhis2_surveillance.py and map it to your")
    print("     DHIS2 data element.")
    print(f"  3. python scripts/add_new_disease.py --validate {slug}")
    print(f"  4. jupyter lab notebooks/explore_{slug}.ipynb")
    print(f"  5. python scripts/run_backtest.py --disease {slug}")
    print("\n  The model must beat all three naive baselines before deployment (rule #10).")
    return 0


def validate_only(slug: str) -> int:
    """Check a disease's config and module against the platform contract."""
    from src.core.config_loader import list_disease_configs, load_disease_config, validate_disease_config
    from src.models.registry import DISEASE_MODULE_REGISTRY, build_module

    print(f"Validating {slug!r}\n")
    if slug not in list_disease_configs():
        print(f"  FAIL  no config at config/diseases/{slug}.yaml")
        print(f"        available: {', '.join(list_disease_configs())}")
        return 1

    config = load_disease_config(slug)
    problems = validate_disease_config(config)
    print(f"  config      : {config.name} ({config.code})")
    print(f"  proxies     : {len(config.digital_proxies)}")
    print(f"  sources     : {len(config.required_sources)} non-optional "
          f"({', '.join(config.required_sources)})")
    print(f"  horizon     : {config.model.forecast_horizon_weeks} weeks")
    print(f"  spatial     : {'enabled' if config.spatial.enabled else 'disabled'}")

    registered = slug in DISEASE_MODULE_REGISTRY
    print(f"  module class: {DISEASE_MODULE_REGISTRY[slug].__name__ if registered else 'none (falls back to StandardDiseaseModule)'}")

    todos: List[str] = []
    raw = (Path(DISEASE_CONFIG_DIR) / f"{slug}.yaml").read_text()
    if "TODO" in raw:
        todos = [line.strip() for line in raw.splitlines() if "TODO" in line]

    required = ["get_feature_config", "build_feature_matrix", "train", "predict",
                "detect_outbreak", "get_spatial_risk", "generate_recommendations"]
    try:
        module = build_module(slug)
        missing = [m for m in required
                   if getattr(getattr(type(module), m, None), "__isabstractmethod__", False)]
    except Exception as exc:
        print(f"\n  FAIL  module could not be built: {exc}")
        return 1

    print()
    if problems:
        print(f"  {len(problems)} config problem(s):")
        for problem in problems:
            print(f"    - {problem}")
    if missing:
        print(f"  {len(missing)} unimplemented interface method(s): {', '.join(missing)}")
    if todos:
        print(f"  {len(todos)} unresolved TODO(s) in the config:")
        for todo in todos[:6]:
            print(f"    - {todo}")

    if problems or missing:
        print("\n  RESULT: NOT READY")
        return 1
    if todos:
        print("\n  RESULT: valid, but TODOs remain - review before deploying")
        return 0
    print("  RESULT: READY - run scripts/run_backtest.py next")
    return 0


if __name__ == "__main__":
    sys.exit(main())
