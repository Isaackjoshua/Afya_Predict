"""Rigorous, out-of-sample evaluation (shortcoming #13, critical rule #10).

A 2026 Frontiers review found only 14% of LMIC AI health models were externally
validated; most were tested on the data they were fitted on. This package makes
honest evaluation the default path: walk-forward validation, outbreak-detection
metrics at the operational threshold, calibration, spatial accuracy, and a
benchmark gate that a model must clear against three naive baselines before it
is allowed to deploy.
"""

from src.evaluation.benchmark import BaselineComparison, benchmark_against_naive  # noqa: F401
from src.evaluation.metrics import (  # noqa: F401
    classification_metrics,
    regression_metrics,
    skill_score,
)
from src.evaluation.walk_forward_cv import WalkForwardCV, WalkForwardResult  # noqa: F401
