"""Feature engineering (shortcomings #5, #8, #10).

The pipeline turns a fused `district x week` panel into a model-ready design
matrix. The defining choice here is that **lags are fitted, not assumed**
(critical rule #3): the disease YAML supplies a search range and a starting
estimate, and `lag_features` finds the lag that actually maximises the
district's own signal.
"""

from src.feature_engineering.builder import FeatureMatrix, FeatureBuilder  # noqa: F401
from src.feature_engineering.lag_features import (  # noqa: F401
    LagFit,
    add_lag_features,
    fit_optimal_lags,
)
