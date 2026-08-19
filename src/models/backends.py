"""Regressor backends with graceful degradation.

Preference order for a gradient-boosted model:

    XGBoost -> LightGBM -> scikit-learn HistGradientBoosting -> NumpyGBMRegressor

The last is bundled with the platform (see :mod:`src.models.gbm`), so a
district-level deployment can train and serve with nothing but NumPy and pandas
installed. Which backend actually ran is recorded on the fitted model and
surfaced in `model_version`, so a forecast is always traceable to the code that
produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.logging import get_logger

log = get_logger("models.backends")


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


HAS_XGBOOST = _has("xgboost")
HAS_LIGHTGBM = _has("lightgbm")
HAS_SKLEARN = _has("sklearn")
HAS_STATSMODELS = _has("statsmodels")


@dataclass
class BackendInfo:
    """What actually produced a set of predictions."""

    requested: str
    resolved: str
    library: str
    version: str = "bundled"

    def __str__(self) -> str:
        return f"{self.resolved}({self.library}=={self.version})"


def _library_version(name: str) -> str:
    try:
        import importlib.metadata as metadata

        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _build_xgboost(params: Dict[str, Any]):
    import xgboost as xgb

    return xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 5),
        subsample=params.get("subsample", 0.85),
        colsample_bytree=params.get("colsample", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        min_child_weight=params.get("min_samples_leaf", 8),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", 2),
        tree_method="hist",
        objective="reg:squarederror",
    )


def _build_lightgbm(params: Dict[str, Any]):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 5),
        subsample=params.get("subsample", 0.85),
        colsample_bytree=params.get("colsample", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        min_child_samples=params.get("min_samples_leaf", 8),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", 2),
        verbose=-1,
    )


def _build_sklearn_hist(params: Dict[str, Any]):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 5),
        min_samples_leaf=params.get("min_samples_leaf", 8),
        l2_regularization=params.get("reg_lambda", 1.0),
        random_state=params.get("random_state", 42),
    )


def _build_numpy_gbm(params: Dict[str, Any]):
    from src.models.gbm import NumpyGBMRegressor

    return NumpyGBMRegressor(
        n_estimators=params.get("n_estimators", 150),
        learning_rate=params.get("learning_rate", 0.07),
        max_depth=params.get("max_depth", 4),
        min_samples_leaf=params.get("min_samples_leaf", 8),
        reg_lambda=params.get("reg_lambda", 1.0),
        subsample=params.get("subsample", 0.85),
        colsample=params.get("colsample", 0.8),
        random_state=params.get("random_state", 42),
    )


def _build_ridge(params: Dict[str, Any]):
    """Linear fallback — fast, stable and hard to overfit on short histories."""
    if HAS_SKLEARN:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=params.get("alpha", 1.0))),
            ]
        )
    return NumpyRidge(alpha=params.get("alpha", 1.0))


#: name -> (builder, library, availability flag)
_BACKENDS: Dict[str, tuple] = {
    "xgboost": (_build_xgboost, "xgboost", lambda: HAS_XGBOOST),
    "lightgbm": (_build_lightgbm, "lightgbm", lambda: HAS_LIGHTGBM),
    "sklearn_hist": (_build_sklearn_hist, "scikit-learn", lambda: HAS_SKLEARN),
    "numpy_gbm": (_build_numpy_gbm, "afya-predict", lambda: True),
    "ridge": (_build_ridge, "scikit-learn" if HAS_SKLEARN else "afya-predict", lambda: True),
}

#: Fallback chain applied when the requested backend is unavailable.
FALLBACK_CHAIN: List[str] = ["xgboost", "lightgbm", "sklearn_hist", "numpy_gbm"]


def available_backends() -> List[str]:
    return [name for name, (_, _, check) in _BACKENDS.items() if check()]


def resolve_backend(requested: str) -> str:
    """Return the first available backend at or below `requested`."""
    requested = (requested or "xgboost").lower()
    if requested in _BACKENDS and _BACKENDS[requested][2]():
        return requested
    if requested in ("sarima", "arima"):
        return "sarima" if HAS_STATSMODELS else "ridge"
    start = FALLBACK_CHAIN.index(requested) if requested in FALLBACK_CHAIN else 0
    for name in FALLBACK_CHAIN[start:]:
        if _BACKENDS[name][2]():
            log.debug("backend %s unavailable; using %s", requested, name)
            return name
    return "numpy_gbm"


def build_regressor(name: str, **params) -> tuple:
    """Instantiate a regressor, returning `(estimator, BackendInfo)`."""
    resolved = resolve_backend(name)
    if resolved == "sarima":
        from src.models.sarima import SarimaRegressor

        return SarimaRegressor(**params), BackendInfo(
            requested=name, resolved="sarima", library="statsmodels",
            version=_library_version("statsmodels"),
        )
    builder, library, _ = _BACKENDS[resolved]
    estimator = builder(params)
    version = _library_version(library) if library != "afya-predict" else "bundled"
    return estimator, BackendInfo(
        requested=name, resolved=resolved, library=library, version=version
    )


def feature_importances(estimator) -> Optional[np.ndarray]:
    """Best-effort native importance vector from any backend."""
    for attribute in ("feature_importances_", "coef_"):
        values = getattr(estimator, attribute, None)
        if values is not None:
            return np.abs(np.asarray(values, dtype=float)).ravel()
    inner = getattr(estimator, "named_steps", {}).get("model") if hasattr(estimator, "named_steps") else None
    if inner is not None:
        return feature_importances(inner)
    return None


class NumpyRidge:
    """Ridge regression with median imputation and standardisation, in NumPy."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.medians_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.scales_: Optional[np.ndarray] = None

    def _prepare(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        if fit:
            self.medians_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
            self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        assert self.medians_ is not None
        missing = ~np.isfinite(X)
        X[missing] = np.take(self.medians_, np.where(missing)[1])
        if fit:
            self.means_ = X.mean(axis=0)
            scales = X.std(axis=0)
            self.scales_ = np.where(scales > 0, scales, 1.0)
        return (X - self.means_) / self.scales_

    def fit(self, X, y, sample_weight=None) -> "NumpyRidge":
        Z = self._prepare(X, fit=True)
        y = np.asarray(y, dtype=float).ravel()
        n_features = Z.shape[1]
        gram = Z.T @ Z + self.alpha * np.eye(n_features)
        self.coef_ = np.linalg.solve(gram, Z.T @ (y - y.mean()))
        self.intercept_ = float(y.mean())
        return self

    def predict(self, X) -> np.ndarray:
        Z = self._prepare(X, fit=False)
        return Z @ self.coef_ + self.intercept_

    def get_params(self, deep: bool = True) -> dict:
        return {"alpha": self.alpha}

    def set_params(self, **params) -> "NumpyRidge":
        for key, value in params.items():
            setattr(self, key, value)
        return self


def backend_report() -> Dict[str, Any]:
    """What this installation can actually run — surfaced by `GET /admin/health`."""
    return {
        "available": available_backends(),
        "xgboost": HAS_XGBOOST,
        "lightgbm": HAS_LIGHTGBM,
        "sklearn": HAS_SKLEARN,
        "statsmodels": HAS_STATSMODELS,
        "default_resolved": resolve_backend("xgboost"),
    }
