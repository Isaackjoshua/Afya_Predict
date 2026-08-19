"""SARIMA wrapper with a seasonal-naive fallback.

The ensemble includes a classical seasonal time-series model alongside the
tree learners because they fail differently: gradient boosting extrapolates
poorly beyond its training range, while SARIMA carries the seasonal shape but
ignores the digital proxies entirely. Averaging them is what makes the ensemble
robust when one component drifts (shortcoming #3).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.core.logging import get_logger

log = get_logger("models.sarima")


class SarimaRegressor:
    """`fit(X, y)`-compatible SARIMA on the target series alone.

    The design matrix is ignored except for its row ordering: the model is
    univariate by construction. Exogenous regressors are deliberately not used
    so the ensemble keeps a genuinely independent view of the series.
    """

    def __init__(
        self,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 52),
        season_length: int = 52,
        **_ignored,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.season_length = season_length
        self.result_ = None
        self.history_: Optional[np.ndarray] = None
        self.fallback_: bool = False

    def fit(self, X, y, sample_weight=None) -> "SarimaRegressor":
        y = np.asarray(y, dtype=float).ravel()
        self.history_ = y
        # Below two seasons SARIMA cannot identify the seasonal component.
        if len(y) < 2 * self.season_length:
            self.fallback_ = True
            return self
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.result_ = model.fit(disp=False)
            self.fallback_ = False
        except Exception as exc:  # noqa: BLE001
            log.warning("SARIMA fit failed (%s); using the seasonal-naive fallback", exc)
            self.fallback_ = True
        return self

    def predict(self, X) -> np.ndarray:
        n = len(X)
        if self.history_ is None:
            return np.zeros(n)
        if not self.fallback_ and self.result_ is not None:
            try:
                forecast = np.asarray(self.result_.forecast(steps=max(n, 1)), dtype=float)
                if len(forecast) >= n:
                    return forecast[:n]
                return np.concatenate([forecast, np.repeat(forecast[-1], n - len(forecast))])
            except Exception as exc:  # noqa: BLE001
                log.warning("SARIMA forecast failed (%s); using the seasonal-naive fallback", exc)
        return self._seasonal_naive(n)

    def _seasonal_naive(self, n: int) -> np.ndarray:
        """Repeat the value from one season ago; the classic hard-to-beat baseline."""
        history = self.history_
        if len(history) >= self.season_length:
            tail = history[-self.season_length:]
            return np.array([tail[i % self.season_length] for i in range(n)], dtype=float)
        return np.repeat(float(np.mean(history)), n)

    def get_params(self, deep: bool = True) -> dict:
        return {"order": self.order, "seasonal_order": self.seasonal_order,
                "season_length": self.season_length}

    def set_params(self, **params) -> "SarimaRegressor":
        for key, value in params.items():
            setattr(self, key, value)
        return self
