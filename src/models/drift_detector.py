"""Concept-drift detection (shortcoming #3).

Google Flu Trends was fitted on 2003-2008 data and never refitted. Google's own
search ranking changed 86 times in two months in 2012, silently breaking the
model's inputs; by 2013 media-driven panic searching had it overestimating by
140%. Nobody noticed automatically, because nothing was watching the residuals.

Something watches here. Two complementary detectors run on the prediction-error
stream:

* **Page-Hinkley** — cumulative-sum test, sensitive to a sustained shift in the
  mean error (a model that has quietly started running high).
* **ADWIN** — adaptive windowing, which compares recent and older sub-windows
  and detects a distribution change without a prior on its size.

Either firing triggers a refit on the latest rolling window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Sequence

from collections import deque

import numpy as np

from src.core.logging import get_logger

log = get_logger("models.drift")


@dataclass
class DriftEvent:
    """A detected change in the error distribution."""

    detector: str
    index: int
    statistic: float
    threshold: float
    mean_before: float
    mean_after: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "index": self.index,
            "statistic": round(self.statistic, 5),
            "threshold": round(self.threshold, 5),
            "mean_before": round(self.mean_before, 5),
            "mean_after": round(self.mean_after, 5),
            "detected_at": self.detected_at.isoformat(timespec="seconds"),
            "message": self.message,
        }


class PageHinkley:
    """Page-Hinkley change detector on a stream of scalar errors.

    Tracks the cumulative deviation of each observation from the running mean,
    less a tolerance `delta`; when the gap between that cumulative sum and its
    own minimum exceeds `threshold`, a sustained upward shift has occurred.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.9999) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.detected = False

    def update(self, value: float) -> bool:
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cumulative = self.alpha * self.cumulative + (value - self.mean - self.delta)
        self.minimum = min(self.minimum, self.cumulative)
        self.detected = (self.cumulative - self.minimum) > self.threshold
        return self.detected

    @property
    def statistic(self) -> float:
        return self.cumulative - self.minimum


class ADWIN:
    """Adaptive windowing drift detector (simplified exact-window variant).

    Keeps a window of recent values and, at each update, tests every split point
    for a significant difference in sub-window means using a Hoeffding-style
    bound. When one is found, the older sub-window is dropped — the window
    itself adapts to the current concept.
    """

    def __init__(self, delta: float = 0.002, max_window: int = 512, min_sub_window: int = 8) -> None:
        self.delta = delta
        self.max_window = max_window
        self.min_sub_window = min_sub_window
        self.window: Deque[float] = deque(maxlen=max_window)
        self.detected = False
        self.last_split: Optional[int] = None

    def reset(self) -> None:
        self.window.clear()
        self.detected = False
        self.last_split = None

    def update(self, value: float) -> bool:
        self.window.append(float(value))
        self.detected = False
        n = len(self.window)
        if n < 2 * self.min_sub_window:
            return False

        values = np.array(self.window, dtype=float)
        variance = float(np.var(values)) or 1e-9
        for split in range(self.min_sub_window, n - self.min_sub_window + 1):
            left, right = values[:split], values[split:]
            n0, n1 = len(left), len(right)
            harmonic = 1.0 / (1.0 / n0 + 1.0 / n1)
            # Hoeffding bound with a variance term (Bifet & Gavalda, 2007).
            log_term = np.log(2.0 * np.log(n) / self.delta) if n > 2 else np.log(2.0 / self.delta)
            epsilon = np.sqrt(2.0 * variance * log_term / harmonic) + 2.0 / (3.0 * harmonic) * log_term
            if abs(left.mean() - right.mean()) > epsilon:
                self.detected = True
                self.last_split = split
                # Drop the stale concept.
                for _ in range(split):
                    self.window.popleft()
                return True
        return False

    @property
    def width(self) -> int:
        return len(self.window)


class DriftDetector:
    """Run both detectors over a residual stream and report any change."""

    def __init__(
        self,
        page_hinkley_threshold: float = 25.0,
        page_hinkley_delta: float = 0.005,
        adwin_delta: float = 0.002,
        min_observations: int = 20,
        normalise: bool = True,
    ) -> None:
        self.page_hinkley = PageHinkley(delta=page_hinkley_delta, threshold=page_hinkley_threshold)
        self.adwin = ADWIN(delta=adwin_delta)
        self.min_observations = min_observations
        self.normalise = normalise
        self.events: List[DriftEvent] = []
        self.n_seen = 0
        self._scale: Optional[float] = None

    def reset(self) -> None:
        self.page_hinkley.reset()
        self.adwin.reset()
        self.events.clear()
        self.n_seen = 0
        self._scale = None

    # -- streaming ---------------------------------------------------------
    def update(self, residual: float, history: Optional[Sequence[float]] = None) -> Optional[DriftEvent]:
        """Feed one residual; returns a `DriftEvent` when drift is detected."""
        self.n_seen += 1
        value = self._scaled(residual, history)

        if self.page_hinkley.update(value) and self.n_seen >= self.min_observations:
            event = DriftEvent(
                detector="page_hinkley",
                index=self.n_seen,
                statistic=self.page_hinkley.statistic,
                threshold=self.page_hinkley.threshold,
                mean_before=self.page_hinkley.mean,
                mean_after=value,
                message=(
                    "Page-Hinkley detected a sustained shift in prediction error — the model is "
                    "drifting away from the data it was fitted on."
                ),
            )
            self.events.append(event)
            self.page_hinkley.reset()
            return event

        if self.adwin.update(value) and self.n_seen >= self.min_observations:
            split = self.adwin.last_split or 0
            event = DriftEvent(
                detector="adwin",
                index=self.n_seen,
                statistic=float(split),
                threshold=float(self.adwin.delta),
                mean_before=0.0,
                mean_after=value,
                message=(
                    "ADWIN detected a change in the error distribution — recent errors no longer "
                    "look like older ones."
                ),
            )
            self.events.append(event)
            return event
        return None

    def scan(self, residuals: Sequence[float]) -> List[DriftEvent]:
        """Replay a whole residual series; used by the retraining scheduler."""
        self.reset()
        scale = float(np.std(residuals)) if self.normalise else 1.0
        self._scale = scale if scale > 1e-9 else 1.0
        for residual in residuals:
            self.update(float(residual))
        return list(self.events)

    def _scaled(self, residual: float, history: Optional[Sequence[float]]) -> float:
        if not self.normalise:
            return float(residual)
        if self._scale is None:
            values = list(history or [])
            scale = float(np.std(values)) if len(values) > 3 else 1.0
            self._scale = scale if scale > 1e-9 else 1.0
        return float(residual) / self._scale

    # -- summary -----------------------------------------------------------
    @property
    def drift_detected(self) -> bool:
        return bool(self.events)

    def summary(self) -> Dict[str, object]:
        return {
            "drift_detected": self.drift_detected,
            "observations": self.n_seen,
            "events": [e.to_dict() for e in self.events],
            "detectors": ["page_hinkley", "adwin"],
        }


def residual_series(actual: Sequence[float], predicted: Sequence[float]) -> np.ndarray:
    """Signed prediction error, the quantity both detectors monitor."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    return actual[mask] - predicted[mask]


def detect_drift(
    actual: Sequence[float], predicted: Sequence[float], min_observations: int = 20, **kwargs
) -> Dict[str, object]:
    """Convenience wrapper: residuals in, drift verdict out."""
    residuals = residual_series(actual, predicted)
    if len(residuals) < min_observations:
        return {
            "drift_detected": False,
            "observations": int(len(residuals)),
            "events": [],
            "reason": f"only {len(residuals)} residuals; need {min_observations}",
        }
    detector = DriftDetector(min_observations=min_observations, **kwargs)
    detector.scan(residuals)
    return detector.summary()
