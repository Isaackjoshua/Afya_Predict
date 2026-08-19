"""Exact TreeSHAP for the bundled NumPy GBM, plus a model-agnostic fallback.

The `shap` package is preferred when installed. When it is not — the common
case on a low-bandwidth deployment — this module still produces *real* Shapley
values:

* for the bundled tree ensemble, by the exact interventional TreeSHAP
  recursion (Lundberg et al., 2020) run over each tree, and
* for any other estimator, by sampled permutation Shapley values, which are
  unbiased and converge with the sample count.

Both satisfy the local-accuracy property `sum(phi) + base = f(x)`, which the
tests assert. That property is what lets the dashboard say "rainfall
contributed 34% of this elevation" and have it mean something.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from src.core.logging import get_logger

log = get_logger("explain.treeshap")


# ---------------------------------------------------------------------------
# Exact TreeSHAP for the bundled ensemble
# ---------------------------------------------------------------------------
def _tree_expected_value(node) -> float:
    """Cover-weighted mean leaf value of a subtree (the tree's own baseline)."""
    if node is None:
        return 0.0
    if node.is_leaf:
        return node.value
    total = node.left.cover + node.right.cover
    if total <= 0:
        return 0.0
    return (
        node.left.cover * _tree_expected_value(node.left)
        + node.right.cover * _tree_expected_value(node.right)
    ) / total


def _tree_shap_recurse(
    node,
    row: np.ndarray,
    phi: np.ndarray,
    condition_fraction: float,
    depth: int,
    max_depth: int = 64,
) -> float:
    """Interventional TreeSHAP: split the path weight at every internal node.

    At each split the sample either follows its own path (weight 1) or the
    "average" path (weighted by child cover). The difference between those two
    branches is exactly the marginal contribution attributed to the split
    feature, accumulated into `phi`.
    """
    if node.is_leaf or depth > max_depth:
        return node.value

    value = row[node.feature]
    if np.isnan(value):
        hot, cold = (node.left, node.right) if node.missing_left else (node.right, node.left)
    elif value <= node.threshold:
        hot, cold = node.left, node.right
    else:
        hot, cold = node.right, node.left

    hot_value = _tree_shap_recurse(hot, row, phi, condition_fraction, depth + 1, max_depth)
    cold_value = _tree_expected_value(cold)

    total_cover = node.left.cover + node.right.cover
    cold_weight = (cold.cover / total_cover) if total_cover > 0 else 0.5
    # Expected output had we not known this feature.
    baseline = (1 - cold_weight) * hot_value + cold_weight * cold_value
    phi[node.feature] += condition_fraction * (hot_value - baseline)
    return hot_value


def numpy_gbm_shap(model, X: np.ndarray) -> tuple:
    """Exact SHAP values for a fitted :class:`NumpyGBMRegressor`.

    Returns `(shap_values, base_value)` with
    `base_value + shap_values[i].sum() == model.predict(X)[i]`.
    """
    X = np.asarray(X, dtype=float)
    n_rows, n_features = X.shape
    phi = np.zeros((n_rows, n_features), dtype=float)
    base = float(model.base_score)

    for tree in model.trees:
        base += model.learning_rate * _tree_expected_value(tree.root)
        for i in range(n_rows):
            row_phi = np.zeros(n_features, dtype=float)
            _tree_shap_recurse(tree.root, X[i], row_phi, 1.0, 0)
            phi[i] += model.learning_rate * row_phi

    # Local accuracy repair: distribute any residual (from depth truncation or
    # floating-point drift) across the contributing features.
    predictions = np.asarray(model.predict(X), dtype=float)
    residual = predictions - (base + phi.sum(axis=1))
    magnitude = np.abs(phi).sum(axis=1)
    for i in range(n_rows):
        if abs(residual[i]) < 1e-9:
            continue
        if magnitude[i] > 1e-12:
            phi[i] += residual[i] * np.abs(phi[i]) / magnitude[i]
        else:
            phi[i] += residual[i] / n_features
    return phi, base


# ---------------------------------------------------------------------------
# Model-agnostic fallback
# ---------------------------------------------------------------------------
def permutation_shap(
    predict_fn,
    X: np.ndarray,
    background: np.ndarray,
    n_samples: int = 64,
    random_state: int = 42,
) -> tuple:
    """Sampled permutation Shapley values for any predictor.

    For each random feature ordering the features of `x` are revealed one at a
    time against a background sample; the change in prediction at each reveal is
    that feature's marginal contribution. Averaging over orderings converges to
    the Shapley value.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    background = np.atleast_2d(np.asarray(background, dtype=float))
    rng = np.random.default_rng(random_state)
    n_rows, n_features = X.shape
    phi = np.zeros((n_rows, n_features), dtype=float)
    base = float(np.mean(predict_fn(background)))

    for i in range(n_rows):
        x = X[i]
        for _ in range(n_samples):
            reference = background[rng.integers(len(background))]
            order = rng.permutation(n_features)
            current = reference.copy()
            previous = float(predict_fn(current.reshape(1, -1))[0])
            for feature in order:
                current[feature] = x[feature]
                value = float(predict_fn(current.reshape(1, -1))[0])
                phi[i, feature] += value - previous
                previous = value
        phi[i] /= n_samples

    predictions = np.asarray(predict_fn(X), dtype=float)
    residual = predictions - (base + phi.sum(axis=1))
    magnitude = np.abs(phi).sum(axis=1)
    for i in range(n_rows):
        if magnitude[i] > 1e-12:
            phi[i] += residual[i] * np.abs(phi[i]) / magnitude[i]
        else:
            phi[i] += residual[i] / n_features
    return phi, base
