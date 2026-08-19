"""A dependency-free histogram gradient-boosted regression tree.

Why this exists
---------------
AFYA-PREDICT must run "on commodity hardware" in districts where installing a
57 MB XGBoost wheel over an intermittent link is not realistic (shortcoming #9),
and the offline/edge deployment (rule #6) needs a model with no native
dependencies at all. So the model layer prefers XGBoost -> LightGBM ->
scikit-learn when they are installed, and falls back to this NumPy
implementation when none of them are.

It is a genuine implementation, not a stub: histogram binning, depth-limited
best-first splits, L2 regularisation, shrinkage, row/column subsampling and
exact per-node gain — the same algorithm family, just smaller and slower. It
also exposes the node structure that :mod:`src.explainability.tree_shap` needs
to compute exact Shapley values without the `shap` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

MISSING_BIN = 0  # bin 0 is reserved for NaN, so missing values are learnable


@dataclass
class TreeNode:
    """One node of a regression tree."""

    is_leaf: bool = False
    value: float = 0.0
    feature: int = -1
    threshold: float = 0.0
    missing_left: bool = True
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    n_samples: int = 0
    cover: float = 0.0

    def predict_row(self, row: np.ndarray) -> float:
        node = self
        while not node.is_leaf:
            value = row[node.feature]
            if np.isnan(value):
                node = node.left if node.missing_left else node.right
            else:
                node = node.left if value <= node.threshold else node.right
        return node.value


class HistogramRegressionTree:
    """Depth-limited regression tree fitted on pre-binned features."""

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 8,
        reg_lambda: float = 1.0,
        min_gain: float = 1e-7,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.min_gain = min_gain
        self.root: Optional[TreeNode] = None

    def fit(
        self,
        binned: np.ndarray,
        residuals: np.ndarray,
        bin_edges: List[np.ndarray],
        feature_subset: Optional[np.ndarray] = None,
    ) -> "HistogramRegressionTree":
        indices = np.arange(binned.shape[0])
        self.root = self._build(binned, residuals, bin_edges, indices, 0, feature_subset)
        return self

    def _leaf_value(self, residuals: np.ndarray) -> float:
        # Newton step for squared error with L2 regularisation.
        return float(residuals.sum() / (len(residuals) + self.reg_lambda))

    def _build(
        self,
        binned: np.ndarray,
        residuals: np.ndarray,
        bin_edges: List[np.ndarray],
        indices: np.ndarray,
        depth: int,
        feature_subset: Optional[np.ndarray],
    ) -> TreeNode:
        node_residuals = residuals[indices]
        if (
            depth >= self.max_depth
            or len(indices) < 2 * self.min_samples_leaf
            or np.allclose(node_residuals, node_residuals[0])
        ):
            return TreeNode(
                is_leaf=True,
                value=self._leaf_value(node_residuals),
                n_samples=len(indices),
                cover=float(len(indices)),
            )

        best = self._best_split(binned, residuals, indices, feature_subset)
        if best is None:
            return TreeNode(
                is_leaf=True,
                value=self._leaf_value(node_residuals),
                n_samples=len(indices),
                cover=float(len(indices)),
            )

        feature, bin_threshold, gain, missing_left = best
        column = binned[indices, feature]
        missing = column == MISSING_BIN
        goes_left = (column <= bin_threshold) & ~missing
        if missing_left:
            goes_left = goes_left | missing

        left_idx, right_idx = indices[goes_left], indices[~goes_left]
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return TreeNode(
                is_leaf=True,
                value=self._leaf_value(node_residuals),
                n_samples=len(indices),
                cover=float(len(indices)),
            )

        edges = bin_edges[feature]
        # bin b covers (edges[b-2], edges[b-1]] once bin 0 is reserved for NaN.
        threshold = float(edges[min(bin_threshold - 1, len(edges) - 1)]) if len(edges) else 0.0
        return TreeNode(
            is_leaf=False,
            feature=int(feature),
            threshold=threshold,
            missing_left=bool(missing_left),
            n_samples=len(indices),
            cover=float(len(indices)),
            left=self._build(binned, residuals, bin_edges, left_idx, depth + 1, feature_subset),
            right=self._build(binned, residuals, bin_edges, right_idx, depth + 1, feature_subset),
        )

    def _best_split(
        self,
        binned: np.ndarray,
        residuals: np.ndarray,
        indices: np.ndarray,
        feature_subset: Optional[np.ndarray],
    ) -> Optional[Tuple[int, int, float, bool]]:
        node_residuals = residuals[indices]
        total_sum = float(node_residuals.sum())
        total_count = len(indices)
        parent = total_sum**2 / (total_count + self.reg_lambda)
        n_bins = int(binned.max()) + 1

        features = (
            feature_subset if feature_subset is not None else np.arange(binned.shape[1])
        )
        best: Optional[Tuple[int, int, float, bool]] = None
        for feature in features:
            column = binned[indices, feature]
            sums = np.bincount(column, weights=node_residuals, minlength=n_bins)
            counts = np.bincount(column, minlength=n_bins)
            missing_sum, missing_count = sums[MISSING_BIN], counts[MISSING_BIN]

            present_sums = sums[1:]
            present_counts = counts[1:]
            if present_counts.sum() < 2 * self.min_samples_leaf:
                continue
            cumulative_sum = np.cumsum(present_sums)
            cumulative_count = np.cumsum(present_counts)

            for missing_left in (True, False):
                left_sum = cumulative_sum + (missing_sum if missing_left else 0.0)
                left_count = cumulative_count + (missing_count if missing_left else 0)
                right_sum = total_sum - left_sum
                right_count = total_count - left_count

                valid = (left_count >= self.min_samples_leaf) & (
                    right_count >= self.min_samples_leaf
                )
                if not valid.any():
                    continue
                gains = np.full(len(left_sum), -np.inf)
                gains[valid] = (
                    left_sum[valid] ** 2 / (left_count[valid] + self.reg_lambda)
                    + right_sum[valid] ** 2 / (right_count[valid] + self.reg_lambda)
                    - parent
                )
                position = int(np.argmax(gains))
                gain = float(gains[position])
                if gain > self.min_gain and (best is None or gain > best[2]):
                    # +1 because present bins start at index 1.
                    best = (int(feature), position + 1, gain, missing_left)
        return best

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            return np.zeros(len(X))
        return np.array([self.root.predict_row(row) for row in X], dtype=float)


class NumpyGBMRegressor:
    """Gradient-boosted regression trees with a scikit-learn-style interface."""

    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        min_samples_leaf: int = 8,
        reg_lambda: float = 1.0,
        subsample: float = 0.85,
        colsample: float = 0.8,
        max_bins: int = 32,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample = colsample
        self.max_bins = max_bins
        self.random_state = random_state

        self.trees: List[HistogramRegressionTree] = []
        self.base_score: float = 0.0
        self.bin_edges: List[np.ndarray] = []
        self.n_features_in_: int = 0
        self.feature_importances_: np.ndarray = np.array([])
        self.train_loss_: List[float] = field(default_factory=list)  # type: ignore[assignment]

    # -- binning ----------------------------------------------------------
    def _fit_bins(self, X: np.ndarray) -> None:
        self.bin_edges = []
        quantiles = np.linspace(0, 100, self.max_bins)
        for j in range(X.shape[1]):
            column = X[:, j]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                self.bin_edges.append(np.array([0.0]))
                continue
            edges = np.unique(np.percentile(finite, quantiles))
            if edges.size < 2:
                edges = np.array([finite[0], finite[0] + 1e-9])
            self.bin_edges.append(edges)

    def _bin(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(X.shape, dtype=np.int32)
        for j in range(X.shape[1]):
            column = X[:, j]
            edges = self.bin_edges[j]
            # searchsorted gives 0..len(edges); +1 keeps bin 0 free for NaN.
            bins = np.searchsorted(edges, column, side="left") + 1
            bins = np.clip(bins, 1, len(edges) + 1)
            bins[~np.isfinite(column)] = MISSING_BIN
            out[:, j] = bins
        return out

    # -- fit / predict -----------------------------------------------------
    def fit(self, X, y, sample_weight=None) -> "NumpyGBMRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        self._fit_bins(X)
        binned = self._bin(X)
        self.base_score = float(np.mean(y))
        predictions = np.full(len(y), self.base_score, dtype=float)

        importances = np.zeros(self.n_features_in_, dtype=float)
        self.trees = []
        self.train_loss_ = []

        n_rows = len(y)
        n_subsample = max(self.min_samples_leaf * 3, int(self.subsample * n_rows))
        n_columns = max(1, int(self.colsample * self.n_features_in_))

        for _ in range(self.n_estimators):
            residuals = y - predictions
            if sample_weight is not None:
                residuals = residuals * np.asarray(sample_weight, dtype=float)

            rows = (
                rng.choice(n_rows, size=min(n_subsample, n_rows), replace=False)
                if n_subsample < n_rows
                else np.arange(n_rows)
            )
            columns = (
                rng.choice(self.n_features_in_, size=n_columns, replace=False)
                if n_columns < self.n_features_in_
                else np.arange(self.n_features_in_)
            )

            tree = HistogramRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                reg_lambda=self.reg_lambda,
            )
            sub_residuals = np.zeros(n_rows)
            sub_residuals[rows] = residuals[rows]
            tree.fit(binned[rows], residuals[rows], self.bin_edges, feature_subset=columns)
            # Re-index: the tree was fitted on a row subset but predicts on all.
            step = self.learning_rate * tree.predict(X)
            predictions += step
            self.trees.append(tree)
            _accumulate_importance(tree.root, importances)
            self.train_loss_.append(float(np.mean((y - predictions) ** 2)))

        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), self.base_score, dtype=float)
        for tree in self.trees:
            out += self.learning_rate * tree.predict(X)
        return out

    def staged_predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), self.base_score, dtype=float)
        for tree in self.trees:
            out = out + self.learning_rate * tree.predict(X)
            yield out.copy()

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "colsample": self.colsample,
            "max_bins": self.max_bins,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "NumpyGBMRegressor":
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _accumulate_importance(node: Optional[TreeNode], importances: np.ndarray) -> None:
    """Split-count importance weighted by the samples each split covered."""
    if node is None or node.is_leaf:
        return
    importances[node.feature] += node.n_samples
    _accumulate_importance(node.left, importances)
    _accumulate_importance(node.right, importances)
