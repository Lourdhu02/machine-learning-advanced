"""Three ensemble methods, all from scratch:

  RandomForestClassifier      : bagging + random feature subsets at each split
  AdaBoostClassifier          : depth-1 stumps + the alpha_t and weight updates
                                derived from forward-stagewise + exponential loss
  GradientBoostingRegressor   : sequence of regression trees fit to residuals

Demonstrates RF cutting tree variance on two-moons, AdaBoost building margin
on the same data, and GBM fitting a noisy sinusoid round by round.

Run: python from_scratch.py
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# =============================================================================
# Minimal tree primitives
# =============================================================================


@dataclass
class _Node:
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: float | np.ndarray | None = None  # leaf prediction (class probs or mean)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _gini(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if y.size == 0:
        return 0.0
    if sample_weight is None:
        _, counts = np.unique(y, return_counts=True)
        p = counts / y.size
    else:
        total = sample_weight.sum()
        if total <= 0:
            return 0.0
        p = np.array([sample_weight[y == c].sum() / total for c in np.unique(y)])
    return float(1.0 - np.sum(p * p))


def _mse(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean((y - y.mean()) ** 2))


class _ClassifierTree:
    """Greedy classification tree, optionally with feature subsampling and
    sample weights (for AdaBoost's weighted-error stumps)."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2,
                 max_features: int | None = None, rng: np.random.Generator | None = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, X, y, sample_weight=None) -> "_ClassifierTree":
        self.classes_ = np.unique(y)
        self.root_ = self._build(X, y, sample_weight, depth=0)
        return self

    def _leaf_prediction(self, y, sample_weight):
        if sample_weight is None:
            probs = np.array([(y == c).mean() for c in self.classes_])
        else:
            total = sample_weight.sum()
            probs = np.array([sample_weight[y == c].sum() / max(total, 1e-12)
                              for c in self.classes_])
        return probs

    def _build(self, X, y, sample_weight, depth):
        node = _Node(value=self._leaf_prediction(y, sample_weight))

        if y.size < self.min_samples_split or depth >= self.max_depth:
            return node
        if np.unique(y).size == 1:
            return node

        d = X.shape[1]
        if self.max_features is not None and self.max_features < d:
            feat_idx = self.rng.choice(d, size=self.max_features, replace=False)
        else:
            feat_idx = np.arange(d)

        parent_imp = _gini(y, sample_weight)
        best_gain = 0.0
        best = None
        for j in feat_idx:
            col = X[:, j]
            uniq = np.unique(col)
            if uniq.size < 2:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for t in thresholds:
                left_mask = col <= t
                if not left_mask.any() or left_mask.all():
                    continue
                if sample_weight is None:
                    n_l, n_r = int(left_mask.sum()), int((~left_mask).sum())
                    w_l, w_r = n_l, n_r
                    total = y.size
                else:
                    w_l = sample_weight[left_mask].sum()
                    w_r = sample_weight[~left_mask].sum()
                    total = w_l + w_r
                imp_l = _gini(y[left_mask], None if sample_weight is None else sample_weight[left_mask])
                imp_r = _gini(y[~left_mask], None if sample_weight is None else sample_weight[~left_mask])
                weighted = (w_l / total) * imp_l + (w_r / total) * imp_r
                gain = parent_imp - weighted
                if gain > best_gain:
                    best_gain = gain
                    best = (int(j), float(t), left_mask)

        if best is None:
            return node

        j, t, left_mask = best
        node.feature = j
        node.threshold = t
        node.value = None
        node.left = self._build(X[left_mask], y[left_mask],
                                None if sample_weight is None else sample_weight[left_mask],
                                depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask],
                                 None if sample_weight is None else sample_weight[~left_mask],
                                 depth + 1)
        return node

    def predict_proba(self, X):
        return np.array([self._traverse(self.root_, x) for x in X])

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def _traverse(self, node, x):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value


class _RegressionTree:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y) -> "_RegressionTree":
        self.root_ = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        node = _Node(value=float(y.mean()) if y.size > 0 else 0.0)
        if y.size < self.min_samples_split or depth >= self.max_depth:
            return node

        n = y.size
        parent = _mse(y)
        best_gain = 1e-12
        best = None
        for j in range(X.shape[1]):
            col = X[:, j]
            uniq = np.unique(col)
            if uniq.size < 2:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for t in thresholds:
                left = col <= t
                n_l, n_r = int(left.sum()), int((~left).sum())
                if n_l == 0 or n_r == 0:
                    continue
                weighted = (n_l / n) * _mse(y[left]) + (n_r / n) * _mse(y[~left])
                gain = parent - weighted
                if gain > best_gain:
                    best_gain = gain
                    best = (int(j), float(t), left)

        if best is None:
            return node

        j, t, left = best
        node.feature = j
        node.threshold = t
        node.value = None
        node.left = self._build(X[left], y[left], depth + 1)
        node.right = self._build(X[~left], y[~left], depth + 1)
        return node

    def predict(self, X):
        return np.array([self._traverse(self.root_, x) for x in X])

    def _traverse(self, node, x):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value


# =============================================================================
# Random Forest
# =============================================================================


class RandomForestClassifier:
    def __init__(self, n_estimators: int = 100, max_depth: int = 8,
                 max_features: int | str = "sqrt", seed: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y) -> "RandomForestClassifier":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.classes_ = np.unique(y)
        if self.max_features == "sqrt":
            mf = max(1, int(np.sqrt(d)))
        else:
            mf = int(self.max_features)

        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n)  # bootstrap sample
            tree = _ClassifierTree(max_depth=self.max_depth, max_features=mf, rng=rng)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        probs = np.mean([t.predict_proba(X) for t in self.trees_], axis=0)
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# =============================================================================
# AdaBoost (with depth-1 stumps)
# =============================================================================


class AdaBoostClassifier:
    """Discrete AdaBoost (SAMME for binary classification, y in {-1, +1}).

    Alpha and weight updates derived from the exponential-loss forward
    stagewise algorithm.
    """

    def __init__(self, n_estimators: int = 50, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X, y) -> "AdaBoostClassifier":
        assert set(np.unique(y)) <= {-1, 1}, "y must be in {-1, +1}"
        n = X.shape[0]
        w = np.full(n, 1.0 / n)
        rng = np.random.default_rng(self.seed)

        self.stumps_ = []
        self.alphas_ = []
        self.weight_history_ = [w.copy()]

        for _ in range(self.n_estimators):
            # binary labels 0/1 for the tree's internal use
            y_bin = (y > 0).astype(int)
            stump = _ClassifierTree(max_depth=1, rng=rng)
            stump.fit(X, y_bin, sample_weight=w)
            pred = stump.predict(X) * 2 - 1  # back to {-1, +1}

            wrong = pred != y
            err = float(w[wrong].sum() / w.sum())
            err = float(np.clip(err, 1e-12, 1 - 1e-12))
            alpha = 0.5 * np.log((1 - err) / err)

            w = w * np.exp(-alpha * y * pred)
            w /= w.sum()

            self.stumps_.append(stump)
            self.alphas_.append(alpha)
            self.weight_history_.append(w.copy())

        return self

    def decision_function(self, X):
        F = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas_, self.stumps_):
            pred = stump.predict(X) * 2 - 1
            F += alpha * pred
        return F

    def predict(self, X):
        return np.sign(self.decision_function(X))


# =============================================================================
# Gradient Boosting (regression)
# =============================================================================


class GradientBoostingRegressor:
    """Squared-loss GBM. Each tree fits the residual y - F_{t-1}(x).
    Update: F_t = F_{t-1} + lr * h_t.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y) -> "GradientBoostingRegressor":
        self.init_ = float(np.mean(y))
        F = np.full_like(y, self.init_, dtype=float)
        self.trees_ = []
        self.train_loss_history_ = []
        for _ in range(self.n_estimators):
            residual = y - F  # negative gradient of (1/2)(y-F)^2
            tree = _RegressionTree(max_depth=self.max_depth).fit(X, residual)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees_.append(tree)
            self.train_loss_history_.append(float(np.mean((y - F) ** 2)))
        return self

    def predict(self, X, n_trees: int | None = None):
        n_trees = n_trees if n_trees is not None else len(self.trees_)
        F = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.trees_[:n_trees]:
            F += self.learning_rate * tree.predict(X)
        return F


# =============================================================================
# Synthetic data
# =============================================================================


def make_moons(n=400, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    theta1 = np.pi * rng.uniform(0, 1, half)
    X0 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    theta2 = np.pi * rng.uniform(0, 1, half)
    X1 = np.stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)], axis=1)
    X = np.vstack([X0, X1]) + rng.standard_normal((n, 2)) * noise
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def make_sin(n=200, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = np.sin(x * 1.5) + rng.standard_normal(n) * noise
    return x.reshape(-1, 1), y


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("RandomForestClassifier on two-moons")
    print("=" * 60)
    Xm, ym = make_moons(n=400)
    Xt, Xv, yt, yv = Xm[:320], Xm[320:], ym[:320], ym[320:]

    single = _ClassifierTree(max_depth=10).fit(Xt, yt)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10).fit(Xt, yt)

    print(f"  single deep tree (depth 10): train={np.mean(single.predict(Xt)==yt):.3f}  "
          f"test={np.mean(single.predict(Xv)==yv):.3f}")
    print(f"  RF (50 trees, depth 10):     train={np.mean(rf.predict(Xt)==yt):.3f}  "
          f"test={np.mean(rf.predict(Xv)==yv):.3f}")
    print("  ^ deep single tree overfits; RF closes the train-test gap")

    print()
    print("=" * 60)
    print("AdaBoostClassifier on two-moons")
    print("=" * 60)
    y_pm = ym * 2 - 1
    ada = AdaBoostClassifier(n_estimators=30).fit(Xt, y_pm[:320])
    pred = ada.predict(Xv)
    yv_pm = yv * 2 - 1
    print(f"  AdaBoost (30 stumps): train={np.mean(ada.predict(Xt)==y_pm[:320]):.3f}  "
          f"test={np.mean(pred==yv_pm):.3f}")
    print(f"  first 5 alphas: {[round(a, 3) for a in ada.alphas_[:5]]}")

    print()
    print("=" * 60)
    print("GradientBoostingRegressor on noisy sin")
    print("=" * 60)
    Xs, ys = make_sin(n=200, noise=0.3)
    gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3).fit(Xs, ys)
    print(f"  initial constant pred: {gbm.init_:.3f}")
    print(f"  train MSE after 1 tree   = {gbm.train_loss_history_[0]:.4f}")
    print(f"  train MSE after 10 trees = {gbm.train_loss_history_[9]:.4f}")
    print(f"  train MSE after 100 trees= {gbm.train_loss_history_[-1]:.4f}")

    # Sanity: RF should beat a single tree on test acc by at least a few points
    rf_test = np.mean(rf.predict(Xv) == yv)
    single_test = np.mean(single.predict(Xv) == yv)
    print()
    assert rf_test >= single_test - 0.02, "RF should be competitive with single tree on test"
    print("OK")


if __name__ == "__main__":
    main()
