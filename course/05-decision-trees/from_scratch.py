"""DecisionTreeClassifier from scratch with greedy axis-aligned splits,
Gini (default) or entropy impurity, and a print_tree() method that dumps
the learned rules in human-readable form.

Trains four trees of increasing depth on the "two moons" dataset and shows
the classic overfitting pattern.

Run: python from_scratch.py
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# -----------------------------------------------------------------------------
# Impurity measures
# -----------------------------------------------------------------------------


def gini(y: np.ndarray) -> float:
    """1 - sum_k p_k^2."""
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return float(1.0 - np.sum(p * p))


def entropy(y: np.ndarray) -> float:
    """- sum_k p_k log2 p_k."""
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return float(-np.sum(p * np.log2(p + 1e-12)))


# -----------------------------------------------------------------------------
# Tree node
# -----------------------------------------------------------------------------


@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    prediction: int | None = None  # leaf label
    n_samples: int = 0
    impurity: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# -----------------------------------------------------------------------------
# Decision tree
# -----------------------------------------------------------------------------


class DecisionTreeClassifier:
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0, criterion: str = "gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.impurity_fn = gini if criterion == "gini" else entropy
        self.criterion = criterion

    # ---- training ----

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.classes_ = np.unique(y)
        self.root_ = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n = y.size
        node_impurity = self.impurity_fn(y)
        majority = int(np.bincount(y).argmax())
        node = Node(prediction=majority, n_samples=n, impurity=node_impurity)

        if (n < self.min_samples_split
                or node_impurity == 0.0
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node

        best = self._best_split(X, y, node_impurity)
        if best is None:
            return node

        feature, threshold, left_idx, right_idx, gain = best
        if gain < self.min_impurity_decrease:
            return node

        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_idx], y[left_idx], depth + 1)
        node.right = self._build(X[right_idx], y[right_idx], depth + 1)
        node.prediction = None
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray, parent_impurity: float):
        n, d = X.shape
        best_gain = 0.0
        best = None
        for j in range(d):
            col = X[:, j]
            # candidate thresholds: midpoints between consecutive unique values
            uniq = np.unique(col)
            if uniq.size < 2:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for t in thresholds:
                left_mask = col <= t
                n_left = int(left_mask.sum())
                n_right = n - n_left
                if n_left == 0 or n_right == 0:
                    continue
                impurity_left = self.impurity_fn(y[left_mask])
                impurity_right = self.impurity_fn(y[~left_mask])
                weighted = (n_left / n) * impurity_left + (n_right / n) * impurity_right
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_gain = gain
                    best = (j, float(t), np.where(left_mask)[0],
                            np.where(~left_mask)[0], gain)
        return best

    # ---- inference ----

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(self.root_, x) for x in X])

    def _predict_one(self, node: Node, x: np.ndarray) -> int:
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    # ---- pretty-printing ----

    def print_tree(self, feature_names: list[str] | None = None,
                   indent: str = "  "):
        def _walk(node: Node, depth: int):
            prefix = indent * depth
            if node.is_leaf:
                print(f"{prefix}-> predict class {node.prediction}  "
                      f"(n={node.n_samples}, impurity={node.impurity:.3f})")
                return
            fname = (feature_names[node.feature] if feature_names
                     else f"x[{node.feature}]")
            print(f"{prefix}if {fname} <= {node.threshold:.3f}    "
                  f"(n={node.n_samples}, impurity={node.impurity:.3f})")
            _walk(node.left, depth + 1)
            print(f"{prefix}else:")
            _walk(node.right, depth + 1)

        _walk(self.root_, 0)


# -----------------------------------------------------------------------------
# Synthetic data: two interleaving moons
# -----------------------------------------------------------------------------


def make_moons(n: int = 400, noise: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    half = n // 2
    theta1 = np.pi * rng.uniform(0, 1, half)
    X0 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    theta2 = np.pi * rng.uniform(0, 1, half)
    X1 = np.stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)], axis=1)
    X = np.vstack([X0, X1])
    X += rng.standard_normal(X.shape) * noise
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    X, y = make_moons(n=400, noise=0.25, seed=0)
    n_tr = 320
    Xt, Xv, yt, yv = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]

    print(f"{'depth':>10} {'train acc':>12} {'test acc':>12} {'gap':>10}")
    for depth in [1, 2, 4, None]:
        clf = DecisionTreeClassifier(max_depth=depth).fit(Xt, yt)
        a_tr = float(np.mean(clf.predict(Xt) == yt))
        a_te = float(np.mean(clf.predict(Xv) == yv))
        label = "None (unlimited)" if depth is None else str(depth)
        print(f"{label:>10} {a_tr:>12.3f} {a_te:>12.3f} {a_tr - a_te:>+10.3f}")

    print()
    print("Learned tree at max_depth=3:")
    print("-" * 60)
    clf3 = DecisionTreeClassifier(max_depth=3).fit(Xt, yt)
    clf3.print_tree(feature_names=["x1", "x2"])

    # Sanity: deeper tree should reach >90% train accuracy on this dataset
    clf_deep = DecisionTreeClassifier(max_depth=10).fit(Xt, yt)
    assert np.mean(clf_deep.predict(Xt) == yt) > 0.9
    print("\nOK")


if __name__ == "__main__":
    main()
