"""Gaussian Naive Bayes and k-Nearest Neighbours from scratch.

Trains both on 3-class 2D mixture data, reports accuracy, then reproduces
the curse of dimensionality: pairwise distance ratios collapse as d grows.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# Gaussian Naive Bayes
# -----------------------------------------------------------------------------


class GaussianNB:
    """Fits per-class mean and variance; classifies by log P(y) + sum_j log P(x_j|y)."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        self.classes_ = np.unique(y)
        n, d = X.shape
        self.mean_ = np.empty((len(self.classes_), d))
        self.var_ = np.empty((len(self.classes_), d))
        self.log_prior_ = np.empty(len(self.classes_))
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.mean_[k] = Xc.mean(axis=0)
            self.var_[k] = Xc.var(axis=0) + self.var_smoothing
            self.log_prior_[k] = np.log(Xc.shape[0] / n)
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        # log N(x; mu, sigma^2) = -0.5 log(2 pi sigma^2) - (x - mu)^2 / (2 sigma^2)
        # broadcast: (n, K, d) - (K, d)
        diff = X[:, None, :] - self.mean_[None, :, :]
        log_lik = -0.5 * (np.log(2 * np.pi * self.var_) + diff**2 / self.var_)
        log_post = log_lik.sum(axis=-1) + self.log_prior_
        return log_post

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# -----------------------------------------------------------------------------
# k-Nearest Neighbours
# -----------------------------------------------------------------------------


class KNeighborsClassifier:
    """Brute-force Euclidean kNN. No training; everything happens at predict()."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsClassifier":
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # squared Euclidean distances (no sqrt needed for ranking)
        sq1 = np.sum(X**2, axis=1, keepdims=True)
        sq2 = np.sum(self.X_**2, axis=1, keepdims=True)
        dists = sq1 + sq2.T - 2 * X @ self.X_.T

        nearest = np.argpartition(dists, kth=min(self.k, self.y_.size - 1),
                                  axis=1)[:, :self.k]
        votes = self.y_[nearest]
        preds = np.array([np.bincount(row.astype(int)).argmax() for row in votes])
        return preds


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------


def make_three_mixture(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    per = n // 3
    X0 = rng.standard_normal((per, 2)) * 0.5 + np.array([-2.0, -1.0])
    X1 = rng.standard_normal((per, 2)) * 0.5 + np.array([2.0, -1.0])
    X2 = rng.standard_normal((per, 2)) * 0.5 + np.array([0.0, 2.0])
    X = np.vstack([X0, X1, X2])
    y = np.concatenate([
        np.zeros(per, dtype=int),
        np.ones(per, dtype=int),
        2 * np.ones(per, dtype=int),
    ])
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


# -----------------------------------------------------------------------------
# Curse-of-dimensionality demonstration
# -----------------------------------------------------------------------------


def distance_ratio_in_dim(d: int, n: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """Return (min dist, mean dist, max dist) over pairwise distances of n random
    points uniform in [0, 1]^d. As d grows, max/min approaches 1."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    sq = np.sum(X**2, axis=1, keepdims=True)
    dists = np.sqrt(np.clip(sq + sq.T - 2 * X @ X.T, 0, None))
    iu = np.triu_indices(n, k=1)
    pairs = dists[iu]
    return float(pairs.min()), float(pairs.mean()), float(pairs.max())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    np.set_printoptions(precision=3, suppress=True)

    print("=" * 60)
    print("Gaussian NB and kNN on a 3-class 2D mixture")
    print("=" * 60)
    X, y = make_three_mixture(n=300)
    n_tr = 240
    Xt, Xv, yt, yv = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]

    nb = GaussianNB().fit(Xt, yt)
    print(f"  GaussianNB    train={np.mean(nb.predict(Xt)==yt):.3f}  "
          f"test={np.mean(nb.predict(Xv)==yv):.3f}")
    print(f"    learned means : {nb.mean_}")

    for k in [1, 5, 15, 50]:
        knn = KNeighborsClassifier(k=k).fit(Xt, yt)
        print(f"  kNN  k={k:>3} train={np.mean(knn.predict(Xt)==yt):.3f}  "
              f"test={np.mean(knn.predict(Xv)==yv):.3f}")

    print()
    print("=" * 60)
    print("Curse of dimensionality: pairwise distance ratios")
    print("=" * 60)
    print(f"  {'d':>4}  {'min':>8}  {'mean':>8}  {'max':>8}  {'max/min':>9}  {'(max-min)/mean':>16}")
    for d in [1, 2, 5, 10, 50, 200, 1000]:
        dmin, dmean, dmax = distance_ratio_in_dim(d, n=200)
        print(f"  {d:>4}  {dmin:>8.3f}  {dmean:>8.3f}  {dmax:>8.3f}  "
              f"{dmax/dmin:>9.3f}  {(dmax-dmin)/dmean:>16.3f}")
    print("  ^ ratios collapse -- kNN can no longer distinguish neighbours from non-neighbours")

    print("\nOK")


if __name__ == "__main__":
    main()
