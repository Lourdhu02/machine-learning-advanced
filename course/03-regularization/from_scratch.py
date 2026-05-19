"""Ridge, Lasso, ElasticNet — all from scratch.

Demonstrates the key lesson: on data where most features are noise,
Lasso recovers the sparse support; Ridge does not.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


def make_sparse_data(n: int = 200, d: int = 20, k_true: int = 3, seed: int = 0):
    """y depends linearly on only the first k_true features; the rest are noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    true_w = np.zeros(d)
    true_w[:k_true] = rng.standard_normal(k_true) * 2.0
    noise = rng.standard_normal(n) * 0.3
    y = X @ true_w + noise
    return X, y, true_w


def standardize(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def soft_threshold(x: float, lam: float) -> float:
    """S_lam(x) = sign(x) * max(|x| - lam, 0). Snaps to 0 when |x| <= lam."""
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


class RidgeClosedForm:
    """w* = (XᵀX + λI)^-1 Xᵀy."""

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeClosedForm":
        d = X.shape[1]
        A = X.T @ X + self.lam * np.eye(d)
        self.w_ = np.linalg.solve(A, X.T @ y)
        return self


class LassoCoordinateDescent:
    """Cycle through coordinates, apply soft-thresholding.

    Assumes X is standardized so each column has ||x_j||^2 = n. The factor
    is absorbed into the update.
    """

    def __init__(self, lam: float = 0.5, n_iter: int = 500, tol: float = 1e-7):
        self.lam = lam
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoCoordinateDescent":
        n, d = X.shape
        w = np.zeros(d)
        col_norms_sq = np.sum(X**2, axis=0)  # ||x_j||^2

        for it in range(self.n_iter):
            w_old = w.copy()
            for j in range(d):
                # partial residual without column j
                r_j = y - X @ w + X[:, j] * w[j]
                rho_j = X[:, j] @ r_j
                w[j] = soft_threshold(rho_j, self.lam * n) / col_norms_sq[j]
            if np.max(np.abs(w - w_old)) < self.tol:
                self.n_iter_used_ = it + 1
                break
        else:
            self.n_iter_used_ = self.n_iter

        self.w_ = w
        return self


class ElasticNetCoordinateDescent:
    """L1 + L2 penalty. Same loop as Lasso, divides by (n + lam2)."""

    def __init__(self, lam1: float = 0.5, lam2: float = 0.5, n_iter: int = 500, tol: float = 1e-7):
        self.lam1 = lam1
        self.lam2 = lam2
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetCoordinateDescent":
        n, d = X.shape
        w = np.zeros(d)
        col_norms_sq = np.sum(X**2, axis=0)

        for it in range(self.n_iter):
            w_old = w.copy()
            for j in range(d):
                r_j = y - X @ w + X[:, j] * w[j]
                rho_j = X[:, j] @ r_j
                w[j] = soft_threshold(rho_j, self.lam1 * n) / (col_norms_sq[j] + self.lam2 * n)
            if np.max(np.abs(w - w_old)) < self.tol:
                self.n_iter_used_ = it + 1
                break
        else:
            self.n_iter_used_ = self.n_iter

        self.w_ = w
        return self


def main():
    X_raw, y, true_w = make_sparse_data(n=200, d=20, k_true=3, seed=0)
    X, _, _ = standardize(X_raw)
    y = y - y.mean()

    ridge = RidgeClosedForm(lam=5.0).fit(X, y)
    lasso = LassoCoordinateDescent(lam=0.4, n_iter=500).fit(X, y)
    enet = ElasticNetCoordinateDescent(lam1=0.4, lam2=0.5, n_iter=500).fit(X, y)

    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("true w (only first 3 nonzero):")
    print(f"  {true_w}")
    print()
    print(f"Ridge       weights (zeros: {int((np.abs(ridge.w_) < 1e-6).sum())} / {ridge.w_.size}):")
    print(f"  {ridge.w_}")
    print()
    print(f"Lasso       weights (zeros: {int((np.abs(lasso.w_) < 1e-6).sum())} / {lasso.w_.size}):  converged in {lasso.n_iter_used_} passes")
    print(f"  {lasso.w_}")
    print()
    print(f"ElasticNet  weights (zeros: {int((np.abs(enet.w_) < 1e-6).sum())} / {enet.w_.size}):  converged in {enet.n_iter_used_} passes")
    print(f"  {enet.w_}")
    print()

    n_zero_lasso = int((np.abs(lasso.w_) < 1e-6).sum())
    n_zero_ridge = int((np.abs(ridge.w_) < 1e-6).sum())
    assert n_zero_lasso > n_zero_ridge, "Lasso should produce more zeros than Ridge"
    print(f"OK  (Lasso zeros {n_zero_lasso} weights, Ridge zeros {n_zero_ridge})")


if __name__ == "__main__":
    main()
