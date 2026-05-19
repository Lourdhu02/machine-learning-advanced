"""Logistic regression two ways: gradient descent and Newton / IRLS.
Confirms they agree on synthetic 2D data and reports train/test accuracy.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (no overflow for very negative z)."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy with clipping to avoid log(0)."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def make_blobs(n: int = 400, seed: int = 0):
    """Two 2D Gaussian blobs with some overlap. Returns X (n, 2), y (n,)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.standard_normal((half, 2)) + np.array([-1.5, -1.0])
    X1 = rng.standard_normal((half, 2)) + np.array([1.5, 1.0])
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(half), np.ones(half)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


class LogisticRegressionGD:
    """Gradient descent on binary cross-entropy.

    Update: w <- w - eta * (1/n) Xᵀ (sigma(Xw) - y).
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 5000, tol: float = 1e-9):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        Xb = add_intercept(X)
        n, d = Xb.shape
        w = np.zeros(d)
        self.losses_ = []
        prev = np.inf
        for _ in range(self.n_iter):
            p = sigmoid(Xb @ w)
            grad = (1 / n) * Xb.T @ (p - y)
            w -= self.lr * grad
            loss = log_loss(y, p)
            self.losses_.append(loss)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        self.w_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(add_intercept(X) @ self.w_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class LogisticRegressionNewton:
    """Newton's method = IRLS.

    H = (1/n) Xᵀ S X,  S = diag(sigma (1 - sigma)).
    w <- w + (Xᵀ S X)^-1 Xᵀ (y - sigma).
    """

    def __init__(self, n_iter: int = 20, tol: float = 1e-10, ridge: float = 1e-8):
        self.n_iter = n_iter
        self.tol = tol
        self.ridge = ridge  # tiny L2 to keep Hessian invertible

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionNewton":
        Xb = add_intercept(X)
        n, d = Xb.shape
        w = np.zeros(d)
        self.losses_ = []
        prev = np.inf
        for _ in range(self.n_iter):
            p = sigmoid(Xb @ w)
            grad = Xb.T @ (p - y)
            S = p * (1 - p)
            H = (Xb.T * S) @ Xb + self.ridge * np.eye(d)
            w += np.linalg.solve(H, Xb.T @ (y - p))
            loss = log_loss(y, p)
            self.losses_.append(loss)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        self.w_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(add_intercept(X) @ self.w_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def main():
    X, y = make_blobs(n=400, seed=0)
    n_train = 320
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    gd = LogisticRegressionGD(lr=0.5, n_iter=5000).fit(X_train, y_train)
    nw = LogisticRegressionNewton(n_iter=20).fit(X_train, y_train)

    print(f"GD     weights: {gd.w_}  | final loss: {gd.losses_[-1]:.5f}  | steps: {len(gd.losses_)}")
    print(f"Newton weights: {nw.w_}  | final loss: {nw.losses_[-1]:.5f}  | steps: {len(nw.losses_)}")

    diff = float(np.max(np.abs(gd.w_ - nw.w_)))
    print(f"max |GD - Newton| = {diff:.5f}")

    for name, m in [("GD", gd), ("Newton", nw)]:
        acc_train = float(np.mean(m.predict(X_train) == y_train))
        acc_test = float(np.mean(m.predict(X_test) == y_test))
        print(f"{name:<6} accuracy  train={acc_train:.3f}  test={acc_test:.3f}")

    assert diff < 1e-2, "GD and Newton disagree -- increase n_iter or lower lr"
    print("OK")


if __name__ == "__main__":
    main()
