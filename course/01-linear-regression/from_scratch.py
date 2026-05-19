"""Linear regression two ways: normal equation and gradient descent.
Confirms they agree on synthetic data.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


def make_data(n: int = 200, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    true_b = 0.7
    noise = rng.standard_normal(n) * 0.3
    y = X @ true_w + true_b + noise
    return X, y, true_w, true_b


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


class LinearRegressionClosedForm:
    """w* = (X^T X)^-1 X^T y, solved via np.linalg.solve for stability."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionClosedForm":
        Xb = add_intercept(X)
        self.w_ = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return add_intercept(X) @ self.w_


class LinearRegressionGD:
    """w_{t+1} = w_t - eta * (2/n) X^T (X w_t - y)."""

    def __init__(self, lr: float = 0.05, n_iter: int = 5000, tol: float = 1e-10):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        Xb = add_intercept(X)
        n, d = Xb.shape
        w = np.zeros(d)
        self.losses_ = []
        prev_loss = np.inf
        for _ in range(self.n_iter):
            residual = Xb @ w - y
            grad = (2 / n) * (Xb.T @ residual)
            w -= self.lr * grad
            loss = float(np.mean(residual**2))
            self.losses_.append(loss)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        self.w_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return add_intercept(X) @ self.w_


def main():
    X, y, true_w, true_b = make_data()

    cf = LinearRegressionClosedForm().fit(X, y)
    gd = LinearRegressionGD(lr=0.05, n_iter=5000).fit(X, y)

    print(f"true w           : {true_w}, bias = {true_b:.4f}")
    print(f"closed-form w    : {cf.w_[:-1]}, bias = {cf.w_[-1]:.4f}")
    print(f"gradient-descent : {gd.w_[:-1]}, bias = {gd.w_[-1]:.4f}")
    print(f"GD final loss    : {gd.losses_[-1]:.6f} after {len(gd.losses_)} steps")

    diff = float(np.max(np.abs(cf.w_ - gd.w_)))
    print(f"max |closed-form - GD| = {diff:.6f}")
    assert diff < 1e-3, "solvers disagree -- check learning rate or iterations"
    print("OK")


if __name__ == "__main__":
    main()
