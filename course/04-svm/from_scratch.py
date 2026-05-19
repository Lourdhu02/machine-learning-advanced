"""Two SVMs from scratch:

  LinearSVM  : soft-margin primal, subgradient descent on hinge + L2.
  KernelSVM  : soft-margin dual, simplified SMO (Platt 1998, simplified).

Linear SVM trained on a linearly separable blob. Kernel SVM trained on a
non-separable concentric-rings dataset with an RBF kernel.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# Linear SVM via subgradient descent on the hinge + L2 primal (Pegasos-style)
# -----------------------------------------------------------------------------


class LinearSVM:
    """min  (lam/2)||w||^2 + (1/n) sum_i max(0, 1 - y_i (w.x_i + b))."""

    def __init__(self, lam: float = 0.01, lr: float = 0.01, n_iter: int = 3000):
        self.lam = lam
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        n, d = X.shape
        # labels must be +-1
        assert set(np.unique(y)) <= {-1, 1}, "y must be in {-1, +1}"
        w = np.zeros(d)
        b = 0.0
        self.losses_ = []
        for _ in range(self.n_iter):
            margins = y * (X @ w + b)
            active = margins < 1
            grad_w = self.lam * w - (X[active].T @ y[active]) / n
            grad_b = -np.sum(y[active]) / n
            w -= self.lr * grad_w
            b -= self.lr * grad_b
            hinge = np.maximum(0, 1 - margins).mean()
            self.losses_.append(0.5 * self.lam * (w @ w) + hinge)
        self.w_ = w
        self.b_ = float(b)
        # support vectors = points within or violating the margin (margin <= 1)
        self.support_idx_ = np.where(y * (X @ w + b) <= 1.0 + 1e-3)[0]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X))


# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------


def linear_kernel(X1, X2):
    return X1 @ X2.T


def rbf_kernel(X1, X2, gamma: float = 0.5):
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    sq1 = np.sum(X1**2, axis=1, keepdims=True)
    sq2 = np.sum(X2**2, axis=1, keepdims=True)
    sq = sq1 + sq2.T - 2 * X1 @ X2.T
    return np.exp(-gamma * np.clip(sq, 0, None))


def poly_kernel(X1, X2, degree: int = 3, coef0: float = 1.0):
    return (X1 @ X2.T + coef0) ** degree


# -----------------------------------------------------------------------------
# Kernel SVM via Simplified SMO (Platt's simplified version; see CS229 notes)
# -----------------------------------------------------------------------------


class KernelSVM:
    """Soft-margin SVM dual:

        max  sum_i a_i - 0.5 sum_i sum_j a_i a_j y_i y_j K(x_i, x_j)
        s.t. sum_i a_i y_i = 0,  0 <= a_i <= C
    """

    def __init__(self, kernel="rbf", C: float = 1.0, gamma: float = 0.5,
                 degree: int = 3, tol: float = 1e-3, max_passes: int = 20,
                 seed: int = 0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_passes = max_passes
        self.seed = seed

    def _K(self, X1, X2):
        if self.kernel == "linear":
            return linear_kernel(X1, X2)
        if self.kernel == "rbf":
            return rbf_kernel(X1, X2, self.gamma)
        if self.kernel == "poly":
            return poly_kernel(X1, X2, self.degree)
        raise ValueError(self.kernel)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        assert set(np.unique(y)) <= {-1, 1}
        alpha = np.zeros(n)
        b = 0.0
        K = self._K(X, X)

        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                f_i = float((alpha * y) @ K[:, i] + b)
                E_i = f_i - y[i]
                # KKT violation?
                if ((y[i] * E_i < -self.tol and alpha[i] < self.C) or
                        (y[i] * E_i > self.tol and alpha[i] > 0)):
                    # pick j != i at random
                    j = i
                    while j == i:
                        j = int(rng.integers(0, n))
                    f_j = float((alpha * y) @ K[:, j] + b)
                    E_j = f_j - y[j]

                    a_i_old, a_j_old = alpha[i], alpha[j]

                    if y[i] != y[j]:
                        L = max(0.0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0.0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] -= y[j] * (E_i - E_j) / eta
                    alpha[j] = float(np.clip(alpha[j], L, H))
                    if abs(alpha[j] - a_j_old) < 1e-5:
                        continue
                    alpha[i] += y[i] * y[j] * (a_j_old - alpha[j])

                    b1 = b - E_i - y[i] * (alpha[i] - a_i_old) * K[i, i] \
                            - y[j] * (alpha[j] - a_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alpha[i] - a_i_old) * K[i, j] \
                            - y[j] * (alpha[j] - a_j_old) * K[j, j]
                    if 0 < alpha[i] < self.C:
                        b = float(b1)
                    elif 0 < alpha[j] < self.C:
                        b = float(b2)
                    else:
                        b = float(0.5 * (b1 + b2))

                    num_changed += 1
            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        sv_mask = alpha > 1e-6
        self.X_ = X[sv_mask]
        self.y_ = y[sv_mask]
        self.alpha_ = alpha[sv_mask]
        self.b_ = float(b)
        self.n_sv_ = int(sv_mask.sum())
        return self

    def decision_function(self, X):
        K = self._K(X, self.X_)
        return K @ (self.alpha_ * self.y_) + self.b_

    def predict(self, X):
        return np.sign(self.decision_function(X))


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------


def make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.standard_normal((half, 2)) * 0.6 + np.array([-2.0, -1.5])
    X1 = rng.standard_normal((half, 2)) * 0.6 + np.array([2.0, 1.5])
    X = np.vstack([X0, X1])
    y = np.concatenate([-np.ones(half), np.ones(half)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def make_rings(n=200, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.concatenate([
        rng.normal(1.0, 0.15, half),
        rng.normal(2.5, 0.15, half),
    ])
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    y = np.concatenate([-np.ones(half), np.ones(half)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    np.set_printoptions(precision=3, suppress=True)

    print("=" * 60)
    print("LinearSVM on linearly separable blobs")
    print("=" * 60)
    X, y = make_blobs(n=200)
    n_tr = 160
    Xt, Xv, yt, yv = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
    lsvm = LinearSVM(lam=0.05, lr=0.05, n_iter=3000).fit(Xt, yt)
    print(f"  w = {lsvm.w_},  b = {lsvm.b_:.3f}")
    print(f"  final hinge+L2 loss = {lsvm.losses_[-1]:.4f}")
    print(f"  train acc = {np.mean(lsvm.predict(Xt) == yt):.3f}")
    print(f"  test  acc = {np.mean(lsvm.predict(Xv) == yv):.3f}")
    print(f"  #support vectors (margin <= 1) = {len(lsvm.support_idx_)}")

    print()
    print("=" * 60)
    print("KernelSVM (RBF) on concentric rings")
    print("=" * 60)
    X, y = make_rings(n=300)
    n_tr = 240
    Xt, Xv, yt, yv = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
    ksvm = KernelSVM(kernel="rbf", C=1.0, gamma=1.0).fit(Xt, yt)
    print(f"  #support vectors = {ksvm.n_sv_} / {n_tr}")
    print(f"  train acc = {np.mean(ksvm.predict(Xt) == yt):.3f}")
    print(f"  test  acc = {np.mean(ksvm.predict(Xv) == yv):.3f}")

    # Sanity: linear SVM on rings should be near random
    lsvm_rings = LinearSVM(lam=0.01, lr=0.01, n_iter=3000).fit(Xt, yt)
    print(f"  (linear SVM on the same data: test acc = {np.mean(lsvm_rings.predict(Xv) == yv):.3f}"
          "  -- linear can't carve a circle)")

    assert np.mean(ksvm.predict(Xv) == yv) > 0.9, "RBF kernel should easily separate rings"
    print("\nOK")


if __name__ == "__main__":
    main()
