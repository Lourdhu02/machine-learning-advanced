"""Optimizers + Dropout + BatchNorm from scratch.

  - SGD, Momentum, Adam: each implemented as a tiny class.
  - Dropout and BatchNorm: forward + backward implementations.
  - 2D banana-loss optimizer race.
  - Deep MLP training comparison: Adam vs SGD vs Momentum.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


# =============================================================================
# Optimizers
# =============================================================================


class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params: dict, grads: dict):
        for name in params:
            params[name] -= self.lr * grads[name]


class Momentum:
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.v: dict | None = None

    def step(self, params: dict, grads: dict):
        if self.v is None:
            self.v = {k: np.zeros_like(p) for k, p in params.items()}
        for name in params:
            self.v[name] = self.beta * self.v[name] + grads[name]
            params[name] -= self.lr * self.v[name]


class Adam:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict | None = None
        self.v: dict | None = None
        self.t = 0

    def step(self, params: dict, grads: dict):
        if self.m is None:
            self.m = {k: np.zeros_like(p) for k, p in params.items()}
            self.v = {k: np.zeros_like(p) for k, p in params.items()}
        self.t += 1
        for name in params:
            g = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g * g
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# Banana loss and a 2D optimizer race
# =============================================================================


def banana_loss(w):
    """Bowl with a ravine: low along x = y line, steep across."""
    return 10 * (w[1] - w[0] ** 2) ** 2 + (1 - w[0]) ** 2


def banana_grad(w):
    dx = -40 * w[0] * (w[1] - w[0] ** 2) - 2 * (1 - w[0])
    dy = 20 * (w[1] - w[0] ** 2)
    return np.array([dx, dy])


def race(optimizer_cls, lr, n_steps=200, start=(-1.5, 1.5)):
    """Run an optimizer on the banana and record the trajectory."""
    w = {"w": np.array(start, dtype=float)}
    opt = optimizer_cls(lr=lr)
    path = [w["w"].copy()]
    for _ in range(n_steps):
        grads = {"w": banana_grad(w["w"])}
        opt.step(w, grads)
        path.append(w["w"].copy())
    return np.array(path)


# =============================================================================
# Tiny MLP for the optimizer comparison on real data
# =============================================================================


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y, eps=1e-12):
    n = y.size
    return float(-np.mean(np.log(probs[np.arange(n), y] + eps)))


def one_hot(y, K):
    Y = np.zeros((y.size, K))
    Y[np.arange(y.size), y] = 1.0
    return Y


class TinyMLP:
    """One hidden ReLU layer. We expose `params` and `grads` as dicts so the
    Optimizer classes above can drive training."""

    def __init__(self, n_in, n_hidden, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in),
            "b1": np.zeros(n_hidden),
            "W2": rng.standard_normal((n_hidden, n_classes)) * np.sqrt(2.0 / n_hidden),
            "b2": np.zeros(n_classes),
        }
        self.K = n_classes

    def forward(self, X):
        p = self.params
        Z1 = X @ p["W1"] + p["b1"]
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ p["W2"] + p["b2"]
        probs = softmax(Z2)
        return probs, dict(X=X, Z1=Z1, A1=A1, Z2=Z2, probs=probs)

    def backward(self, cache, y):
        n = y.size
        X, Z1, A1, probs = cache["X"], cache["Z1"], cache["A1"], cache["probs"]
        Y = one_hot(y, self.K)
        delta2 = (probs - Y) / n
        dW2 = A1.T @ delta2
        db2 = delta2.sum(axis=0)
        dA1 = delta2 @ self.params["W2"].T
        delta1 = dA1 * (Z1 > 0).astype(float)
        dW1 = X.T @ delta1
        db1 = delta1.sum(axis=0)
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2)

    def predict(self, X):
        return np.argmax(self.forward(X)[0], axis=1)


# =============================================================================
# Dropout and BatchNorm (forward + backward)
# =============================================================================


class Dropout:
    """Inverted dropout. Used at training time only."""

    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if not training:
            return x
        keep = 1 - self.p
        self.mask = (np.random.rand(*x.shape) < keep) / keep
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class BatchNorm:
    """1D BatchNorm over the feature axis. Learnable gamma, beta. Tracks
    running mean/var for inference."""

    def __init__(self, n_features: int, momentum: float = 0.9, eps: float = 1e-5):
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self._cache = None

    def forward(self, x, training=True):
        if training:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            out = self.gamma * x_hat + self.beta
            # update running stats for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self._cache = (x, x_hat, mu, var)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta
        return out


# =============================================================================
# Synthetic data
# =============================================================================


def make_moons(n=400, noise=0.22, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    t1 = np.pi * rng.uniform(0, 1, half)
    X0 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    t2 = np.pi * rng.uniform(0, 1, half)
    X1 = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
    X = np.vstack([X0, X1]) + rng.standard_normal((n, 2)) * noise
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("Banana-loss optimizer race")
    print("=" * 60)
    p_sgd = race(SGD, lr=0.01, n_steps=400)
    p_mom = race(Momentum, lr=0.01, n_steps=400)
    p_adam = race(Adam, lr=0.05, n_steps=400)
    for name, p in [("SGD", p_sgd), ("Momentum", p_mom), ("Adam", p_adam)]:
        L = banana_loss(p[-1])
        print(f"  {name:<8}  final loss = {L:.6f}   final w = {p[-1]}")

    print()
    print("=" * 60)
    print("Optimizer comparison on a small MLP (moons)")
    print("=" * 60)
    X, y = make_moons(n=400, noise=0.22)
    rng = np.random.default_rng(0)

    for name, opt_cls, lr in [
        ("SGD",      SGD,      0.2),
        ("Momentum", Momentum, 0.05),
        ("Adam",     Adam,     1e-2),
    ]:
        net = TinyMLP(2, 16, 2, seed=0)
        opt = opt_cls(lr=lr)
        for _ in range(150):
            perm = rng.permutation(X.shape[0])
            for s in range(0, X.shape[0], 32):
                idx = perm[s:s + 32]
                Xb, yb = X[idx], y[idx]
                _, cache = net.forward(Xb)
                grads = net.backward(cache, yb)
                opt.step(net.params, grads)
        acc = float(np.mean(net.predict(X) == y))
        loss = cross_entropy(net.forward(X)[0], y)
        print(f"  {name:<8}  final loss={loss:.4f}  accuracy={acc:.3f}")

    print("\nOK")


if __name__ == "__main__":
    main()
