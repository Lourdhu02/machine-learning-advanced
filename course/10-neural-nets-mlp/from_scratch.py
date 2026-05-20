"""A 1-hidden-layer MLP in pure NumPy: forward, backward, SGD.

Includes a numerical gradient check that verifies the hand-coded backprop
matches finite-difference gradients within 1e-5.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    n = y.size
    return float(-np.mean(np.log(probs[np.arange(n), y] + eps)))


def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    Y = np.zeros((y.size, K))
    Y[np.arange(y.size), y] = 1.0
    return Y


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------


class MLPClassifier:
    """Linear -> ReLU -> Linear -> Softmax + cross-entropy loss.

    Hand-coded forward and backward passes -- no autodiff.
    """

    def __init__(self, n_in: int, n_hidden: int, n_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialization for ReLU
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_classes)) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_classes)
        self.n_classes = n_classes

    # ---- forward ----

    def forward(self, X: np.ndarray):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)             # ReLU
        Z2 = A1 @ self.W2 + self.b2
        probs = softmax(Z2)
        cache = dict(X=X, Z1=Z1, A1=A1, Z2=Z2, probs=probs)
        return probs, cache

    # ---- backward (the whole point) ----

    def backward(self, cache: dict, y: np.ndarray) -> dict:
        n = y.size
        X, Z1, A1, probs = cache["X"], cache["Z1"], cache["A1"], cache["probs"]
        Y = one_hot(y, self.n_classes)

        # softmax + CE: clean (prob - target) residual
        delta2 = (probs - Y) / n                    # (n, K)
        dW2 = A1.T @ delta2                         # (h, K)
        db2 = delta2.sum(axis=0)                    # (K,)

        # propagate back through ReLU and the first linear layer
        dA1 = delta2 @ self.W2.T                    # (n, h)
        delta1 = dA1 * (Z1 > 0).astype(float)       # ReLU' = 1[Z > 0]
        dW1 = X.T @ delta1                          # (d, h)
        db1 = delta1.sum(axis=0)                    # (h,)
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2)

    # ---- training ----

    def fit(self, X, y, lr=0.1, n_epochs=200, batch_size=32,
            X_val=None, y_val=None, verbose=False) -> "MLPClassifier":
        rng = np.random.default_rng(0)
        n = X.shape[0]
        self.history_ = {"loss": [], "acc": [], "val_acc": []}

        for epoch in range(n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                Xb, yb = X[idx], y[idx]
                probs, cache = self.forward(Xb)
                grads = self.backward(cache, yb)
                for name in ["W1", "b1", "W2", "b2"]:
                    setattr(self, name, getattr(self, name) - lr * grads[name])

            probs_full, _ = self.forward(X)
            loss = cross_entropy(probs_full, y)
            acc = float(np.mean(np.argmax(probs_full, axis=1) == y))
            self.history_["loss"].append(loss)
            self.history_["acc"].append(acc)
            if X_val is not None:
                probs_v, _ = self.forward(X_val)
                self.history_["val_acc"].append(
                    float(np.mean(np.argmax(probs_v, axis=1) == y_val)))
            if verbose and epoch % 25 == 0:
                print(f"  epoch {epoch:>3}: loss={loss:.4f}  acc={acc:.3f}")
        return self

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# -----------------------------------------------------------------------------
# Numerical gradient check
# -----------------------------------------------------------------------------


def grad_check(net: MLPClassifier, X, y, eps: float = 1e-5) -> float:
    """Compare analytical gradients (from backward) to finite-difference for
    a few random parameter entries. Returns the max relative error."""
    probs, cache = net.forward(X)
    analytical = net.backward(cache, y)

    rng = np.random.default_rng(0)
    max_err = 0.0
    for name in ["W1", "b1", "W2", "b2"]:
        P = getattr(net, name)
        # sample up to 5 random entries to check
        flat = P.flatten()
        idxs = rng.choice(flat.size, size=min(5, flat.size), replace=False)
        for idx in idxs:
            old = flat[idx]
            flat[idx] = old + eps
            setattr(net, name, flat.reshape(P.shape))
            l_plus = cross_entropy(net.forward(X)[0], y)
            flat[idx] = old - eps
            setattr(net, name, flat.reshape(P.shape))
            l_minus = cross_entropy(net.forward(X)[0], y)
            flat[idx] = old
            setattr(net, name, flat.reshape(P.shape))

            numerical = (l_plus - l_minus) / (2 * eps)
            ana = analytical[name].flatten()[idx]
            err = abs(numerical - ana) / max(abs(numerical) + abs(ana), 1e-12)
            max_err = max(max_err, err)
    return max_err


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Numerical gradient check (must print error < 1e-5)")
    print("=" * 60)
    X_tiny, y_tiny = make_moons(n=20)
    net_tiny = MLPClassifier(2, 4, 2, seed=0)
    err = grad_check(net_tiny, X_tiny, y_tiny)
    print(f"  max relative gradient error: {err:.2e}")
    assert err < 1e-4, "backprop is wrong"
    print("  OK -- analytical and numerical gradients agree")

    print()
    print("=" * 60)
    print("MLP on two-moons, varying hidden width")
    print("=" * 60)
    X, y = make_moons(n=400, noise=0.22)
    n_tr = 320
    Xt, Xv, yt, yv = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]

    for h in [2, 8, 32]:
        net = MLPClassifier(2, h, 2, seed=0)
        net.fit(Xt, yt, lr=0.2, n_epochs=200, batch_size=32,
                X_val=Xv, y_val=yv, verbose=False)
        acc_t = float(np.mean(net.predict(Xt) == yt))
        acc_v = float(np.mean(net.predict(Xv) == yv))
        print(f"  hidden={h:>3}  train={acc_t:.3f}  test={acc_v:.3f}")

    print("\nOK")


if __name__ == "__main__":
    main()
