"""2D logistic regression: data colored by class, probability heatmap behind,
learned decision boundary line drawn on top.

Output: diagram_decision_boundary.png
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def make_blobs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.standard_normal((half, 2)) + np.array([-1.5, -1.0])
    X1 = rng.standard_normal((half, 2)) + np.array([1.5, 1.0])
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(half), np.ones(half)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def fit_newton(X, y, n_iter=20):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(n_iter):
        p = sigmoid(Xb @ w)
        S = p * (1 - p)
        H = (Xb.T * S) @ Xb + 1e-8 * np.eye(Xb.shape[1])
        w += np.linalg.solve(H, Xb.T @ (y - p))
    return w


def main():
    X, y = make_blobs(n=400, seed=0)
    w = fit_newton(X, y)

    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
    probs = sigmoid(grid @ w).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cs = ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.7)
    fig.colorbar(cs, ax=ax, label="P(y = 1 | x)")

    # decision boundary: P = 0.5 means w[0]*x + w[1]*y + w[2] = 0
    x_line = np.array([X[:, 0].min() - pad, X[:, 0].max() + pad])
    y_line = -(w[0] * x_line + w[2]) / w[1]
    ax.plot(x_line, y_line, "k--", lw=2, label="decision boundary (p = 0.5)")

    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="steelblue", edgecolors="white",
               label="class 0", s=40, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="crimson", edgecolors="white",
               label="class 1", s=40, zorder=3)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Logistic regression: linear boundary, smooth probability heatmap")
    ax.legend(loc="lower right")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    fig.tight_layout()
    fig.savefig("diagram_decision_boundary.png", dpi=140)
    print("wrote diagram_decision_boundary.png")


if __name__ == "__main__":
    main()
