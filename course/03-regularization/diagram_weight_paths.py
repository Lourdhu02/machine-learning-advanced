"""Ridge and Lasso coefficient paths as the regularization strength varies.

Ridge: weights shrink smoothly toward zero, never hit it exactly.
Lasso: weights snap to zero one by one as lambda grows.

Output: diagram_weight_paths.png
"""

import numpy as np
import matplotlib.pyplot as plt


def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def make_data(n=150, d=8, k_true=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    true_w = np.zeros(d)
    true_w[:k_true] = rng.standard_normal(k_true) * 2.0
    y = X @ true_w + 0.3 * rng.standard_normal(n)
    # standardize
    X = (X - X.mean(0)) / X.std(0)
    y = y - y.mean()
    return X, y


def ridge_path(X, y, lams):
    d = X.shape[1]
    coefs = np.empty((len(lams), d))
    for i, lam in enumerate(lams):
        coefs[i] = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    return coefs


def lasso_path(X, y, lams):
    n, d = X.shape
    col_norms_sq = np.sum(X**2, axis=0)
    coefs = np.empty((len(lams), d))
    w = np.zeros(d)  # warm-start along the path
    for i, lam in enumerate(lams):
        for _ in range(200):
            w_old = w.copy()
            for j in range(d):
                r_j = y - X @ w + X[:, j] * w[j]
                rho_j = X[:, j] @ r_j
                w[j] = soft_threshold(rho_j, lam * n) / col_norms_sq[j]
            if np.max(np.abs(w - w_old)) < 1e-8:
                break
        coefs[i] = w
    return coefs


def main():
    X, y = make_data()
    lams = np.logspace(-3, 1.2, 60)

    ridge_coefs = ridge_path(X, y, lams * y.size)  # scale lam comparably
    lasso_coefs = lasso_path(X, y, lams)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, coefs, title in [
        (axes[0], ridge_coefs, "Ridge: smooth shrinkage, no exact zeros"),
        (axes[1], lasso_coefs, "Lasso: coefficients snap to zero as λ grows"),
    ]:
        for j in range(coefs.shape[1]):
            ax.plot(lams, coefs[:, j], lw=1.8, label=f"w[{j}]" if j < 4 else None)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("λ  (regularization strength, log scale)")
        ax.set_ylabel("coefficient value")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=9, ncols=2)

    fig.tight_layout()
    fig.savefig("diagram_weight_paths.png", dpi=140)
    print("wrote diagram_weight_paths.png")


if __name__ == "__main__":
    main()
