"""Scatter of (x, y) with the fitted line and residuals drawn as vertical drops.

Output: diagram_fit.png
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 25)
    y = 1.3 * x + 2.0 + rng.standard_normal(x.size) * 1.5

    X = np.vstack([x, np.ones_like(x)]).T
    w, b = np.linalg.solve(X.T @ X, X.T @ y)
    y_hat = w * x + b

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="steelblue", label="data", zorder=3)
    ax.plot(x, y_hat, color="crimson", lw=2, label=f"fit: y = {w:.2f} x + {b:.2f}")
    for xi, yi, yh in zip(x, y, y_hat):
        ax.plot([xi, xi], [yi, yh], color="gray", lw=0.8, alpha=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("OLS minimizes the sum of squared vertical residuals.")
    ax.legend()
    fig.tight_layout()
    fig.savefig("diagram_fit.png", dpi=140)
    print("wrote diagram_fit.png")


if __name__ == "__main__":
    main()
