"""MLP decision boundary on two-moons at hidden widths 2, 8, 32.

Output: diagram_decision_boundary.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import MLPClassifier, make_moons


def plot_one(ax, net, X, y, title):
    pad = 0.4
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 220),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 220),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = net.predict_proba(grid)[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.7)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.4)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="steelblue", edgecolors="white", s=22, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="crimson", edgecolors="white", s=22, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())


def main():
    X, y = make_moons(n=400, noise=0.22, seed=0)
    widths = [2, 8, 32]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, h in zip(axes, widths):
        net = MLPClassifier(2, h, 2, seed=0).fit(X, y, lr=0.2, n_epochs=300, batch_size=32)
        acc = float(np.mean(net.predict(X) == y))
        plot_one(ax, net, X, y, f"hidden = {h}   acc = {acc:.2f}")

    fig.suptitle("MLP decision boundary: more hidden units → more flexible boundary",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_decision_boundary.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_decision_boundary.png")


if __name__ == "__main__":
    main()
