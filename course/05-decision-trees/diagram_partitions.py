"""Recursive axis-aligned partitioning: how a decision tree carves the plane
into rectangles at increasing depth (1, 2, 4, and unlimited).

Output: diagram_partitions.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import DecisionTreeClassifier, make_moons


def plot_one(ax, model, X, y, title):
    pad = 0.4
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 250),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                colors=["#a8c5e0", "#e5a8a8"], alpha=0.55)

    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="steelblue", edgecolors="white",
               s=22, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="crimson", edgecolors="white",
               s=22, zorder=3)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    X, y = make_moons(n=300, noise=0.22, seed=1)

    depths = [1, 2, 4, None]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, depth in zip(axes, depths):
        clf = DecisionTreeClassifier(max_depth=depth).fit(X, y)
        acc = float(np.mean(clf.predict(X) == y))
        label = "unlimited" if depth is None else f"depth = {depth}"
        ax.set_title(f"{label}  —  train acc = {acc:.2f}", fontsize=11)
        plot_one(ax, clf, X, y, label)

    fig.suptitle("Decision-tree partitions: axis-aligned, layer by layer",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_partitions.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_partitions.png")


if __name__ == "__main__":
    main()
