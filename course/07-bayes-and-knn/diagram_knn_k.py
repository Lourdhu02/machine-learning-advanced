"""kNN decision boundary on the same 3-class data, for k in {1, 5, 15, 50}.
Watch the boundary smooth as k grows.

Output: diagram_knn_k.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import KNeighborsClassifier, make_three_mixture


def plot_one(ax, k, X, y):
    knn = KNeighborsClassifier(k=k).fit(X, y)
    pad = 0.6
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 220),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 220),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid).reshape(xx.shape)

    cmap = plt.get_cmap("Pastel1")
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=[cmap(0), cmap(1), cmap(2)], alpha=0.7)

    colors = ["steelblue", "crimson", "darkgreen"]
    for c in np.unique(y):
        ax.scatter(X[y == c, 0], X[y == c, 1], c=colors[int(c)],
                   edgecolors="white", s=25, zorder=3)

    ax.set_title(f"k = {k}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())


def main():
    X, y = make_three_mixture(n=180, seed=1)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, k in zip(axes, [1, 5, 15, 50]):
        plot_one(ax, k, X, y)
    fig.suptitle("kNN boundary smooths as k grows: bias-variance, made visual",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_knn_k.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_knn_k.png")


if __name__ == "__main__":
    main()
