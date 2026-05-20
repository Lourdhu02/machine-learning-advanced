"""Gaussian NB: decision regions on a 3-class 2D mixture with the fitted
per-class Gaussian ellipses overlaid.

Output: diagram_nb_regions.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import GaussianNB, make_three_mixture


def draw_ellipse(ax, mean, var, color):
    # Diagonal-covariance Gaussian: axes are sqrt(variance) * 2 sigma
    std = np.sqrt(var)
    theta = np.linspace(0, 2 * np.pi, 200)
    for n_sigma in [1.0, 2.0]:
        x = mean[0] + n_sigma * std[0] * np.cos(theta)
        y = mean[1] + n_sigma * std[1] * np.sin(theta)
        ax.plot(x, y, color=color, lw=1.2, alpha=0.8)


def main():
    X, y = make_three_mixture(n=300, seed=1)
    nb = GaussianNB().fit(X, y)

    pad = 0.6
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 250),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = nb.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = plt.get_cmap("Pastel1")
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=[cmap(0), cmap(1), cmap(2)], alpha=0.7)

    colors = ["steelblue", "crimson", "darkgreen"]
    for k, c in enumerate(nb.classes_):
        ax.scatter(X[y == c, 0], X[y == c, 1], c=colors[k],
                   edgecolors="white", s=35, label=f"class {c}", zorder=3)
        draw_ellipse(ax, nb.mean_[k], nb.var_[k], colors[k])

    ax.set_title("Gaussian Naive Bayes:  decision regions  +  fitted ±1σ / ±2σ ellipses")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="lower right")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("diagram_nb_regions.png", dpi=140)
    print("wrote diagram_nb_regions.png")


if __name__ == "__main__":
    main()
