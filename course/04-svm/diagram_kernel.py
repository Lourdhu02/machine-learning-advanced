"""Side-by-side: linear SVM fails on concentric rings, RBF kernel SVM works.

Output: diagram_kernel.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import LinearSVM, KernelSVM, make_rings


def plot_boundary(ax, model, X, y, title):
    pad = 0.6
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 250),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.6, vmin=-2.5, vmax=2.5)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="black",
               linestyles=["--", "-", "--"], linewidths=[1, 2, 1])

    ax.scatter(X[y == -1, 0], X[y == -1, 1], c="steelblue", edgecolors="white", s=35)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="crimson", edgecolors="white", s=35)

    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def main():
    X, y = make_rings(n=300, seed=0)
    lsvm = LinearSVM(lam=0.01, lr=0.01, n_iter=3000).fit(X, y)
    ksvm = KernelSVM(kernel="rbf", C=1.0, gamma=1.0).fit(X, y)

    acc_l = float(np.mean(lsvm.predict(X) == y))
    acc_k = float(np.mean(ksvm.predict(X) == y))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_boundary(axes[0], lsvm, X, y,
                  f"Linear SVM  —  can't curve a boundary\nacc = {acc_l:.2f}")
    plot_boundary(axes[1], ksvm, X, y,
                  f"Kernel SVM (RBF, γ=1.0)  —  {ksvm.n_sv_} SVs\nacc = {acc_k:.2f}")

    fig.tight_layout()
    fig.savefig("diagram_kernel.png", dpi=140)
    print("wrote diagram_kernel.png")


if __name__ == "__main__":
    main()
