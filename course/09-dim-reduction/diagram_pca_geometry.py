"""PCA on a 2D correlated Gaussian: data scatter with the two principal axes
drawn over it, and the 1D projection onto PC1 shown alongside.

Output: diagram_pca_geometry.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import PCA, make_correlated_gaussian


def main():
    X = make_correlated_gaussian(n=400, seed=0)
    pca = PCA().fit(X)
    mean = pca.mean_
    pcs = pca.components_  # rows are PCs
    svals = pca.singular_values_

    proj_pc1 = pca.transform(X, k=1).ravel()

    fig = plt.figure(figsize=(13, 6))

    # Left: scatter with PC axes
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X[:, 0], X[:, 1], color="steelblue", edgecolors="white", s=24, alpha=0.85)
    for i, color in enumerate(["crimson", "darkgreen"]):
        v = pcs[i] * (svals[i] / np.sqrt(X.shape[0])) * 2.5
        ax1.annotate("", xy=mean + v, xytext=mean,
                     arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        ax1.annotate("", xy=mean - v, xytext=mean,
                     arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        ax1.text(*(mean + v + 0.08 * v / np.linalg.norm(v)),
                 f"PC{i+1}  (σ={svals[i]/np.sqrt(X.shape[0]):.2f})",
                 color=color, fontsize=10)

    ax1.set_aspect("equal")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("Principal axes = directions of maximum variance")
    ax1.grid(alpha=0.3)

    # Right: 1D projection histogram
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(proj_pc1, bins=30, color="crimson", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("projection onto PC1")
    ax2.set_ylabel("count")
    ax2.set_title("Most of the variance lives along PC1 alone")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"PCA: explained variance ratio = "
        f"{pca.explained_variance_ratio_[0]:.2f} / {pca.explained_variance_ratio_[1]:.2f}",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_pca_geometry.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_pca_geometry.png")


if __name__ == "__main__":
    main()
