"""4 well-separated 50-D Gaussian clusters: PCA's top-2 projection vs t-SNE.
Both should recover the clusters; the layouts will differ markedly.

Output: diagram_pca_vs_tsne.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import PCA, tsne, make_blobs_hd


def main():
    X, y = make_blobs_hd(n_per_cluster=40, d=50, n_clusters=4, sep=4.0, seed=0)

    pca = PCA().fit(X)
    Z_pca = pca.transform(X, k=2)

    print("running t-SNE (1-2 minutes on first run)...")
    Y_tsne = tsne(X, n_iter=400, perplexity=20.0, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = ["steelblue", "crimson", "darkgreen", "orange"]

    for ax, Z, title in [
        (axes[0], Z_pca, "PCA (top-2 components, linear)"),
        (axes[1], Y_tsne, "t-SNE (non-linear, perplexity = 20)"),
    ]:
        for k in range(4):
            ax.scatter(Z[y == k, 0], Z[y == k, 1], c=colors[k], edgecolors="white",
                       s=40, label=f"cluster {k}")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("4 Gaussian clusters in 50-D, projected two ways",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_pca_vs_tsne.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_pca_vs_tsne.png")


if __name__ == "__main__":
    main()
