"""Side-by-side: same elongated 3-cluster dataset, k-means assumes spherical
(slices through ellipses), GMM fits full covariance ellipses correctly.

Output: diagram_gmm_vs_kmeans.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import KMeans, GaussianMixture, make_elongated


def draw_ellipse(ax, mean, cov, color, lw=1.5):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    theta = np.linspace(0, 2 * np.pi, 200)
    for n_sigma in [1, 2]:
        a = n_sigma * np.sqrt(eigvals[0])
        b = n_sigma * np.sqrt(eigvals[1])
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        rot = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                        [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]])
        xy = rot @ np.vstack([x, y])
        ax.plot(mean[0] + xy[0], mean[1] + xy[1], color=color, lw=lw, alpha=0.85)


def main():
    X, _ = make_elongated(n=300, seed=3)
    km = KMeans(n_clusters=3, seed=0).fit(X)
    gmm = GaussianMixture(n_components=3, seed=0).fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = ["steelblue", "crimson", "darkgreen"]

    # k-means panel
    ax = axes[0]
    for k in range(3):
        mask = km.labels_ == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], edgecolors="white", s=30, zorder=2)
        ax.scatter(*km.centroids_[k], marker="*", c=colors[k], edgecolors="black",
                   s=320, lw=1.5, zorder=4)
        # k-means implies isotropic, draw a circle of radius = mean distance
        if mask.any():
            r = np.linalg.norm(X[mask] - km.centroids_[k], axis=1).mean()
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(km.centroids_[k, 0] + r * np.cos(theta),
                    km.centroids_[k, 1] + r * np.sin(theta),
                    color=colors[k], lw=1.5, alpha=0.6)
    ax.set_title("k-means: assumes spherical, gets ellipses wrong")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    # GMM panel
    ax = axes[1]
    labels = gmm.predict(X)
    for k in range(3):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], edgecolors="white", s=30, zorder=2)
        ax.scatter(*gmm.means_[k], marker="*", c=colors[k], edgecolors="black",
                   s=320, lw=1.5, zorder=4)
        draw_ellipse(ax, gmm.means_[k], gmm.covariances_[k], colors[k])
    ax.set_title("GMM: full covariance, ellipses snap to orientation + scale")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig("diagram_gmm_vs_kmeans.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_gmm_vs_kmeans.png")


if __name__ == "__main__":
    main()
