"""k-means iterations 0, 1, 3, final on a 3-cluster dataset.
Points are colored by current assignment; centroids drawn as stars.

Output: diagram_kmeans_iters.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import make_spherical


def run_kmeans_capture(X, K=3, seed=0):
    """Run k-means and capture (centroids, labels) after each iteration."""
    rng = np.random.default_rng(seed)
    # bad-ish init for visualization (sample 3 random points)
    centroids = X[rng.choice(X.shape[0], K, replace=False)].astype(float)
    history = []
    for _ in range(20):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        labels = np.argmin(d2, axis=1)
        history.append((centroids.copy(), labels.copy()))
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
            for k in range(K)
        ])
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            history.append((new_centroids.copy(), labels.copy()))
            break
        centroids = new_centroids
    return history


def plot_one(ax, X, centroids, labels, title):
    colors = ["steelblue", "crimson", "darkgreen"]
    for k in range(3):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], edgecolors="white",
                   s=30, zorder=2)
        ax.scatter(*centroids[k], marker="*", c=colors[k], edgecolors="black",
                   s=300, lw=1.5, zorder=4)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def main():
    X, _ = make_spherical(n=300, seed=2)
    history = run_kmeans_capture(X, K=3, seed=42)

    iters_to_show = [0, 1, 3, len(history) - 1]
    iters_to_show = [i for i in iters_to_show if i < len(history)]
    fig, axes = plt.subplots(1, len(iters_to_show), figsize=(4.2 * len(iters_to_show), 4.2))

    for ax, t in zip(axes, iters_to_show):
        cents, lbls = history[t]
        plot_one(ax, X, cents, lbls, f"iter {t}" if t < len(history) - 1
                 else f"converged (iter {t})")

    fig.suptitle("k-means iterations: assignments flip, centroids slide",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_kmeans_iters.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_kmeans_iters.png")


if __name__ == "__main__":
    main()
