"""The curse of dimensionality, visualized: min / mean / max pairwise distance
between random uniform points, as dimension grows. The min/max ratio shoots
toward 1 — distances become uninformative.

Output: diagram_curse.png
"""

import numpy as np
import matplotlib.pyplot as plt


def distance_stats(d, n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    sq = np.sum(X**2, axis=1, keepdims=True)
    dists = np.sqrt(np.clip(sq + sq.T - 2 * X @ X.T, 0, None))
    iu = np.triu_indices(n, k=1)
    pairs = dists[iu]
    return pairs.min(), pairs.mean(), pairs.max()


def main():
    dims = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500, 1000]
    mins, means, maxs = [], [], []
    for d in dims:
        mn, me, mx = distance_stats(d, n=200, seed=0)
        mins.append(mn); means.append(me); maxs.append(mx)
    mins = np.array(mins); means = np.array(means); maxs = np.array(maxs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(dims, mins, "o-", color="steelblue", label="min pairwise distance")
    axes[0].plot(dims, means, "o-", color="black", label="mean pairwise distance")
    axes[0].plot(dims, maxs, "o-", color="crimson", label="max pairwise distance")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("dimension d  (log scale)")
    axes[0].set_ylabel("Euclidean distance")
    axes[0].set_title("All pairwise distances grow with d, but…")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(dims, (maxs - mins) / means, "o-", color="darkgreen", lw=2.2)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("dimension d  (log scale)")
    axes[1].set_ylabel("(max − min) / mean")
    axes[1].set_title("…their spread vanishes: nearest is barely closer than farthest")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_curse.png", dpi=140)
    print("wrote diagram_curse.png")


if __name__ == "__main__":
    main()
