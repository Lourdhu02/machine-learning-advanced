"""PCA vs t-SNE on blobs (linear) and Swiss roll (manifold).

Imports the from-scratch PCA and t-SNE from module 09.

Run: python compare.py
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


COURSE = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COURSE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dr_mod = _load("dr_mod", "09-dim-reduction/from_scratch.py")
PCA = dr_mod.PCA
tsne = dr_mod.tsne
make_blobs_hd = dr_mod.make_blobs_hd


def make_swiss_roll(n=600, noise=0.0, seed=0):
    """Standard Swiss roll: 2D manifold curled into 3D. Color = position along the roll."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))  # angle
    h = 21 * rng.uniform(0, 1, n)                     # height
    x = t * np.cos(t)
    z = t * np.sin(t)
    X = np.stack([x, h, z], axis=1)
    X += rng.standard_normal(X.shape) * noise
    return X, t  # t serves as the color label


def plot_panel(ax, Z, color, title, cmap="viridis"):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap=cmap, s=20, edgecolors="white", lw=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # --- row 1: blobs ---
    Xb, yb = make_blobs_hd(n_per_cluster=40, d=50, n_clusters=4, sep=4.0, seed=0)
    pca_b = PCA().fit(Xb)
    Z_pca_b = pca_b.transform(Xb, k=2)
    print("running t-SNE on blobs (~1 min)...")
    Y_tsne_b = tsne(Xb, n_iter=400, perplexity=20.0, seed=0)

    plot_panel(axes[0, 0], Z_pca_b, yb, "PCA — 4 Gaussian blobs in 50-D")
    plot_panel(axes[0, 1], Y_tsne_b, yb, "t-SNE — same data")

    # --- row 2: swiss roll ---
    Xs, ts = make_swiss_roll(n=600, noise=0.05, seed=0)
    pca_s = PCA().fit(Xs)
    Z_pca_s = pca_s.transform(Xs, k=2)
    print("running t-SNE on swiss roll (~1-2 min)...")
    Y_tsne_s = tsne(Xs, n_iter=500, perplexity=30.0, seed=0)

    plot_panel(axes[1, 0], Z_pca_s, ts,
               "PCA — Swiss roll (crushes curl, colour gradient overlaps)")
    plot_panel(axes[1, 1], Y_tsne_s, ts,
               "t-SNE — unrolls the spiral, clean colour gradient")

    fig.suptitle("PCA vs t-SNE: linear vs manifold data", fontsize=13)
    fig.tight_layout()
    fig.savefig("diagram_compare.png", dpi=140, bbox_inches="tight")
    print("\nwrote diagram_compare.png")


if __name__ == "__main__":
    main()
