"""Attention weights on a toy 8-token sequence.

Output: diagram_attention_heatmap.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import scaled_dot_product_attention, make_causal_mask


def main():
    L, d = 8, 16
    rng = np.random.default_rng(1)

    # Make a synthetic sequence where tokens 0..3 are "similar to each other"
    # and 4..7 are "similar to each other", by sampling from two centers.
    center_a = rng.standard_normal(d) * 0.5
    center_b = rng.standard_normal(d) * 0.5
    X = np.vstack([
        np.tile(center_a, (4, 1)) + rng.standard_normal((4, d)) * 0.15,
        np.tile(center_b, (4, 1)) + rng.standard_normal((4, d)) * 0.15,
    ])

    _, attn = scaled_dot_product_attention(X, X, X)
    mask = make_causal_mask(L)
    _, attn_causal = scaled_dot_product_attention(X, X, X, mask=mask)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, A, title in [
        (axes[0], attn, "Self-attention\n(token 0-3 cluster, 4-7 cluster)"),
        (axes[1], attn_causal,
         "Causal self-attention\n(upper triangle zeroed)"),
    ]:
        im = ax.imshow(A, cmap="viridis", vmin=0, vmax=1)
        for i in range(L):
            for j in range(L):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                        color="white" if A[i, j] < 0.5 else "black", fontsize=8)
        ax.set_xlabel("attended-to position j")
        ax.set_ylabel("query position i")
        ax.set_title(title)
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig("diagram_attention_heatmap.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_attention_heatmap.png")


if __name__ == "__main__":
    main()
