"""Why divide by sqrt(d_k)? Without scaling, softmax becomes a one-hot for
moderate d_k and gradients vanish. Show the softmax distribution with and
without the scale.

Output: diagram_scale_factor.png
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    d_k = 64
    L = 8
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((L, d_k)) * 1.0
    K = rng.standard_normal((L, d_k)) * 1.0

    scores_unscaled = Q @ K.T
    scores_scaled = scores_unscaled / np.sqrt(d_k)

    attn_unscaled = softmax(scores_unscaled, axis=-1)
    attn_scaled = softmax(scores_scaled, axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, A, title in [
        (axes[0], attn_unscaled, f"Without 1/√d_k scaling  (d_k={d_k})"),
        (axes[1], attn_scaled, f"With 1/√d_k scaling  (d_k={d_k})"),
    ]:
        im = ax.imshow(A, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("attended-to position j")
        ax.set_ylabel("query position i")
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Unscaled softmax collapses to one-hot. Scaled stays in a sane range.",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_scale_factor.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_scale_factor.png")


if __name__ == "__main__":
    main()
