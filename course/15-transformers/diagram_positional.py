"""Sinusoidal positional encoding visualized as a (pos, feature) heatmap.

Output: diagram_positional.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import sinusoidal_position_encoding


def main():
    L = 64
    d_model = 64
    pe = sinusoidal_position_encoding(L, d_model)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Full heatmap
    im0 = axes[0].imshow(pe, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_xlabel("feature dimension i")
    axes[0].set_ylabel("position pos")
    axes[0].set_title("Sinusoidal PE:  PE[pos, 2i]=sin(pos/10000^(2i/d)),\n  PE[pos, 2i+1]=cos(...)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    # A few feature dimensions over position (different frequencies)
    for i, color in zip([0, 4, 16, 32], ["crimson", "darkorange", "darkgreen", "steelblue"]):
        axes[1].plot(pe[:, i], color=color, lw=2, label=f"feature i={i}")
    axes[1].set_xlabel("position pos")
    axes[1].set_ylabel("PE value")
    axes[1].set_title("Different feature dims have different frequencies\n(coarse position ↔ fine position)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_positional.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_positional.png")


if __name__ == "__main__":
    main()
