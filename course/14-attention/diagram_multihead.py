"""Schematic of multi-head attention: parallel projections -> per-head
attention -> concat -> output projection.

Output: diagram_multihead.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Input
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 4.0), 1.4, 0.7,
                                         boxstyle="round,pad=0.1",
                                         facecolor="lightblue", edgecolor="black"))
    ax.text(1.2, 4.35, "Input X\n(L, d_model)", ha="center", va="center", fontsize=10)

    # 3 heads of projections
    head_x = [3.0, 5.5, 8.0]
    for h, x0 in enumerate(head_x):
        # WQ / WK / WV projections
        for j, label in enumerate(["W_Q^h", "W_K^h", "W_V^h"]):
            ax.add_patch(mpatches.Rectangle((x0 + 0.3 * j - 0.1, 3.0), 0.25, 0.5,
                                            facecolor="lightyellow", edgecolor="black"))
            ax.text(x0 + 0.3 * j + 0.02, 3.25,
                    label.replace("^h", f"^{h+1}"), ha="center", va="center", fontsize=8)
        # arrow input -> projections
        ax.annotate("", xy=(x0 + 0.3, 3.5), xytext=(1.5, 4.0),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2,
                                    connectionstyle="arc3,rad=0.05"))
        # Attention block
        ax.add_patch(mpatches.FancyBboxPatch((x0 - 0.05, 1.7), 1.0, 0.7,
                                             boxstyle="round,pad=0.1",
                                             facecolor="lightcoral", edgecolor="black"))
        ax.text(x0 + 0.45, 2.05, f"Attention\n(head {h+1})", ha="center", va="center", fontsize=9)
        ax.annotate("", xy=(x0 + 0.45, 2.4), xytext=(x0 + 0.4, 2.95),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Concat
    ax.add_patch(mpatches.FancyBboxPatch((3.0, 0.5), 6.0, 0.7,
                                         boxstyle="round,pad=0.1",
                                         facecolor="lightyellow", edgecolor="black"))
    ax.text(6.0, 0.85, "Concat heads → (L, d_model)", ha="center", va="center", fontsize=11)
    for x0 in head_x:
        ax.annotate("", xy=(x0 + 0.45, 1.2), xytext=(x0 + 0.45, 1.65),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Output projection
    ax.add_patch(mpatches.FancyBboxPatch((4.5, -0.8), 3.0, 0.7,
                                         boxstyle="round,pad=0.1",
                                         facecolor="lightgreen", edgecolor="black"))
    ax.text(6.0, -0.45, "W_O   →   Output", ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(6.0, -0.1), xytext=(6.0, 0.45),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))

    ax.set_xlim(0, 11)
    ax.set_ylim(-1.5, 5.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Multi-head attention: h heads run in parallel, concat, final projection",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig("diagram_multihead.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_multihead.png")


if __name__ == "__main__":
    main()
