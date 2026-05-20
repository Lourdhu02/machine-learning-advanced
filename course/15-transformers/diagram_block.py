"""Transformer block (pre-norm) schematic.

Output: diagram_block.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    fig, ax = plt.subplots(figsize=(8, 11))

    boxes = [
        ("Input x", 8.0, "#e0e0e0"),
        ("LayerNorm", 7.0, "#fff4d6"),
        ("Multi-Head Attention", 6.0, "#fcd5c6"),
        ("+ residual", 5.0, "#d6e7ff"),
        ("LayerNorm", 4.0, "#fff4d6"),
        ("FFN: Linear → GELU → Linear", 3.0, "#fcd5c6"),
        ("+ residual", 2.0, "#d6e7ff"),
        ("Output", 1.0, "#e0e0e0"),
    ]

    for label, y, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((2, y - 0.3), 5, 0.65,
                                             boxstyle="round,pad=0.05",
                                             facecolor=color, edgecolor="black", lw=1.2))
        ax.text(4.5, y, label, ha="center", va="center", fontsize=11)

    # Arrows between boxes
    ys = [box[1] for box in boxes]
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.annotate("", xy=(4.5, y1 + 0.3), xytext=(4.5, y0 - 0.3),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Residual skip-connection arrows
    for from_y, to_y in [(8.0, 5.0), (5.0, 2.0)]:
        ax.annotate("", xy=(2.0, to_y + 0.0), xytext=(2.0, from_y),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=2,
                                    connectionstyle="arc3,rad=0.4"))
        ax.text(0.4, (from_y + to_y) / 2, "residual\nskip", color="crimson",
                fontsize=9, ha="center", va="center")

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(0.3, 9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("One Transformer block (pre-norm variant)\n"
                 "Each sublayer:  LayerNorm → sublayer → residual add",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig("diagram_block.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_block.png")


if __name__ == "__main__":
    main()
