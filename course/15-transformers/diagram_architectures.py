"""Encoder-only / decoder-only / encoder-decoder architectures, side by side.

Output: diagram_architectures.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_stack(ax, x0, label, n_blocks=4, color="#fcd5c6"):
    for i in range(n_blocks):
        y = i * 0.8
        ax.add_patch(mpatches.FancyBboxPatch((x0 - 0.6, y), 1.2, 0.55,
                                             boxstyle="round,pad=0.05",
                                             facecolor=color, edgecolor="black"))
        ax.text(x0, y + 0.275, f"block {i+1}", ha="center", va="center", fontsize=9)
    ax.text(x0, n_blocks * 0.8 + 0.5, label, ha="center", va="bottom",
            fontsize=11, fontweight="bold")


def main():
    fig, ax = plt.subplots(figsize=(14, 7))

    # Encoder-only (BERT)
    draw_stack(ax, 1.0, "Encoder-only\n(BERT, RoBERTa)", color="#d6e7ff")
    ax.text(1.0, -0.5, "bidirectional\nattention", ha="center", fontsize=9)

    # Decoder-only (GPT, LLaMA)
    draw_stack(ax, 4.5, "Decoder-only\n(GPT, LLaMA, Mistral)", color="#fcd5c6")
    ax.text(4.5, -0.5, "causal mask\n(past tokens only)", ha="center", fontsize=9)

    # Encoder-decoder
    draw_stack(ax, 8.0, "Encoder", n_blocks=3, color="#d6e7ff")
    draw_stack(ax, 10.5, "Decoder", n_blocks=3, color="#fcd5c6")
    ax.text(9.25, 3.5, "Encoder-Decoder\n(T5, original Transformer)",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(8.0, -0.5, "bidirectional", ha="center", fontsize=9)
    ax.text(10.5, -0.5, "causal + cross-attn", ha="center", fontsize=9)
    # Cross-attention arrow
    for i in range(3):
        ax.annotate("", xy=(10.0, i * 0.8 + 0.275), xytext=(8.6, i * 0.8 + 0.275),
                    arrowprops=dict(arrowstyle="->", color="purple", lw=1.5,
                                    connectionstyle="arc3,rad=0.2"))
    ax.text(9.25, 0.05, "cross-attention", ha="center", color="purple", fontsize=9)

    ax.set_xlim(0, 12)
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Three Transformer architectures: same block, different masking + wiring",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig("diagram_architectures.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_architectures.png")


if __name__ == "__main__":
    main()
