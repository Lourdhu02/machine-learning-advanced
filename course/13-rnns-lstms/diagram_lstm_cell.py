"""LSTM cell schematic: gates labeled, additive cell-state highlighted.

Output: diagram_lstm_cell.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Cell outline
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 0.5), 7, 4,
                                         boxstyle="round,pad=0.1",
                                         facecolor="#f5f5fa", edgecolor="black", lw=1.2))

    # Inputs
    ax.text(0.0, 4.0, "x_t", fontsize=13, ha="right")
    ax.text(0.0, 1.5, "h_{t-1}", fontsize=13, ha="right")
    ax.text(0.0, 0.7, "c_{t-1}", fontsize=13, ha="right")

    # Gates
    gate_positions = {
        "forget σ":  (1.5, 1.7),
        "input σ":   (2.8, 1.7),
        "g  tanh":   (4.1, 1.7),
        "output σ":  (5.4, 1.7),
    }
    for name, (x, y) in gate_positions.items():
        ax.add_patch(mpatches.Circle((x, y), 0.3, facecolor="lightyellow", edgecolor="black"))
        ax.text(x, y, name.split()[1], ha="center", va="center", fontsize=10)
        ax.text(x, y - 0.5, name.split()[0], ha="center", va="center", fontsize=9, color="steelblue")

    # Cell-state pipeline (the additive highway)
    ax.annotate("", xy=(7.4, 3.5), xytext=(0.5, 3.5),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=4))
    ax.text(4.0, 3.8, "additive cell-state highway   c_t = f_t · c_{t-1} + i_t · g_t",
            ha="center", color="crimson", fontsize=11, fontweight="bold")

    # Gate -> cell-state interactions
    for x, label in [(1.5, "f"), (2.8, "i · g")]:
        ax.annotate("", xy=(x, 3.3), xytext=(x, 2.05),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.add_patch(mpatches.Circle((1.5, 3.3), 0.18, facecolor="white", edgecolor="black"))
    ax.text(1.5, 3.3, "×", ha="center", va="center", fontsize=14)
    ax.add_patch(mpatches.Circle((2.8, 3.3), 0.18, facecolor="white", edgecolor="black"))
    ax.text(2.8, 3.3, "+", ha="center", va="center", fontsize=14)

    # Output gate -> h_t
    ax.annotate("", xy=(6.6, 3.5), xytext=(5.4, 2.05),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(6.0, 2.7, "× tanh(c_t)", fontsize=10)

    # Outputs
    ax.annotate("", xy=(8.6, 3.5), xytext=(7.4, 3.5),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5))
    ax.text(8.7, 3.5, "c_t", fontsize=13)
    ax.annotate("", xy=(8.6, 2.5), xytext=(6.8, 3.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(8.7, 2.5, "h_t", fontsize=13)

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("LSTM cell: 4 gates + additive cell-state recurrence (red).",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig("diagram_lstm_cell.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_lstm_cell.png")


if __name__ == "__main__":
    main()
