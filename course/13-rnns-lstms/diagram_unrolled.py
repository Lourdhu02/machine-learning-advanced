"""An unrolled 4-step RNN with forward (blue) and backward / BPTT (red) arrows.

Output: diagram_unrolled.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    fig, ax = plt.subplots(figsize=(13, 5))

    T = 4
    x_positions = list(range(T))

    # Inputs x_t (bottom), hidden h_t (middle), outputs y_t (top)
    for t in range(T):
        # x_t
        ax.add_patch(mpatches.Rectangle((t - 0.2, -1.3), 0.4, 0.4,
                                        facecolor="lightgray", edgecolor="black"))
        ax.text(t, -1.1, f"x_{t}", ha="center", va="center", fontsize=11)
        # h_t
        ax.add_patch(mpatches.Circle((t, 0), 0.32, facecolor="lightyellow", edgecolor="black"))
        ax.text(t, 0, f"h_{t}", ha="center", va="center", fontsize=12, fontweight="bold")
        # y_t
        ax.add_patch(mpatches.Rectangle((t - 0.2, 1.1), 0.4, 0.4,
                                        facecolor="lightblue", edgecolor="black"))
        ax.text(t, 1.3, f"y_{t}", ha="center", va="center", fontsize=11)

        # input -> h
        ax.annotate("", xy=(t, -0.35), xytext=(t, -0.9),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.8))
        # h -> y
        ax.annotate("", xy=(t, 1.05), xytext=(t, 0.35),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.8))

    # Recurrent forward arrows (h_{t-1} -> h_t)
    for t in range(T - 1):
        ax.annotate("", xy=(t + 1 - 0.35, 0), xytext=(t + 0.35, 0),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=2.4))
        ax.text(t + 0.5, 0.15, "W_h", color="steelblue", fontsize=9)

    # BPTT backward arrows (red dashed, just above center)
    for t in range(T - 1):
        ax.annotate("", xy=(t + 0.35, -0.2), xytext=(t + 1 - 0.35, -0.2),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=2,
                                    linestyle="dashed"))

    ax.text(-0.5, 0, "h:", fontsize=11, ha="right")
    ax.text(-0.5, 1.3, "y:", fontsize=11, ha="right")
    ax.text(-0.5, -1.1, "x:", fontsize=11, ha="right")

    ax.plot([], [], color="steelblue", lw=2.4, label="forward (computes h_t, y_t)")
    ax.plot([], [], color="crimson", lw=2, linestyle="dashed",
            label="BPTT (dL/dh propagates back through W_h^T)")
    ax.legend(loc="upper left")

    ax.set_xlim(-1, T + 0.2)
    ax.set_ylim(-1.7, 2.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Unrolled RNN: shared weights W_x, W_h, W_y across all timesteps")

    fig.tight_layout()
    fig.savefig("diagram_unrolled.png", dpi=140)
    print("wrote diagram_unrolled.png")


if __name__ == "__main__":
    main()
