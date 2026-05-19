"""Plot a 2D loss surface (contour) and the gradient arrow at one point,
with one gradient-descent step in the OPPOSITE direction.

Output: diagram_gradient_geometry.png
"""

import numpy as np
import matplotlib.pyplot as plt


def grad(w):
    return np.array([w[0], 3 * w[1]])


def main():
    grid = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(grid, grid)
    L = 0.5 * W1**2 + 1.5 * W2**2

    point = np.array([2.0, 1.5])
    g = grad(point)
    lr = 0.25
    step = point - lr * g

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(W1, W2, L, levels=15, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8)

    ax.annotate("", xy=point + g * 0.25, xytext=point,
                arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
    ax.annotate("", xy=step, xytext=point,
                arrowprops=dict(arrowstyle="->", color="royalblue", lw=2))
    ax.plot(*point, "ko")
    ax.plot(*step, "ko")
    ax.text(point[0] + 0.15, point[1] + 0.55, "grad L (steepest ascent)", color="crimson")
    ax.text(step[0] - 1.6, step[1] - 0.35, "- eta * grad L (descent step)", color="royalblue")
    ax.text(point[0] + 0.1, point[1] - 0.15, "current w", color="black")

    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_title("Gradient = direction of steepest ascent. We step the other way.")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("diagram_gradient_geometry.png", dpi=140)
    print("wrote diagram_gradient_geometry.png")


if __name__ == "__main__":
    main()
