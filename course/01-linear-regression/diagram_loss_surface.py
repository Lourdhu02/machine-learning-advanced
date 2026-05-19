"""3D MSE loss surface over (w, b) with the gradient-descent path overlaid,
plus a top-down contour view.

Output: diagram_loss_surface.png
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2, 40)
    y = 1.5 * x + 0.5 + rng.standard_normal(x.size) * 0.4

    w_grid = np.linspace(-1, 4, 60)
    b_grid = np.linspace(-2, 3, 60)
    W, B = np.meshgrid(w_grid, b_grid)
    loss = np.mean((W[..., None] * x + B[..., None] - y) ** 2, axis=-1)

    # GD path from a far starting point
    w, b = -0.5, -1.5
    lr = 0.08
    path = [(w, b)]
    for _ in range(30):
        y_hat = w * x + b
        gw = 2 * np.mean((y_hat - y) * x)
        gb = 2 * np.mean(y_hat - y)
        w -= lr * gw
        b -= lr * gb
        path.append((w, b))
    path = np.array(path)
    path_loss = np.array([np.mean((p[0] * x + p[1] - y) ** 2) for p in path])

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(W, B, loss, cmap="viridis", alpha=0.7, linewidth=0)
    ax1.plot(path[:, 0], path[:, 1], path_loss, "r.-", lw=2, label="GD path")
    ax1.set_xlabel("w")
    ax1.set_ylabel("b")
    ax1.set_zlabel("MSE")
    ax1.set_title("Loss surface")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contour(W, B, loss, levels=20, cmap="viridis")
    ax2.clabel(cs, inline=True, fontsize=7)
    ax2.plot(path[:, 0], path[:, 1], "r.-", lw=2, label="GD path")
    ax2.plot(path[0, 0], path[0, 1], "ko", label="start")
    ax2.plot(path[-1, 0], path[-1, 1], "g*", markersize=14, label="end")
    ax2.set_xlabel("w")
    ax2.set_ylabel("b")
    ax2.set_title("Same surface, top-down")
    ax2.legend()
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("diagram_loss_surface.png", dpi=140)
    print("wrote diagram_loss_surface.png")


if __name__ == "__main__":
    main()
