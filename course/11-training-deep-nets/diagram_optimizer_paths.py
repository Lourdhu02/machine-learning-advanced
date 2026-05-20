"""SGD vs Momentum vs Adam trajectories on a 2D banana loss surface.

Output: diagram_optimizer_paths.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import SGD, Momentum, Adam, banana_loss, race


def main():
    paths = {
        "SGD (lr=0.01)":      race(SGD,      lr=0.01, n_steps=600),
        "Momentum (lr=0.01)": race(Momentum, lr=0.01, n_steps=600),
        "Adam (lr=0.05)":     race(Adam,     lr=0.05, n_steps=600),
    }
    colors = {"SGD (lr=0.01)": "steelblue",
              "Momentum (lr=0.01)": "darkgreen",
              "Adam (lr=0.05)": "crimson"}

    grid = np.linspace(-2.0, 2.0, 200)
    X, Y = np.meshgrid(grid, grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = banana_loss(np.array([X[i, j], Y[i, j]]))

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.contour(X, Y, Z, levels=np.logspace(-2, 3, 25), cmap="Greys", alpha=0.5)
    for name, path in paths.items():
        ax.plot(path[:, 0], path[:, 1], "-", color=colors[name], lw=2.4,
                label=name, alpha=0.9)
        ax.plot(*path[0], "ko")
        ax.plot(*path[-1], marker="*", markersize=18, color=colors[name],
                markeredgecolor="black")

    ax.plot(1, 1, "x", color="black", markersize=14, mew=3,
            label="global minimum (1, 1)")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_title("Banana loss: SGD ping-pongs across the ravine; momentum + Adam glide along it")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_optimizer_paths.png", dpi=140)
    print("wrote diagram_optimizer_paths.png")


if __name__ == "__main__":
    main()
