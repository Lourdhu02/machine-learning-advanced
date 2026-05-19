"""The classic L1-vs-L2 geometry diagram: OLS contour ellipse,
constraint sets (circle for L2, diamond for L1), and the touch points.

Output: diagram_penalty_geometry.png
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # An OLS quadratic centered away from the origin so the touch points are visible.
    w_ols = np.array([1.8, 1.2])
    # Random PSD matrix for the quadratic form (the "ellipse" shape)
    A = np.array([[1.4, 0.9], [0.9, 1.0]])

    grid = np.linspace(-2, 3, 400)
    W1, W2 = np.meshgrid(grid, grid)
    diff_1 = W1 - w_ols[0]
    diff_2 = W2 - w_ols[1]
    Z = (
        A[0, 0] * diff_1**2
        + 2 * A[0, 1] * diff_1 * diff_2
        + A[1, 1] * diff_2**2
    )

    # Find the touch points by minimizing Z subject to ||w||_p = c (numerically)
    def project_l2(t):
        return np.array([np.cos(t), np.sin(t)]) * 1.4

    def project_l1(s):
        # Parametrize the diamond ||w||_1 = 1.4
        c = 1.4
        return np.array([c * (1 - 2 * abs(s) if s >= 0 else -1 + 2 * abs(s)), c * s])

    # Just sweep parametrizations and pick the minimum
    ts = np.linspace(0, 2 * np.pi, 1000)
    l2_pts = np.array([project_l2(t) for t in ts])
    l2_z = ((l2_pts - w_ols) @ A * (l2_pts - w_ols)).sum(axis=1)
    l2_touch = l2_pts[np.argmin(l2_z)]

    # L1 diamond parameterization: traverse the four edges
    edges = [
        np.linspace([1.4, 0], [0, 1.4], 250),
        np.linspace([0, 1.4], [-1.4, 0], 250),
        np.linspace([-1.4, 0], [0, -1.4], 250),
        np.linspace([0, -1.4], [1.4, 0], 250),
    ]
    l1_pts = np.vstack(edges)
    l1_z = ((l1_pts - w_ols) @ A * (l1_pts - w_ols)).sum(axis=1)
    l1_touch = l1_pts[np.argmin(l1_z)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, title, touch, shape_pts in [
        (axes[0], "L2 (Ridge): smooth ball, touch can land anywhere", l2_touch, l2_pts),
        (axes[1], "L1 (Lasso): diamond, corners pull touch onto an axis", l1_touch, l1_pts),
    ]:
        ax.contour(W1, W2, Z, levels=12, cmap="Greys", alpha=0.7)
        ax.plot(shape_pts[:, 0], shape_pts[:, 1], color="steelblue", lw=2,
                label=r"$\|w\|_p = c$")
        ax.plot(*w_ols, "k.", markersize=10)
        ax.text(w_ols[0] + 0.05, w_ols[1] + 0.05, "OLS minimum", fontsize=10)
        ax.plot(*touch, "r*", markersize=18, label="constrained optimum")
        ax.text(touch[0] + 0.05, touch[1] + 0.05,
                f"w*=({touch[0]:.2f}, {touch[1]:.2f})", color="crimson", fontsize=10)

        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$w_1$")
        ax.set_ylabel(r"$w_2$")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_penalty_geometry.png", dpi=140)
    print("wrote diagram_penalty_geometry.png")


if __name__ == "__main__":
    main()
