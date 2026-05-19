"""Plot a matrix's action on the unit circle, with eigenvectors highlighted
as the only directions that don't get rotated.

Output: diagram_eigenvectors.png
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    A = np.array([[2.0, 1.0], [0.0, 1.5]])
    eigvals, eigvecs = np.linalg.eig(A)

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(circle[0], circle[1], "--", color="gray", label="unit circle (input)")
    ax.plot(ellipse[0], ellipse[1], "-", color="steelblue", label="A . circle (output)")

    for i in range(2):
        v = eigvecs[:, i]
        lam = eigvals[i].real
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.annotate("", xy=lam * v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
        ax.text(*(lam * v + 0.08), f"  lambda = {lam:.2f}", color="crimson")

    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title("Eigenvectors: directions A only scales, never rotates.")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_eigenvectors.png", dpi=140)
    print("wrote diagram_eigenvectors.png")


if __name__ == "__main__":
    main()
