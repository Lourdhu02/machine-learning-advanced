"""Hinge loss vs log loss vs 0-1 loss as functions of yf(x).

Output: diagram_hinge_loss.png
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    z = np.linspace(-3, 3, 400)
    hinge = np.maximum(0, 1 - z)
    log = np.log1p(np.exp(-z)) / np.log(2)  # rescaled to match at z=0
    zero_one = (z < 0).astype(float)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(z, zero_one, color="gray", lw=2, label="0-1 loss  (non-convex, what we'd love to minimize)")
    ax.plot(z, log, color="steelblue", lw=2, label="logistic / log loss  (module 02)")
    ax.plot(z, hinge, color="crimson", lw=2.5, label="hinge loss  (module 04 / SVM)")

    ax.axvline(1, color="crimson", lw=0.6, ls=":")
    ax.text(1.05, 2.7, "y·f(x) = 1\n(SVM margin)", fontsize=9, color="crimson")
    ax.axvline(0, color="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.5)

    ax.set_xlabel("y · f(x)   (correct & confident →)")
    ax.set_ylabel("loss")
    ax.set_title("Why hinge gives sparsity: zero loss beyond the margin")
    ax.set_ylim(-0.2, 3.2)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_hinge_loss.png", dpi=140)
    print("wrote diagram_hinge_loss.png")


if __name__ == "__main__":
    main()
