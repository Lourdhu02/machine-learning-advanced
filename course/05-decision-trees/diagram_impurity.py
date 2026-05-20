"""Gini, entropy, and classification error as functions of p1 (binary).

Output: diagram_impurity.png
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    p = np.linspace(1e-3, 1 - 1e-3, 400)
    gini = 2 * p * (1 - p)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    error = np.minimum(p, 1 - p)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(p, entropy, color="steelblue", lw=2.2, label="entropy  H = -p log p - (1-p) log(1-p)")
    ax.plot(p, gini, color="crimson", lw=2.2,
            label="Gini  G = 2 p (1 - p)")
    ax.plot(p, error, color="gray", lw=2.0, ls="--",
            label="classification error  E = min(p, 1-p)")

    ax.axvline(0.5, color="black", lw=0.5, ls=":")
    ax.text(0.51, 0.01, "max impurity at p = 0.5", fontsize=9)

    ax.set_xlabel("p₁  (proportion of class 1 in the node)")
    ax.set_ylabel("impurity")
    ax.set_title("Why Gini and entropy beat classification error as split criteria")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig("diagram_impurity.png", dpi=140)
    print("wrote diagram_impurity.png")


if __name__ == "__main__":
    main()
