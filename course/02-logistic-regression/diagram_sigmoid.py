"""Sigmoid curve next to its hard-threshold "step" alternative,
and binary cross-entropy loss as a function of predicted probability.

Output: diagram_sigmoid.png
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    z = np.linspace(-7, 7, 400)
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    step = (z >= 0).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(z, step, color="lightgray", lw=2, label="hard threshold")
    ax.plot(z, sigmoid, color="crimson", lw=2.5, label=r"$\sigma(z) = 1 / (1 + e^{-z})$")
    ax.axhline(0.5, color="black", lw=0.5, ls="--")
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("z = w·x + b  (log-odds)")
    ax.set_ylabel("predicted probability P(y=1 | x)")
    ax.set_title("Sigmoid: smooth, differentiable squash from R to (0, 1)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    p = np.linspace(1e-3, 1 - 1e-3, 400)
    loss_y1 = -np.log(p)
    loss_y0 = -np.log(1 - p)
    ax.plot(p, loss_y1, color="crimson", lw=2, label="loss when true y = 1 :  -log p")
    ax.plot(p, loss_y0, color="steelblue", lw=2, label="loss when true y = 0 :  -log(1 - p)")
    ax.set_xlabel("predicted p")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("BCE blows up when the model is confidently wrong")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 5)

    fig.tight_layout()
    fig.savefig("diagram_sigmoid.png", dpi=140)
    print("wrote diagram_sigmoid.png")


if __name__ == "__main__":
    main()
