"""Sigmoid / tanh / ReLU / GELU and their derivatives, side by side.

Output: diagram_activations.png
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def tanh(z): return np.tanh(z)
def relu(z): return np.maximum(0, z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


def sigmoid_d(z):
    s = sigmoid(z); return s * (1 - s)


def tanh_d(z): return 1 - np.tanh(z)**2
def relu_d(z): return (z > 0).astype(float)


def gelu_d(z, h=1e-4):  # numerical
    return (gelu(z + h) - gelu(z - h)) / (2 * h)


def main():
    z = np.linspace(-4, 4, 400)
    funcs = [
        ("Sigmoid", sigmoid, sigmoid_d, "steelblue"),
        ("Tanh", tanh, tanh_d, "darkgreen"),
        ("ReLU", relu, relu_d, "crimson"),
        ("GELU", gelu, gelu_d, "darkorange"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for name, f, df, color in funcs:
        axes[0].plot(z, f(z), label=name, color=color, lw=2.2)
        axes[1].plot(z, df(z), label=name, color=color, lw=2.2)

    for ax, title in [(axes[0], "Activation"), (axes[1], "Derivative")]:
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("z")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylim(-1.5, 4.5)
    axes[1].set_ylim(-0.1, 1.2)

    fig.tight_layout()
    fig.savefig("diagram_activations.png", dpi=140)
    print("wrote diagram_activations.png")


if __name__ == "__main__":
    main()
