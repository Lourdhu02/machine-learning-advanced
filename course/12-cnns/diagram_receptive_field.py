"""Receptive field growth with depth: input pixels reachable by output neurons
at layers 1, 2, 3 of a stack of 3x3 convs (stride 1).

Output: diagram_receptive_field.png
"""

import numpy as np
import matplotlib.pyplot as plt


def receptive_field(layer: int, k: int = 3, stride: int = 1) -> int:
    """RF size after `layer` stacked k x k convs with stride 1."""
    rf = 1
    for _ in range(layer):
        rf = rf + (k - 1)
    return rf


def main():
    image_size = 15
    center = image_size // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))

    for ax, layer in zip(axes, [1, 2, 3]):
        rf = receptive_field(layer, k=3)
        # draw a grid
        grid = np.zeros((image_size, image_size))
        half = rf // 2
        lo = max(0, center - half)
        hi = min(image_size, center + half + 1)
        grid[lo:hi, lo:hi] = 0.6
        grid[center, center] = 1.0

        ax.imshow(grid, cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"Layer {layer}:  RF = {rf}×{rf} pixels", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Receptive field after stacking 3x3 convs (stride 1):\n"
                 "each layer adds (k-1)=2 pixels to the RF -- depth is cheaper than width",
                 fontsize=11, y=1.05)
    fig.tight_layout()
    fig.savefig("diagram_receptive_field.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_receptive_field.png")


if __name__ == "__main__":
    main()
