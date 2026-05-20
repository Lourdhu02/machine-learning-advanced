"""Visualize the convolution operation: input, kernel at two positions, output.

Output: diagram_convolution.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from from_scratch import conv2d_naive, SOBEL_X


def main():
    # Slightly bigger image to make positions readable
    img = np.zeros((6, 6))
    img[:, 2] = 1
    img[:, 3] = 1
    img[3, :] = 0.5  # add a horizontal stripe for variety

    out = conv2d_naive(img, SOBEL_X)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Input with two kernel positions highlighted
    ax = axes[0]
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
    ax.add_patch(Rectangle((0.5, 0.5), 3, 3, edgecolor="crimson", lw=3, fill=False))
    ax.add_patch(Rectangle((1.5, 1.5), 3, 3, edgecolor="steelblue", lw=3, fill=False))
    ax.set_title("Input image: kernel slid here (red)\nand here (blue)")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))

    # Kernel
    ax = axes[1]
    ax.imshow(SOBEL_X, cmap="RdBu_r", vmin=-2, vmax=2)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(SOBEL_X[i, j]):+d}",
                    ha="center", va="center", color="black", fontsize=14)
    ax.set_title("Sobel-x kernel (3×3)")
    ax.set_xticks([]); ax.set_yticks([])

    # Output
    ax = axes[2]
    im = ax.imshow(out, cmap="RdBu_r", vmin=-out.max(), vmax=out.max())
    ax.set_title(f"Output feature map ({out.shape[0]}×{out.shape[1]})\nlarge values = vertical edges")
    ax.set_xticks(range(out.shape[1])); ax.set_yticks(range(out.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig("diagram_convolution.png", dpi=140)
    print("wrote diagram_convolution.png")


if __name__ == "__main__":
    main()
