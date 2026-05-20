"""Sobel-x / Sobel-y / edge magnitude on a synthetic geometric image.

Output: diagram_filters.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import conv2d_naive, SOBEL_X, SOBEL_Y


def make_geo_image(size=64):
    img = np.zeros((size, size), dtype=float)
    # vertical stripe
    img[:, 8:14] = 1.0
    # horizontal stripe
    img[40:46, :] = 1.0
    # diagonal-ish square
    img[15:35, 30:50] = 0.6
    # smooth gradient corner
    yy, xx = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
    img += 0.3 * (1 - yy) * xx
    return img


def main():
    img = make_geo_image(64)
    sx = conv2d_naive(img, SOBEL_X, padding=1)
    sy = conv2d_naive(img, SOBEL_Y, padding=1)
    mag = np.sqrt(sx ** 2 + sy ** 2)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))

    for ax, im, title in [
        (axes[0], img, "input image"),
        (axes[1], sx, "Sobel-x  (vertical edges)"),
        (axes[2], sy, "Sobel-y  (horizontal edges)"),
        (axes[3], mag, "edge magnitude"),
    ]:
        if title == "input image":
            ax.imshow(im, cmap="gray_r")
        elif title == "edge magnitude":
            ax.imshow(im, cmap="gray_r")
        else:
            v = max(abs(im.min()), abs(im.max()))
            ax.imshow(im, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Hand-coded edge detectors -- the first layer of a trained CNN learns versions of these",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_filters.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_filters.png")


if __name__ == "__main__":
    main()
