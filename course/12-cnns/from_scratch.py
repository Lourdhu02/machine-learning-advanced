"""Hand-coded 2D convolution + Sobel edge-detection demo.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# Naive 2D convolution (forward only -- backprop is sketched in the README)
# -----------------------------------------------------------------------------


def conv2d_naive(X: np.ndarray, K: np.ndarray,
                 stride: int = 1, padding: int = 0) -> np.ndarray:
    """X: (H, W). K: (kh, kw). Returns Y of shape based on stride/padding."""
    if padding > 0:
        X = np.pad(X, padding, mode="constant", constant_values=0)
    H, W = X.shape
    kh, kw = K.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    Y = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = X[i * stride:i * stride + kh, j * stride:j * stride + kw]
            Y[i, j] = np.sum(patch * K)
    return Y


# -----------------------------------------------------------------------------
# Multi-channel Conv2D layer (forward only)
# -----------------------------------------------------------------------------


class Conv2D:
    """Forward-only 2D conv layer with C_in input channels and C_out output channels.

    Weights: K of shape (C_out, C_in, k, k).
    Bias   : (C_out,).
    """

    def __init__(self, c_in: int, c_out: int, k: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (c_in * k * k))  # He init
        self.K = rng.standard_normal((c_out, c_in, k, k)) * scale
        self.b = np.zeros(c_out)
        self.k = k

    def forward(self, X: np.ndarray, padding: int = 0) -> np.ndarray:
        """X: (C_in, H, W). Returns (C_out, H', W')."""
        c_in, H, W = X.shape
        if padding > 0:
            Xp = np.pad(X, ((0, 0), (padding, padding), (padding, padding)),
                        mode="constant")
        else:
            Xp = X
        c_out = self.K.shape[0]
        k = self.k
        out_h = Xp.shape[1] - k + 1
        out_w = Xp.shape[2] - k + 1
        Y = np.zeros((c_out, out_h, out_w))
        for co in range(c_out):
            acc = np.zeros((out_h, out_w))
            for ci in range(c_in):
                acc += conv2d_naive(Xp[ci], self.K[co, ci], stride=1, padding=0)
            Y[co] = acc + self.b[co]
        return Y


# -----------------------------------------------------------------------------
# Synthetic image (vertical bars)
# -----------------------------------------------------------------------------


def make_bars(size: int = 16) -> np.ndarray:
    img = np.zeros((size, size), dtype=float)
    img[:, 4:6] = 1.0
    img[:, 10:12] = 1.0
    return img


# -----------------------------------------------------------------------------
# Sobel filters
# -----------------------------------------------------------------------------


SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)

SOBEL_Y = SOBEL_X.T


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    img = make_bars(16)
    print("input image (vertical bars at columns 4-5 and 10-11):")
    print(img.astype(int))

    sx = conv2d_naive(img, SOBEL_X, padding=1)
    sy = conv2d_naive(img, SOBEL_Y, padding=1)
    mag = np.sqrt(sx ** 2 + sy ** 2)

    print(f"\nSobel-x max abs : {np.abs(sx).max():.2f}  (large -> vertical edges present)")
    print(f"Sobel-y max abs : {np.abs(sy).max():.2f}  (small -> no horizontal edges)")
    print(f"edge magnitude max: {mag.max():.2f}")

    # Optional verification against scipy if available
    try:
        from scipy.signal import correlate2d
        ref = correlate2d(img, SOBEL_X, mode="same", boundary="fill")
        err = float(np.max(np.abs(sx - ref)))
        print(f"\nverification against scipy.signal.correlate2d: max diff = {err:.2e}")
        assert err < 1e-10, "conv2d_naive disagrees with scipy"
    except ImportError:
        print("(scipy not installed -- skipping verification)")

    # Tiny demo of multi-channel Conv2D layer
    X_rgb = np.stack([img, img * 0.5, img * 0.2])  # fake 3-channel
    layer = Conv2D(c_in=3, c_out=4, k=3, seed=0)
    Y = layer.forward(X_rgb, padding=1)
    print(f"\nConv2D(c_in=3, c_out=4, k=3).forward(X).shape = {Y.shape}")
    print("OK")


if __name__ == "__main__":
    main()
