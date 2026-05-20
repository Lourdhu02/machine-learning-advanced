"""Histogram of activations before and after a BatchNorm layer.

Output: diagram_bn_distributions.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import BatchNorm


def main():
    rng = np.random.default_rng(0)
    # Simulate "deep activations": skewed, large variance, off-center
    raw = rng.standard_normal((512, 1)) * 4.5 + 3.0
    raw[raw < 0] *= 0.3  # asymmetric squash to mimic post-ReLU + scaling

    bn = BatchNorm(n_features=1)
    out = bn.forward(raw, training=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(raw.ravel(), bins=40, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(raw.mean(), color="black", lw=1.5, ls="--",
                    label=f"mean = {raw.mean():.2f}")
    axes[0].set_title(f"Pre-BN activations  (mean={raw.mean():.2f}, std={raw.std():.2f})")
    axes[0].set_xlabel("activation value")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(out.ravel(), bins=40, color="crimson", alpha=0.8, edgecolor="white")
    axes[1].axvline(out.mean(), color="black", lw=1.5, ls="--",
                    label=f"mean = {out.mean():.2f}")
    axes[1].set_title(f"Post-BN activations  (mean={out.mean():.2f}, std={out.std():.2f})")
    axes[1].set_xlabel("activation value")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("BatchNorm re-centers and re-scales the layer's pre-activations to N(0, 1) (then learns γ, β)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_bn_distributions.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_bn_distributions.png")


if __name__ == "__main__":
    main()
