"""Adam's first-moment estimate with vs without bias correction, on a constant
gradient stream. Without correction, the early-step estimate is biased toward 0.

Output: diagram_bias_correction.png
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    beta1 = 0.9
    g = 1.0  # constant gradient -- true EMA is 1.0
    n = 60
    m = 0.0
    raw, corrected = [], []
    for t in range(1, n + 1):
        m = beta1 * m + (1 - beta1) * g
        raw.append(m)
        corrected.append(m / (1 - beta1 ** t))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, n + 1), raw, color="steelblue", lw=2.2,
            label="m_t (raw EMA, biased toward 0 early on)")
    ax.plot(range(1, n + 1), corrected, color="crimson", lw=2.2,
            label="m̂_t = m_t / (1 - β1^t) (bias-corrected)")
    ax.axhline(g, color="black", lw=1, ls="--", label="true gradient (target)")

    ax.set_xlabel("step t")
    ax.set_ylabel("first-moment estimate")
    ax.set_title("Adam bias correction: without it, early steps start near zero")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_bias_correction.png", dpi=140)
    print("wrote diagram_bias_correction.png")


if __name__ == "__main__":
    main()
