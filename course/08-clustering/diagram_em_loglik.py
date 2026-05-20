"""EM log-likelihood vs iteration: must be monotonic non-decreasing.
Run EM on three different initializations to also visualize local-optimum
sensitivity.

Output: diagram_em_loglik.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import GaussianMixture, make_elongated


def main():
    X, _ = make_elongated(n=300, seed=3)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seed in [0, 7, 42]:
        gmm = GaussianMixture(n_components=3, seed=seed, n_iter=80).fit(X)
        ax.plot(gmm.loglik_history_, marker="o", lw=1.8,
                label=f"seed {seed}  (final log-lik = {gmm.loglik_history_[-1]:.2f})")

    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title("EM: monotonic non-decreasing log-likelihood (Jensen + tight ELBO)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_em_loglik.png", dpi=140)
    print("wrote diagram_em_loglik.png")


if __name__ == "__main__":
    main()
