"""Visualize AdaBoost sample weights evolving across rounds.
Each panel: 2D moons data, point sizes proportional to current sample weight.
Misclassified points balloon; correctly classified shrink.

Output: diagram_adaboost_weights.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import AdaBoostClassifier, make_moons


def main():
    X, y = make_moons(n=200, noise=0.25, seed=1)
    y_pm = y * 2 - 1

    ada = AdaBoostClassifier(n_estimators=15).fit(X, y_pm)
    weights = ada.weight_history_  # list of arrays, length 16

    rounds_to_show = [0, 2, 5, 14]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, t in zip(axes, rounds_to_show):
        w = weights[t]
        sizes = w / w.max() * 350 + 12
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=sizes[y == 0],
                   c="steelblue", edgecolors="white", linewidths=0.7, alpha=0.85)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=sizes[y == 1],
                   c="crimson", edgecolors="white", linewidths=0.7, alpha=0.85)
        title = "initial uniform weights" if t == 0 else f"after round {t}"
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle("AdaBoost sample weights:  misclassified samples balloon, easy ones shrink",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("diagram_adaboost_weights.png", dpi=140, bbox_inches="tight")
    print("wrote diagram_adaboost_weights.png")


if __name__ == "__main__":
    main()
