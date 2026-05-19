"""Linear SVM on linearly separable data: hyperplane, both ±1 margin lines,
and support vectors highlighted.

Output: diagram_margin.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import LinearSVM, make_blobs


def main():
    X, y = make_blobs(n=120, seed=1)
    svm = LinearSVM(lam=0.02, lr=0.05, n_iter=4000).fit(X, y)

    w, b = svm.w_, svm.b_

    pad = 0.5
    xx = np.array([X[:, 0].min() - pad, X[:, 0].max() + pad])
    # decision boundary: w[0]*x + w[1]*y + b = 0
    yy_boundary = -(w[0] * xx + b) / w[1]
    yy_pos = -(w[0] * xx + b - 1) / w[1]
    yy_neg = -(w[0] * xx + b + 1) / w[1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(xx, yy_boundary, "k-", lw=2, label="decision boundary  w·x + b = 0")
    ax.plot(xx, yy_pos, "k--", lw=1.2, label="margin  w·x + b = +1")
    ax.plot(xx, yy_neg, "k--", lw=1.2, label="margin  w·x + b = −1")
    ax.fill_between(xx, yy_neg, yy_pos, color="gold", alpha=0.15)

    ax.scatter(X[y == -1, 0], X[y == -1, 1], c="steelblue", edgecolors="white",
               s=55, label="class −1", zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="crimson", edgecolors="white",
               s=55, label="class +1", zorder=3)

    sv = svm.support_idx_
    ax.scatter(X[sv, 0], X[sv, 1], facecolors="none", edgecolors="black",
               s=180, lw=2, label="support vectors (margin ≤ 1)", zorder=4)

    margin_width = 2.0 / np.linalg.norm(w)
    ax.set_title(f"Linear SVM:  margin width = 2/‖w‖ = {margin_width:.2f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(xx[0], xx[1])
    ax.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_margin.png", dpi=140)
    print("wrote diagram_margin.png")


if __name__ == "__main__":
    main()
