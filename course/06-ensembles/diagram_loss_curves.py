"""Learning curves: train/test MSE vs number of trees for bagging and gradient
boosting on the same noisy sin. Bagging plateaus, GBM can over-fit if you
keep going past the sweet spot.

Output: diagram_loss_curves.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import _RegressionTree, GradientBoostingRegressor, make_sin


def bag_train_test_curve(X_tr, y_tr, X_te, y_te, n_trees=150, depth=8, seed=0):
    rng = np.random.default_rng(seed)
    n = X_tr.shape[0]
    sum_tr = np.zeros_like(y_tr, dtype=float)
    sum_te = np.zeros_like(y_te, dtype=float)
    train_mse, test_mse = [], []
    for t in range(n_trees):
        idx = rng.integers(0, n, n)
        tree = _RegressionTree(max_depth=depth).fit(X_tr[idx], y_tr[idx])
        sum_tr += tree.predict(X_tr)
        sum_te += tree.predict(X_te)
        pred_tr = sum_tr / (t + 1)
        pred_te = sum_te / (t + 1)
        train_mse.append(float(np.mean((y_tr - pred_tr) ** 2)))
        test_mse.append(float(np.mean((y_te - pred_te) ** 2)))
    return train_mse, test_mse


def gbm_train_test_curve(X_tr, y_tr, X_te, y_te, n_trees=300, lr=0.1, depth=3):
    gbm = GradientBoostingRegressor(n_estimators=n_trees,
                                     learning_rate=lr, max_depth=depth).fit(X_tr, y_tr)
    train_mse, test_mse = [], []
    for t in range(1, n_trees + 1):
        pred_tr = gbm.predict(X_tr, n_trees=t)
        pred_te = gbm.predict(X_te, n_trees=t)
        train_mse.append(float(np.mean((y_tr - pred_tr) ** 2)))
        test_mse.append(float(np.mean((y_te - pred_te) ** 2)))
    return train_mse, test_mse


def main():
    X, y = make_sin(n=200, noise=0.4, seed=0)
    n_tr = 140
    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    bag_tr, bag_te = bag_train_test_curve(X_tr, y_tr, X_te, y_te, n_trees=150, depth=8)
    gbm_tr, gbm_te = gbm_train_test_curve(X_tr, y_tr, X_te, y_te, n_trees=300, lr=0.1, depth=3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tr, te, title in [
        (axes[0], bag_tr, bag_te,
         "Bagging: 50+ trees → smooth plateau\n(more trees doesn't hurt)"),
        (axes[1], gbm_tr, gbm_te,
         "Gradient boosting: train MSE keeps dropping,\ntest MSE finds a sweet spot then drifts up"),
    ]:
        ax.plot(tr, color="crimson", lw=1.8, label="train MSE")
        ax.plot(te, color="steelblue", lw=1.8, label="test MSE")
        ax.set_xlabel("# trees")
        ax.set_ylabel("MSE")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_loss_curves.png", dpi=140)
    print("wrote diagram_loss_curves.png")


if __name__ == "__main__":
    main()
