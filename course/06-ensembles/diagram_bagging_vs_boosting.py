"""Three side-by-side fits on the same noisy sin:
  - single deep regression tree (jagged, overfits)
  - bagged ensemble of deep trees (smooth, variance averaged out)
  - gradient boosting (gradual fit driven by residuals)

Output: diagram_bagging_vs_boosting.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import _RegressionTree, GradientBoostingRegressor, make_sin


def bagged_predict(X_train, y_train, X_test, n_trees=50, depth=8, seed=0):
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    preds = np.zeros((n_trees, X_test.shape[0]))
    for t in range(n_trees):
        idx = rng.integers(0, n, n)
        tree = _RegressionTree(max_depth=depth).fit(X_train[idx], y_train[idx])
        preds[t] = tree.predict(X_test)
    return preds.mean(axis=0)


def main():
    X, y = make_sin(n=200, noise=0.3, seed=0)
    Xg = np.linspace(-3, 3, 400).reshape(-1, 1)
    true_y = np.sin(Xg.ravel() * 1.5)

    # 1. single deep tree
    single = _RegressionTree(max_depth=8).fit(X, y)
    single_pred = single.predict(Xg)

    # 2. bagged ensemble
    bag_pred = bagged_predict(X, y, Xg, n_trees=50, depth=8)

    # 3. gradient boosting
    gbm = GradientBoostingRegressor(n_estimators=80, learning_rate=0.1, max_depth=3).fit(X, y)
    gbm_pred = gbm.predict(Xg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, pred, title, color in [
        (axes[0], single_pred, "Single deep tree\n(high variance, jagged)", "crimson"),
        (axes[1], bag_pred, "Bagging: 50 deep trees averaged\n(variance shrinks)", "steelblue"),
        (axes[2], gbm_pred, "Gradient boosting: 80 shallow trees\n(residual chase)", "darkgreen"),
    ]:
        ax.scatter(X.ravel(), y, color="lightgray", s=18, zorder=2)
        ax.plot(Xg.ravel(), true_y, "k--", lw=1.2, label="true f(x) = sin(1.5 x)", alpha=0.6)
        ax.plot(Xg.ravel(), pred, color=color, lw=2.4, label="ensemble prediction")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig("diagram_bagging_vs_boosting.png", dpi=140)
    print("wrote diagram_bagging_vs_boosting.png")


if __name__ == "__main__":
    main()
