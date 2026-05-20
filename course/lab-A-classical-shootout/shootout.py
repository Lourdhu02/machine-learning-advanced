"""Classical shootout: logistic vs kernel SVM vs random forest.

Imports the from-scratch implementations from modules 02, 04, 06 via
importlib (so we don't duplicate code). Trains all three on two datasets
designed to favor different geometries, reports accuracy + timing, and
writes a side-by-side decision-boundary grid.

Run: python shootout.py
"""

from __future__ import annotations
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Cross-module imports (each module's from_scratch.py)
# ---------------------------------------------------------------------------

COURSE = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, COURSE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required so @dataclass can find the module
    spec.loader.exec_module(mod)
    return mod


logreg_mod = _load("logreg_mod", "02-logistic-regression/from_scratch.py")
svm_mod = _load("svm_mod", "04-svm/from_scratch.py")
ens_mod = _load("ensembles_mod", "06-ensembles/from_scratch.py")

LogisticRegressionNewton = logreg_mod.LogisticRegressionNewton
KernelSVM = svm_mod.KernelSVM
RandomForestClassifier = ens_mod.RandomForestClassifier


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def make_blobs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.standard_normal((half, 2)) + np.array([-1.6, -1.1])
    X1 = rng.standard_normal((half, 2)) + np.array([1.6, 1.1])
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def make_moons(n=400, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    theta1 = np.pi * rng.uniform(0, 1, half)
    X0 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    theta2 = np.pi * rng.uniform(0, 1, half)
    X1 = np.stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)], axis=1)
    X = np.vstack([X0, X1]) + rng.standard_normal((n, 2)) * noise
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Uniform wrapper around the three classifiers (they have slightly
# different APIs and one needs ±1 labels)
# ---------------------------------------------------------------------------


class _Wrap:
    def __init__(self, name, model, needs_pm1=False):
        self.name = name
        self.model = model
        self.needs_pm1 = needs_pm1
        self.fit_time_ = 0.0

    def fit(self, X, y01):
        y = y01 * 2 - 1 if self.needs_pm1 else y01
        t0 = time.time()
        self.model.fit(X, y)
        self.fit_time_ = time.time() - t0
        return self

    def predict01(self, X):
        pred = self.model.predict(X)
        if self.needs_pm1:
            return ((pred > 0).astype(int))
        return pred.astype(int)

    def decision(self, X):
        # for contour plotting
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)
            if p.ndim == 2:
                return p[:, 1] - 0.5
            return p - 0.5  # logistic returns 1D P(y=1) directly
        return self.predict01(X).astype(float) - 0.5


def make_classifiers():
    return [
        _Wrap("Logistic Regression",
              LogisticRegressionNewton(n_iter=30), needs_pm1=False),
        _Wrap("Kernel SVM (RBF)",
              KernelSVM(kernel="rbf", C=1.0, gamma=1.0), needs_pm1=True),
        _Wrap("Random Forest",
              RandomForestClassifier(n_estimators=80, max_depth=8, seed=0), needs_pm1=False),
    ]


# ---------------------------------------------------------------------------
# Plotting + main
# ---------------------------------------------------------------------------


def plot_boundary(ax, wrap, X, y, title):
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 220),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 220),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = wrap.decision(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.55,
                vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)))
    ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=1.4)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="steelblue", edgecolors="white",
               s=25, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="crimson", edgecolors="white",
               s=25, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())


def run_on(X, y, label):
    n_tr = int(0.8 * len(X))
    Xt, Xv, yt, yv = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
    classifiers = make_classifiers()
    for c in classifiers:
        c.fit(Xt, yt)

    print(f"\n  {label}")
    print(f"  {'classifier':<22} {'train':>8} {'test':>8} {'fit (s)':>9}")
    for c in classifiers:
        acc_t = float(np.mean(c.predict01(Xt) == yt))
        acc_v = float(np.mean(c.predict01(Xv) == yv))
        print(f"  {c.name:<22} {acc_t:>8.3f} {acc_v:>8.3f} {c.fit_time_:>9.2f}")
    return classifiers, Xt, yt


def main():
    print("=" * 60)
    print("Classical shootout: logistic vs kernel SVM vs random forest")
    print("=" * 60)

    Xb, yb = make_blobs()
    Xm, ym = make_moons()

    blob_clfs, Xt_b, yt_b = run_on(Xb, yb, "Dataset 1: linearly separable blobs")
    moon_clfs, Xt_m, yt_m = run_on(Xm, ym, "Dataset 2: two interleaving moons")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for row, (clfs, Xt, yt, ds_label) in enumerate([
        (blob_clfs, Xt_b, yt_b, "blobs"),
        (moon_clfs, Xt_m, yt_m, "moons"),
    ]):
        for col, c in enumerate(clfs):
            plot_boundary(axes[row, col], c, Xt, yt, f"{c.name}\n[{ds_label}]")

    fig.suptitle("Decision boundaries: same data, three classifier families", fontsize=13)
    fig.tight_layout()
    fig.savefig("diagram_shootout.png", dpi=140, bbox_inches="tight")
    print("\nwrote diagram_shootout.png")


if __name__ == "__main__":
    main()
