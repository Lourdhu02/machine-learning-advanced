# 03 — Regularization (Ridge, Lasso, ElasticNet)

> Goal: understand penalty terms geometrically. Why L2 shrinks weights smoothly; why L1 produces exact zeros.

**Status:** stub.

Planned contents:
- Ridge: closed-form `w* = (XᵀX + λI)⁻¹ Xᵀy`, derived. Why adding `λI` cures collinearity.
- Lasso: no closed form. Coordinate descent or proximal gradient. Geometry of the L1 ball with sharp corners → sparsity.
- ElasticNet: convex combination.
- Diagrams: L1 ball vs L2 ball intersecting the OLS contour; weight paths as `λ` varies.
- Mind-map: continuation of the linear-models family from module 01.
- `from_scratch.py`: Ridge closed-form + Lasso via coordinate descent.
- When it breaks: when none of your features matter (Lasso zeros everything), when features are scale-mismatched (always standardize first).

References (preview):
- Hastie, Tibshirani, Friedman — ESL §3.4.
- Tibshirani — *Regression Shrinkage and Selection via the Lasso* (1996).
