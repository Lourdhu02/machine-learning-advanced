# 08 — Clustering (k-means, GMM, EM)

> Goal: derive k-means as a hard-assignment limit of EM on a Gaussian Mixture Model. Two algorithms, one underlying objective.

**Status:** stub.

Planned contents:
- k-means objective: minimize sum of squared distances to assigned centroid. The algorithm: alternate assignment ↔ centroid update. Why this is coordinate descent on a non-convex objective.
- Why k-means can get stuck in bad local minima; why k-means++ initialization helps.
- Gaussian Mixture Model: each cluster is a Gaussian with its own mean, covariance, and weight.
- Expectation-Maximization, derived for GMM:
  - E-step: compute soft assignment `γᵢₖ = P(zᵢ = k | xᵢ)`.
  - M-step: update means/covariances/weights as weighted MLE.
  - Proof sketch that log-likelihood never decreases.
- k-means = EM on isotropic Gaussians with fixed equal covariance, in the limit of hard assignments.
- Diagrams: k-means iterations; GMM ellipses fitted to the same data; soft vs hard assignment side by side.
- Mind-map: unsupervised → density estimation family.
- `from_scratch.py`: k-means + GMM/EM, both from scratch.
- When it breaks: choosing `k` (silhouette / BIC), non-spherical clusters (use GMM), clusters of different densities (use DBSCAN/HDBSCAN).

References (preview):
- Bishop §9.
- Dempster, Laird, Rubin — *Maximum Likelihood from Incomplete Data via the EM Algorithm* (1977). The original EM paper.
