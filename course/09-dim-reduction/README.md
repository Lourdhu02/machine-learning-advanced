# 09 — Dimensionality Reduction (PCA, t-SNE, UMAP)

> Goal: derive PCA from the SVD, then understand t-SNE / UMAP as preserving *local* structure while PCA preserves *global* variance.

**Status:** stub.

Planned contents:
- PCA derived two ways:
  - Maximum variance: find direction `v` that maximizes `Var(Xv)` ⇒ eigenvector of `XᵀX` with largest eigenvalue.
  - Minimum reconstruction error: find rank-`k` projection that minimizes `‖X − X̂‖²`.
- Both derivations land on the same answer via SVD: `X = U Σ Vᵀ`, top-`k` columns of `V` are the principal components.
- Whitening, explained-variance ratio, when to use PCA before another algorithm.
- t-SNE: preserve pairwise *similarities* (Gaussian in input space, t-distribution in 2D output) by minimizing KL divergence between the two distributions.
- UMAP: similar goal, fuzzy simplicial sets, faster.
- Diagrams: PCA on 2D Gaussian blob (rotation), Swiss-roll where PCA fails but t-SNE/UMAP unroll it.
- Mind-map: linear vs non-linear manifold learning.
- `from_scratch.py`: PCA via SVD in 10 lines; tiny t-SNE on a 2D toy set.
- When it breaks: PCA fails on non-linear manifolds; t-SNE distances are NOT meaningful, only neighbours are; t-SNE/UMAP are not deterministic across runs.

References (preview):
- ESL §14.5.
- van der Maaten & Hinton — *Visualizing Data using t-SNE* (2008).
- McInnes, Healy, Melville — *UMAP* (2018).
