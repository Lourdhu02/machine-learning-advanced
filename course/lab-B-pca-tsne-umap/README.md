# Lab B — PCA vs t-SNE vs UMAP

> Run after module 09. Three projections of the same dataset, side by side.

**Status:** stub.

Planned contents:
- Two datasets: digits (MNIST 8×8 subset from sklearn) and the Swiss roll (synthetic non-linear manifold).
- 2D projection from each method, plotted in a 2×3 grid.
- Quantitative comparison: trustworthiness, neighbourhood preservation, runtime.
- Conclusion: PCA is fast and globally consistent; t-SNE preserves local neighbourhoods but distorts distances; UMAP sits in between and is faster than t-SNE.
- Demonstrates the *t-SNE distances are not meaningful* lesson visually.

What you'll learn:
- Pick the right projection for the question you're asking: PCA for "what's the dominant axis of variance", t-SNE/UMAP for "do clusters separate locally".
