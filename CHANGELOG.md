# Changelog

All notable changes to this course are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2024-01-01

### Added

- **00 — Math Foundations**: linear algebra, calculus, probability primer; eigendecomposition and gradient-geometry diagrams.
- **01 — Linear Regression**: OLS closed-form + gradient descent, both derived and implemented; MSE loss surface diagram.
- **02 — Logistic Regression**: sigmoid, log-likelihood, cross-entropy, decision boundary.
- **03 — Regularization**: Ridge, Lasso, ElasticNet; penalty geometry and weight-path diagrams.
- **04 — SVM**: margin → primal → dual → kernel trick; hinge loss, kernel, margin diagrams.
- **05 — Decision Trees**: entropy, Gini, recursive splits; impurity and partition diagrams.
- **06 — Ensembles**: Bagging, Random Forest, AdaBoost, Gradient Boosting; adaboost-weight, bagging-vs-boosting, loss-curve diagrams.
- **07 — Bayes & kNN**: Naive Bayes derivation + kNN baseline; curse-of-dimensionality, k-selection, NB-region diagrams.
- **08 — Clustering**: k-means + EM for GMM; kmeans-iters, gmm-vs-kmeans, em-loglik diagrams.
- **09 — Dimensionality Reduction**: PCA via SVD, t-SNE intuition; pca-geometry, pca-vs-tsne diagrams.
- **Lab A — Classical Shootout**: logistic vs SVM vs RF on one dataset.
- **10 — Neural Nets (MLP)**: forward pass + backprop by chain rule; activation, decision-boundary, training-curve diagrams; numerical gradient check.
- **11 — Training Deep Nets**: SGD, momentum, Adam, dropout, BatchNorm; optimizer-paths, bias-correction, bn-distribution diagrams.
- **12 — CNNs**: convolution as weight-sharing, receptive fields; convolution, filters, receptive-field diagrams.
- **Lab C — MLP vs CNN**: same MNIST, same compute budget.
- **13 — RNNs & LSTMs**: BPTT, vanishing gradients, LSTM gates; unrolled, lstm-cell, vanishing-gradient diagrams.
- **14 — Attention**: scaled dot-product attention from scratch; attention-heatmap, multihead, scale-factor diagrams.
- **15 — Transformers**: block, multi-head, positional encodings; block, architecture, positional diagrams.
- **Lab D — Attention from Scratch**: NumPy attention vs PyTorch.
- **Lab B — PCA vs t-SNE**: three projections side by side.

### Style

- Seven-section module layout: Intuition, Math, Diagram, Mind-map, From scratch, When to use / when it breaks, References.
- Every diagram is a committed matplotlib script + PNG, regenerated with `python diagram_*.py`.
- `from_scratch.py` is NumPy-only for the core loop (no scikit-learn).
- Per-module `requirements.txt` and per-module `venv` pattern documented in `SETUP.md`.

### Project

- MIT license.
- GitHub issue and PR templates.
- GitHub Actions CI that runs every `from_scratch.py` and regenerates every diagram on push/PR.
- `CITATION.cff` for academic citation.
