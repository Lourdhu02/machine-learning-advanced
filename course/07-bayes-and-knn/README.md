# 07 — Naive Bayes & k-Nearest Neighbours

> Goal: two baseline classifiers with almost no parameters — one parametric (NB), one non-parametric (kNN) — and what each one tells you about your problem.

**Status:** stub.

Planned contents:
- Bayes' rule applied to classification: `P(y | x) ∝ P(x | y) P(y)`.
- The "naive" assumption: `P(x | y) = Π P(xⱼ | y)`. Why this awful-looking assumption still produces shockingly good text classifiers.
- Variants: Gaussian NB (continuous features), Multinomial NB (counts), Bernoulli NB (binary).
- kNN: predict by the majority vote of the `k` closest training points. No training, all work at inference time.
- Distance metrics: Euclidean, Manhattan, cosine. When each makes sense.
- Diagrams: NB decision regions; kNN Voronoi diagram; effect of `k` on the boundary.
- Mind-map: non-parametric methods + probabilistic baselines.
- `from_scratch.py`: Gaussian NB + kNN, both ~30 lines.
- When it breaks: NB hates correlated features; kNN dies in high dimensions (curse of dimensionality) and on large training sets (no indexing).

References (preview):
- Bishop §4.2.4 (Gaussian NB), §8.2.
- ESL §13.3 (kNN).
