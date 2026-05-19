# 02 — Logistic Regression

> Goal: derive binary cross-entropy as the MLE under a Bernoulli model, and see why "fit a line then squash it" is the right move for classification.

**Status:** stub. Will be built out in order.

Planned contents:
- Sigmoid as the inverse of the log-odds (logit), derived from first principles.
- Full MLE derivation of binary cross-entropy.
- Gradient of the log-loss — the cleanest gradient derivation in ML once you've seen the MSE one in module 01.
- Diagrams: decision boundary on 2D data, sigmoid vs hard threshold, log-loss surface.
- Mind-map: position inside the linear-models family (links to module 01, 03, 04).
- `from_scratch.py`: NumPy logistic regression with gradient descent, plus a Newton-Raphson alternative (IRLS).
- When it breaks: non-separable classes, class imbalance, perfect separation (weights blow up — fix with regularization, module 03).

References (preview):
- Bishop §4.3.
- Andrew Ng — CS229 Lecture 3 notes (best closed-form derivation).
