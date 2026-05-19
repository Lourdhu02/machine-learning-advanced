# 06 — Ensembles (Bagging, RF, AdaBoost, Gradient Boosting)

> Goal: see why combining many weak learners beats one strong learner, and how *boosting* differs from *bagging* in one sentence.

**Status:** stub.

Planned contents:
- Bias-variance recap. Bagging reduces variance; boosting reduces bias.
- Bagging = bootstrap + average. Why averaging i.i.d. estimators shrinks variance by `1/n`.
- Random Forest = bagging + random feature subsets at each split. Why decorrelating trees helps.
- AdaBoost: derived as forward-stagewise additive modeling with exponential loss. Why misclassified samples get up-weighted.
- Gradient Boosting: AdaBoost generalized to any differentiable loss. Each new tree fits the negative gradient of the loss w.r.t. previous predictions.
- XGBoost / LightGBM intuition: second-order Taylor expansion, histogram-based splits, regularized leaves.
- Diagrams: variance reduction visualized; AdaBoost weights updating; GBM residual fitting step by step.
- Mind-map: tree-based family with bagging and boosting branches.
- `from_scratch.py`: minimal Random Forest + minimal Gradient Boosting on top of module 05's tree.
- When it breaks: less interpretable than a single tree; boosting overfits without proper learning rate / early stopping; bagging doesn't help on stable models (linear regression already has low variance).

References (preview):
- Breiman — *Random Forests* (2001).
- Friedman — *Greedy Function Approximation: A Gradient Boosting Machine* (2001).
- Chen & Guestrin — *XGBoost: A Scalable Tree Boosting System* (2016).
