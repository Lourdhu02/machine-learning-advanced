# 05 — Decision Trees

> Goal: understand why a greedy, top-down split on the "best" feature is a viable algorithm at all, and what "best" means.

**Status:** stub.

Planned contents:
- Entropy and Gini impurity, defined and compared. Why both work.
- Information gain = parent impurity − weighted child impurity.
- The greedy split algorithm (CART). Why it's NP-hard to find the optimal tree but the greedy version still works in practice.
- Stopping criteria: max depth, min samples, min impurity decrease.
- Pruning (cost-complexity).
- Diagrams: a 2D decision boundary made of axis-aligned splits (rectangles); a tree drawing of the same splits.
- Mind-map: tree-based family.
- `from_scratch.py`: build a classification tree recursively. Print the rules.
- When it breaks: high-variance / overfits easily, hates rotated decision boundaries, unstable to small data changes — fixes lead directly into module 06 (ensembles).

References (preview):
- Breiman et al. — *Classification and Regression Trees* (1984). The original.
- ESL §9.2.
- StatQuest — *Decision Trees, Clearly Explained* (YouTube).
