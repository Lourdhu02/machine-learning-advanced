# 04 — Support Vector Machines

> Goal: derive the max-margin classifier from scratch, get to the dual via Lagrange multipliers, and see the kernel trick emerge naturally.

**Status:** stub.

Planned contents:
- Margin definition: distance from a point to a hyperplane.
- Primal optimization: minimize `‖w‖²` subject to `yᵢ(wᵀxᵢ + b) ≥ 1`. Why this is the *right* objective.
- Lagrangian → dual problem (full derivation). Why the dual depends only on `xᵢᵀxⱼ`.
- Kernel trick: replace `xᵢᵀxⱼ` with `K(xᵢ, xⱼ)`. RBF, polynomial, why this works (Mercer's condition).
- Soft margin with slack variables.
- Diagrams: hyperplane + margin + support vectors; the same data made separable by an RBF kernel.
- Mind-map: kernel methods family.
- `from_scratch.py`: SMO algorithm (simplified) for the dual; linear and RBF kernels.
- When it breaks: large `n` (kernel matrix is `n × n`), needs feature scaling, multi-class needs one-vs-rest.

References (preview):
- Bishop §7.1.
- Andrew Ng — CS229 Lecture 6-8 notes.
- Burges — *A Tutorial on Support Vector Machines for Pattern Recognition* (1998).
