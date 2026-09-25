# Contributing to Machine Learning, From Scratch

Thanks for your interest. This course is built one module at a time — each module is self-contained (its own `README.md`, `from_scratch.py`, diagrams, `requirements.txt`). The goal is always the same: derive the math, explain the intuition, implement in NumPy, draw the diagram.

## How to contribute

**Before you start:** open an issue describing what you want to add or fix. For significant changes (new module, rewrite of a module), discuss first — the seven-section layout is intentional.

### Adding a module

1. Create `course/XX-name/` with the standard layout:
   - `README.md` — the seven sections: Intuition, Math, Diagram, Mind-map, From scratch, When to use / when it breaks, References.
   - `from_scratch.py` — NumPy implementation, no scikit-learn in the core loop. Include a gradient/numeric check where applicable.
   - `diagram_*.py` + `diagram_*.png` — matplotlib scripts that generate the labeled diagrams committed alongside.
   - `requirements.txt` — pinned to `numpy>=1.26`, `matplotlib>=3.8`, plus anything else the module needs.
2. Keep the module self-contained — don't pull in dependencies that earlier modules don't need.
3. Run `python from_scratch.py` from the module folder and confirm the printed checks pass (`OK`).
4. Regenerate any diagram with `python diagram_*.py` before committing the `.png`.

### Fixing a typo / improving a derivation

Small doc fixes don't need an issue. Just open a PR.

### Code style

- PEP 8, `from __future__ import annotations`, type hints on public functions.
- NumPy docstrings on classes and public functions.
- No scikit-learn in `from_scratch.py` — this is the whole point.

### Pull requests

- One module or one fix per PR.
- PR title: `feat: `, `fix: `, `docs: `, `chore: ` — see commit history for examples.
- Describe what changed and why. Link the issue if there is one.

## What we're looking for

- Correctness of the math (derivations reviewed against Bishop / ESL / Goodfellow).
- Clear diagrams that actually help understanding.
- Implementations that run and pass their own gradient checks.

## License

By contributing you agree your contribution is licensed under the MIT License (see `LICENSE`).
