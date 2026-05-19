"""Two sanity checks that exercise the two tools we'll reuse everywhere:
chain rule (numerical vs analytical gradient) and eigendecomposition (Q Lambda Q^-1).

Run: python from_scratch.py
"""

import numpy as np


def chain_rule_check() -> bool:
    """f(x) = sin(x^2). df/dx = cos(x^2) * 2x. Compare against finite difference."""
    x = 1.7
    h = 1e-6
    analytical = np.cos(x**2) * 2 * x
    numerical = (np.sin((x + h) ** 2) - np.sin((x - h) ** 2)) / (2 * h)
    err = abs(analytical - numerical)
    print(f"chain-rule check  | analytical={analytical:.8f}  numerical={numerical:.8f}  |diff|={err:.2e}")
    return err < 1e-6


def eigendecomp_check() -> bool:
    """A = Q Lambda Q^-1 for a random symmetric matrix (real eigenvalues guaranteed)."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4))
    A = M + M.T  # symmetric
    eigvals, Q = np.linalg.eig(A)
    A_reconstructed = Q @ np.diag(eigvals) @ np.linalg.inv(Q)
    err = float(np.max(np.abs(A - A_reconstructed)))
    print(f"eigendecomp check | shape={A.shape}  max|A - Q Lambda Q^-1| = {err:.2e}")
    return err < 1e-8


if __name__ == "__main__":
    ok = chain_rule_check() and eigendecomp_check()
    print("OK" if ok else "FAIL")
