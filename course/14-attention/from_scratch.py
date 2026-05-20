"""Scaled dot-product attention and multi-head attention from scratch in NumPy,
verified against PyTorch.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# -----------------------------------------------------------------------------
# Scaled dot-product attention (the five lines)
# -----------------------------------------------------------------------------


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q: (..., L_q, d_k), K: (..., L_k, d_k), V: (..., L_k, d_v).
    Optional mask: (L_q, L_k) with True = mask out.
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    attn = softmax(scores, axis=-1)
    out = attn @ V
    return out, attn


# -----------------------------------------------------------------------------
# Multi-head attention
# -----------------------------------------------------------------------------


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = rng.standard_normal((d_model, d_model)) * scale
        self.W_K = rng.standard_normal((d_model, d_model)) * scale
        self.W_V = rng.standard_normal((d_model, d_model)) * scale
        self.W_O = rng.standard_normal((d_model, d_model)) * scale

    def _split_heads(self, X):
        # (L, d_model) -> (n_heads, L, d_k)
        L = X.shape[0]
        return X.reshape(L, self.n_heads, self.d_k).transpose(1, 0, 2)

    def _merge_heads(self, X):
        # (n_heads, L, d_k) -> (L, d_model)
        L = X.shape[1]
        return X.transpose(1, 0, 2).reshape(L, self.d_model)

    def forward(self, X, mask=None):
        Q = self._split_heads(X @ self.W_Q)
        K = self._split_heads(X @ self.W_K)
        V = self._split_heads(X @ self.W_V)
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        out = self._merge_heads(out)
        return out @ self.W_O, attn


def make_causal_mask(L: int) -> np.ndarray:
    """Upper-triangular (excluding diagonal) -> True means mask out."""
    return np.triu(np.ones((L, L), dtype=bool), k=1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    np.set_printoptions(precision=3, suppress=True)

    L, d = 6, 8
    rng = np.random.default_rng(0)
    X = rng.standard_normal((L, d)) * 0.5

    print("=" * 60)
    print("Self-attention on a 6-token sequence")
    print("=" * 60)
    out, attn = scaled_dot_product_attention(X, X, X)
    print(f"  output shape : {out.shape}")
    print(f"  attention matrix (rows sum to 1):")
    print(attn)

    print()
    print("=" * 60)
    print("Causal masking")
    print("=" * 60)
    mask = make_causal_mask(L)
    out, attn_masked = scaled_dot_product_attention(X, X, X, mask=mask)
    print("  attention matrix (upper triangle should be 0):")
    print(attn_masked)
    upper = np.triu(attn_masked, k=1)
    assert np.max(np.abs(upper)) < 1e-9, "causal mask not working"
    print("  OK -- upper triangle is exactly zero")

    print()
    print("=" * 60)
    print("Multi-head attention (d_model=8, n_heads=2)")
    print("=" * 60)
    mha = MultiHeadAttention(d_model=8, n_heads=2, seed=0)
    out, attn_per_head = mha.forward(X)
    print(f"  output shape : {out.shape}")
    print(f"  per-head attention shape: {attn_per_head.shape}")

    # Verify against PyTorch
    try:
        import torch
        import torch.nn.functional as F
        Q = torch.tensor(X, dtype=torch.float32)
        K = torch.tensor(X, dtype=torch.float32)
        V = torch.tensor(X, dtype=torch.float32)
        torch_out = F.scaled_dot_product_attention(Q, K, V)
        np_out, _ = scaled_dot_product_attention(X, X, X)
        err = float(np.max(np.abs(np_out - torch_out.numpy())))
        print()
        print("=" * 60)
        print("Verification against PyTorch")
        print("=" * 60)
        print(f"  max |numpy - torch.scaled_dot_product_attention| = {err:.2e}")
        assert err < 1e-5, "NumPy attention disagrees with PyTorch"
        print("  OK")
    except ImportError:
        print("  (torch not installed -- skipping verification)")


if __name__ == "__main__":
    main()
