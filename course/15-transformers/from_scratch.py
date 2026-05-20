"""Transformer block (pre-norm) in NumPy, reusing module 14's attention.

Run: python from_scratch.py
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np


# Import attention from module 14
ATTN_PATH = Path(__file__).resolve().parent.parent / "14-attention" / "from_scratch.py"
spec = importlib.util.spec_from_file_location("attn_mod", ATTN_PATH)
attn_mod = importlib.util.module_from_spec(spec)
sys.modules["attn_mod"] = attn_mod
spec.loader.exec_module(attn_mod)

MultiHeadAttention = attn_mod.MultiHeadAttention
scaled_dot_product_attention = attn_mod.scaled_dot_product_attention
make_causal_mask = attn_mod.make_causal_mask


# -----------------------------------------------------------------------------
# LayerNorm
# -----------------------------------------------------------------------------


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """Per-token normalization across the feature axis."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta


# -----------------------------------------------------------------------------
# Feedforward sublayer
# -----------------------------------------------------------------------------


def gelu(x: np.ndarray) -> np.ndarray:
    """Smooth approximation to ReLU. d/dx (x · Φ(x)) via the tanh approximation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                     (x + 0.044715 * x ** 3)))


class FeedForward:
    def __init__(self, d_model: int, d_ff: int | None = None, seed: int = 0):
        rng = np.random.default_rng(seed)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        scale_in = np.sqrt(2.0 / d_model)
        scale_out = np.sqrt(2.0 / d_ff)
        self.W1 = rng.standard_normal((d_model, d_ff)) * scale_in
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_ff, d_model)) * scale_out
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


# -----------------------------------------------------------------------------
# Transformer block (pre-norm)
# -----------------------------------------------------------------------------


class TransformerBlock:
    """Pre-norm:  x = x + MHA(LN(x));  x = x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None,
                 seed: int = 0):
        self.d_model = d_model
        self.mha = MultiHeadAttention(d_model, n_heads, seed=seed)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 1)
        # Two LayerNorm γ, β pairs per block
        self.g1 = np.ones(d_model)
        self.b1 = np.zeros(d_model)
        self.g2 = np.ones(d_model)
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray, mask=None) -> np.ndarray:
        # Sublayer 1: attention
        attn_out, _ = self.mha.forward(layer_norm(x, self.g1, self.b1), mask=mask)
        x = x + attn_out
        # Sublayer 2: FFN
        x = x + self.ffn.forward(layer_norm(x, self.g2, self.b2))
        return x


# -----------------------------------------------------------------------------
# Sinusoidal positional encoding
# -----------------------------------------------------------------------------


def sinusoidal_position_encoding(L: int, d_model: int) -> np.ndarray:
    """PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
       PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
    """
    pos = np.arange(L)[:, None]
    i = np.arange(d_model // 2)[None, :]
    angles = pos / np.power(10000, 2 * i / d_model)
    pe = np.zeros((L, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    np.set_printoptions(precision=3, suppress=True)
    rng = np.random.default_rng(0)

    L = 8
    d_model = 32
    n_heads = 4

    # Token embeddings (random for demo)
    token_emb = rng.standard_normal((L, d_model)) * 0.5
    pe = sinusoidal_position_encoding(L, d_model)
    x = token_emb + pe

    print("=" * 60)
    print(f"2-block decoder-only Transformer  (d_model={d_model}, n_heads={n_heads})")
    print("=" * 60)
    print(f"  input shape : {x.shape}")
    print(f"  positional encoding contributes {np.linalg.norm(pe):.2f} (L2 norm of PE)")

    mask = make_causal_mask(L)
    block1 = TransformerBlock(d_model, n_heads, seed=0)
    block2 = TransformerBlock(d_model, n_heads, seed=10)

    h = block1.forward(x, mask=mask)
    h = block2.forward(h, mask=mask)

    print(f"  output shape after 2 blocks: {h.shape}")
    print(f"  output mean ± std         : {h.mean():.3f} ± {h.std():.3f}")
    print(f"  output norm per token     : {np.linalg.norm(h, axis=1).round(2)}")

    # Determinism check
    h_again = block2.forward(block1.forward(x, mask=mask), mask=mask)
    assert np.allclose(h, h_again), "block is not deterministic"
    print("  forward pass is deterministic (no dropout, no random ops)")

    # LayerNorm sanity: after LN, mean ≈ 0, std ≈ 1
    ln_out = layer_norm(x, np.ones(d_model), np.zeros(d_model))
    print(f"\n  LN sanity:  per-token mean = {ln_out.mean(axis=-1).round(3)}")
    print(f"              per-token std  = {ln_out.std(axis=-1).round(3)}")

    print("\nOK")


if __name__ == "__main__":
    main()
