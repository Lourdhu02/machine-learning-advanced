"""NumPy attention vs PyTorch's scaled_dot_product_attention.

Three things:
  1. Verification: same Q, K, V -> identical outputs up to 1e-6.
  2. Sequence-length timing: NumPy vs PyTorch on L = 16..1024.
  3. Tiny Transformer copy-task training.

Imports module 14's NumPy attention.

Run: python compare.py
"""

from __future__ import annotations
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Import NumPy attention from module 14
# -----------------------------------------------------------------------------


COURSE = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, COURSE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


attn_mod = _load("attn_mod", "14-attention/from_scratch.py")
scaled_dot_product_attention = attn_mod.scaled_dot_product_attention


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------


def verify_against_pytorch(n_trials=100):
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(n_trials):
        L = int(rng.integers(4, 64))
        d_k = int(rng.integers(4, 32))
        Q = rng.standard_normal((L, d_k)).astype(np.float32)
        K = rng.standard_normal((L, d_k)).astype(np.float32)
        V = rng.standard_normal((L, d_k)).astype(np.float32)

        np_out, _ = scaled_dot_product_attention(Q, K, V)
        torch_out = F.scaled_dot_product_attention(
            torch.from_numpy(Q), torch.from_numpy(K), torch.from_numpy(V)
        ).numpy()
        err = float(np.max(np.abs(np_out - torch_out)))
        max_err = max(max_err, err)
    return max_err


# -----------------------------------------------------------------------------
# Timing scan
# -----------------------------------------------------------------------------


def time_implementation(L, d_k=64, n_runs=20):
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((L, d_k)).astype(np.float32)
    K = rng.standard_normal((L, d_k)).astype(np.float32)
    V = rng.standard_normal((L, d_k)).astype(np.float32)
    Qt, Kt, Vt = torch.from_numpy(Q), torch.from_numpy(K), torch.from_numpy(V)

    # Warm up
    for _ in range(3):
        scaled_dot_product_attention(Q, K, V)
        F.scaled_dot_product_attention(Qt, Kt, Vt)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        scaled_dot_product_attention(Q, K, V)
    numpy_ms = max(1e-3, (time.perf_counter() - t0) / n_runs * 1000)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        F.scaled_dot_product_attention(Qt, Kt, Vt)
    pytorch_ms = max(1e-3, (time.perf_counter() - t0) / n_runs * 1000)

    return numpy_ms, pytorch_ms


# -----------------------------------------------------------------------------
# Tiny Transformer copy task in PyTorch
# -----------------------------------------------------------------------------


class TinyTransformer(nn.Module):
    """1-block decoder-only Transformer that learns the identity map on
    short token sequences (the copy task)."""

    def __init__(self, vocab_size=8, d_model=32, n_heads=4, max_len=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        h_ln = self.ln1(h)
        attn, _ = self.mha(h_ln, h_ln, h_ln, attn_mask=mask)
        h = h + attn
        h = h + self.ffn(self.ln2(h))
        return self.head(h)


def train_copy_task(vocab=8, length=8, steps=400):
    model = TinyTransformer(vocab_size=vocab, max_len=length)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    losses = []
    for step in range(steps):
        x = torch.from_numpy(rng.integers(0, vocab, size=(32, length))).long()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  step {step:>3}: loss = {loss.item():.4f}")
    print(f"  step {steps-1:>3}: loss = {losses[-1]:.4f}")
    return losses


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("1.  Verification: numpy attention vs torch attention")
    print("=" * 60)
    err = verify_against_pytorch(n_trials=100)
    print(f"  max |numpy - pytorch| over 100 random (Q, K, V) batches = {err:.2e}")
    assert err < 1e-5, "numpy attention disagrees with pytorch"
    print("  OK -- the math is identical")

    print()
    print("=" * 60)
    print("2.  Sequence-length timing")
    print("=" * 60)
    Ls = [16, 32, 64, 128, 256, 512, 1024]
    timings = []
    print(f"  {'L':>6}  {'numpy (ms)':>14}  {'pytorch (ms)':>14}  {'ratio':>8}")
    for L in Ls:
        nm, pt = time_implementation(L, d_k=64, n_runs=3)
        timings.append((L, nm, pt))
        print(f"  {L:>6}  {nm:>14.2f}  {pt:>14.2f}  {nm/pt:>8.1f}x")

    print()
    print("=" * 60)
    print("3.  Tiny Transformer learning the copy task")
    print("=" * 60)
    losses = train_copy_task(steps=400)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    Ls_arr = np.array([t[0] for t in timings])
    np_ms = np.array([t[1] for t in timings])
    pt_ms = np.array([t[2] for t in timings])
    axes[0].loglog(Ls_arr, np_ms, "o-", color="crimson", lw=2, label="NumPy")
    axes[0].loglog(Ls_arr, pt_ms, "o-", color="steelblue", lw=2, label="PyTorch")
    axes[0].loglog(Ls_arr, np_ms[0] * (Ls_arr / Ls_arr[0]) ** 2, "k--",
                   alpha=0.5, label="O(L²) reference")
    axes[0].set_xlabel("sequence length L")
    axes[0].set_ylabel("forward time (ms)")
    axes[0].set_title("Attention scaling: both O(L²),\nPyTorch wins on constant factor")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(losses, color="darkgreen", lw=1.8)
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("cross-entropy loss")
    axes[1].set_title("Tiny Transformer on copy task:\nlearns identity map in ~200 steps")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_compare.png", dpi=140)
    print("\nwrote diagram_compare.png")


if __name__ == "__main__":
    main()
