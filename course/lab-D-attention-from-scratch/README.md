# Lab D — Attention From Scratch vs PyTorch

> Run after module 15. Your NumPy implementation vs the production version, on a tiny sequence task.

**Status:** stub.

Planned contents:
- Implement multi-head attention in NumPy (you already did this in module 14).
- Implement the same thing using `torch.nn.MultiheadAttention`.
- Feed both the exact same input (deterministic seed) and verify outputs match to within `1e-5`.
- Train a tiny 1-block Transformer on a toy task (e.g., copy task, reverse-string task, or character-level prediction) using each implementation. Compare speed.
- Stress-test: gradually increase sequence length. Watch your O(n²) NumPy attention slow to a crawl while PyTorch's optimized kernel stays fast.

What you'll learn:
- The math you derived in module 14 is *exactly* what runs inside production attention; the only difference is engineering (fused kernels, FlashAttention, memory layout). Closing this loop builds the confidence to read any attention-related paper.
