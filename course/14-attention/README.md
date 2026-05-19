# 14 — Attention

> Goal: derive scaled dot-product attention from scratch and understand each term — Q, K, V, the scale factor, the softmax — as a specific design choice with a specific failure mode it fixes.

**Status:** stub.

Planned contents:
- Motivation: RNNs can't parallelize across time and struggle with long-range dependencies. We need a way to mix information across positions in one shot.
- Soft-lookup analogy: imagine a key-value dictionary where every lookup returns a *weighted blend* of all values, weights coming from how well your query matches each key.
- Scaled dot-product attention, derived term by term:
  - `Q = X W_Q`, `K = X W_K`, `V = X W_V`. Why separate projections.
  - Score `= Q Kᵀ`. Why dot product as similarity.
  - Scale by `√d_k`. Where this comes from (variance of a dot product of two unit Gaussians grows with `d_k` → softmax saturates without scaling).
  - Softmax → attention weights.
  - Weighted sum of `V`.
- Multi-head: do this `h` times in parallel with smaller `d_k`, concatenate. Why this beats a single big head.
- Masking: causal mask for language modeling; padding mask for variable-length batches.
- Cross-attention vs self-attention.
- Diagrams: attention-weight heatmap on a toy sentence; multi-head as parallel projections; mask matrices visualized.
- Mind-map: attention family.
- `from_scratch.py`: scaled dot-product + multi-head attention in NumPy, verified against a tiny PyTorch reference.
- When it breaks: quadratic memory in sequence length (motivates FlashAttention, sliding window, linear attention variants).

References (preview):
- Vaswani et al. — *Attention is All You Need* (2017). Read this carefully — it's short and well-written.
- Bahdanau et al. — *Neural Machine Translation by Jointly Learning to Align and Translate* (2014). Attention's first appearance.
- Alammar — *The Illustrated Transformer* (blog).
