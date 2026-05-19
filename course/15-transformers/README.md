# 15 — Transformers

> Goal: assemble the Transformer block from module 14's attention plus residual connections, layer norm, and a feedforward sublayer. Then understand how LLMs are just this block stacked 50–100 times.

**Status:** stub.

Planned contents:
- The Transformer block, derived:
  - `x = x + Attention(LayerNorm(x))`   (pre-norm, modern variant)
  - `x = x + FFN(LayerNorm(x))`
- Residual connection: why it lets gradients flow through 100+ layers.
- LayerNorm vs BatchNorm: why Transformers prefer LayerNorm (independence across sequence positions).
- Feedforward sublayer = `Linear → GELU → Linear`, hidden width usually `4 × d_model`. Why this is the bulk of the parameters.
- Positional encoding:
  - Sinusoidal (original paper). Geometric intuition.
  - Learned absolute.
  - Rotary (RoPE) — used in LLaMA, Mistral. Why it composes nicely with attention.
- Decoder-only vs encoder-decoder vs encoder-only. Which family each modern model belongs to.
- Scaling laws (one paragraph): loss vs (params, data, compute) follows a power law. Implications for LLM design.
- Diagrams: Transformer block schematic; residual stream; positional encoding patterns visualized; attention patterns at different layers.
- Mind-map: modern deep learning architectures (CNN / RNN / Transformer / mixture).
- `from_scratch.py`: a full Transformer block in NumPy, then a tiny decoder-only Transformer (~3 layers, ~100k params) that learns to predict the next character of a string. No PyTorch in the core loop.
- When it breaks: quadratic attention cost in sequence length; needs lots of data to shine; small Transformers underperform CNNs/RNNs on small datasets.

References (preview):
- Vaswani et al. — *Attention is All You Need* (2017).
- Karpathy — *Let's build GPT: from scratch, in code, spelled out* (YouTube). Best resource on this topic, period.
- Su et al. — *RoFormer* (RoPE, 2021).
- Kaplan et al. — *Scaling Laws for Neural Language Models* (2020).
