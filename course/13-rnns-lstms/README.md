# 13 — RNNs & LSTMs

> Goal: derive backpropagation-through-time on a vanilla RNN, see exactly *where* gradients vanish, and understand the LSTM as a fix for that exact problem.

**Status:** stub.

Planned contents:
- Vanilla RNN: `h_t = tanh(W_x x_t + W_h h_{t-1} + b)`. One set of weights shared across all timesteps.
- BPTT: unroll the recurrence, apply chain rule across time. Full derivation.
- Vanishing / exploding gradients: the product of `W_h` Jacobians across timesteps either shrinks to zero or explodes. Why tanh makes it worse.
- LSTM cell:
  - Cell state `c_t` is the "memory highway" — additive updates instead of multiplicative.
  - Forget gate, input gate, output gate, each with their own sigmoid. Full math.
  - Why the additive cell update makes gradients flow.
- GRU as a simpler LSTM with two gates instead of three.
- Bidirectional and stacked variants.
- Diagrams: unrolled RNN with BPTT arrows; LSTM cell with labeled gates; gradient magnitude vs depth in vanilla RNN vs LSTM.
- Mind-map: sequence models family — leads into attention (module 14).
- `from_scratch.py`: vanilla RNN + LSTM cell, both from scratch in NumPy on a character-level sequence task (predict next character in a short string).
- When it breaks: long-range dependencies are still hard; can't parallelize across timesteps (motivates the Transformer); attention solves both.

References (preview):
- Hochreiter & Schmidhuber — *Long Short-Term Memory* (1997). Famously brutal to read; read the modern explainers first.
- Olah — *Understanding LSTM Networks* (blog, 2015). The standard explainer.
- Karpathy — *The Unreasonable Effectiveness of Recurrent Neural Networks* (blog, 2015).
