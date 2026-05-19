# 10 — Neural Nets: MLP & Backpropagation

> Goal: derive backprop by hand on a 2-layer net using only the chain rule. Implement it in NumPy. Never be afraid of `loss.backward()` again.

**Status:** stub.

Planned contents:
- The MLP: stack of `Linear → activation → Linear → activation → …`. Why depth + nonlinearity = universal approximation.
- Forward pass for a 2-layer net, all shapes annotated.
- Backpropagation as the chain rule applied layer by layer. Full per-layer derivation of:
  - `∂L/∂W_out`, `∂L/∂b_out`
  - `∂L/∂W_hidden`, `∂L/∂b_hidden`
- The "delta" trick: every layer's gradient computed from the next layer's delta.
- Activation choices: sigmoid (legacy), tanh, ReLU (modern default), GELU (Transformers).
- Loss choices: MSE for regression, cross-entropy for classification. Why cross-entropy + softmax has such a clean gradient.
- Diagrams: a labeled 2-layer net with forward and backward arrows; activation functions side by side.
- Mind-map: deep learning architectures family.
- `from_scratch.py`: 2-layer MLP in pure NumPy on a toy classification task. No autodiff.
- When it breaks: vanishing gradients in deep / sigmoid networks (motivates module 11), exploding gradients (gradient clipping), bad init (Xavier / He).

References (preview):
- Rumelhart, Hinton, Williams — *Learning representations by back-propagating errors* (1986). The original.
- Karpathy — *A Recipe for Training Neural Networks* (blog).
- Nielsen — *Neural Networks and Deep Learning* (free book), Ch. 2.
