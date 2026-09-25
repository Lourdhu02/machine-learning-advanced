# Advanced ML — Roadmap (part II)

This file tracks topics beyond the current 16 modules. The modules that exist today are **done** — each has a README, from-scratch NumPy implementation, diagrams, and references. The entries below are **planned**, not placeholder stubs.

## Already covered (modules 07–15)

| Topic | Module | Status |
|---|---|---|
| Neural nets (MLP) | 10 | ✅ Done |
| Training deep nets (optimizers, dropout, BatchNorm) | 11 | ✅ Done |
| CNNs | 12 | ✅ Done |
| RNNs & LSTMs | 13 | ✅ Done |
| Attention | 14 | ✅ Done |
| Transformers | 15 | ✅ Done |
| NLP basics (language model intuition, tokenization) | 15 (transformer context) | ✅ Partial |
| CV basics (convolution, receptive fields, filters) | 12 | ✅ Done |

## Planned — part II modules

These are listed as a roadmap. Each will follow the same seven-section layout when built.

### NLP
1. **Tokenization & embeddings** — word2vec (CBOW / skip-gram), GloVe, subword tokenization (BPE).
2. **Sequence models for NLP** — Seq2Seq, attention-over-RNN, the transformer as a sequence model.
3. **Pretrained language models** — how BERT / GPT work at a structural level; masked vs causal LM; fine-tuning vs prompting.

### Computer Vision
4. **Modern CNN architectures** — ResNet, EfficientNet; skip connections, depthwise separable conv.
5. **Object detection** — two-stage (R-CNN family) vs one-stage (YOLO family) at a structural level.
6. **Segmentation** — semantic vs instance; U-Net structure; mask heads.

### Generative models
7. **Autoencoders & VAEs** — reconstruction loss, KL term, the evidence lower bound.
8. **Generative adversarial networks** — minimax game, mode collapse, the discriminator as a learned loss.
9. **Diffusion models** — forward process, reverse process, the noise-prediction objective.

### Reinforcement learning (intro)
10. **MDPs and value functions** — Bellman equation, value iteration, policy iteration.
11. **Q-learning and policy gradients** — DQN, REINFORCE, the connection to supervised learning.

## Contributing

If you want to build one of these, open an issue first. The same standards apply: derive the math, implement in NumPy (or PyTorch where the math demands it), draw the diagram, cite the canonical source.
