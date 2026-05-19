# 12 — Convolutional Neural Networks

> Goal: see convolution as a constrained, weight-sharing version of a linear layer — and understand why that bias makes vision tractable.

**Status:** stub.

Planned contents:
- 2D convolution as cross-correlation: dot products between a kernel and image patches.
- Stride, padding, dilation — the three knobs.
- Why weight sharing? Translation equivariance: shift the input, the output shifts the same way.
- Pooling (max, average): translation invariance + downsampling. Why modern networks (ResNet, ConvNeXt) often skip explicit pooling.
- Receptive field: which pixels in the input affect a given neuron. How it grows with depth.
- Channels: parallel feature detectors.
- 1×1 convolutions = per-pixel linear projection across channels. Surprisingly useful.
- Modern blocks: ResNet residual, inverted bottleneck (MobileNet), patch-embedding (ViT) as just a strided conv.
- Diagrams: a 3×3 kernel sliding over an image; receptive field growing layer by layer; learned first-layer filters that look like edge detectors.
- Mind-map: CNN family (LeNet → AlexNet → VGG → ResNet → ConvNeXt).
- `from_scratch.py`: forward pass of a 2-conv-layer CNN on a tiny image batch. Backward via `im2col` math.
- When it breaks: not great for non-grid data (use GNNs); ViTs catch up at scale; convolutions are *equivariant* but not *rotation*-invariant — augment.

References (preview):
- LeCun et al. — *Gradient-Based Learning Applied to Document Recognition* (LeNet, 1998).
- He et al. — *Deep Residual Learning* (ResNet, 2015).
- Stanford CS231n notes (still the gold standard).
