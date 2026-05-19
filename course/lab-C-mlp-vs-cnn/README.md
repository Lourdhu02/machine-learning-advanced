# Lab C — MLP vs CNN on MNIST

> Run after module 12. Same data, same compute budget, very different parameter efficiency.

**Status:** stub.

Planned contents:
- Train an MLP and a CNN on MNIST to roughly the same compute budget (e.g., same wall-clock or same number of forward passes).
- Compare: accuracy, parameter count, sample efficiency (accuracy vs % of training data used).
- Visualize the CNN's first-layer filters as learned edge detectors.
- Show the MLP needing a lot more parameters to reach the same accuracy.

What you'll learn:
- Inductive bias is real. Encoding the right prior (translation equivariance via convolution) lets a smaller model with less data match a much larger one without that prior. This is the same lesson that scales up to "Transformers + lots of data > CNNs + less data" on vision.
