# 11 — Training Deep Nets (Optimizers, Dropout, BatchNorm)

> Goal: understand the three optimizer families (SGD-with-momentum, RMSProp, Adam) as variations on one update rule, and why dropout / batchnorm aren't hacks but principled regularizers.

**Status:** stub.

Planned contents:
- Plain SGD recap.
- Momentum: `v_t = β v_{t-1} + ∇L`, `w_{t+1} = w_t − η v_t`. EMA of gradients. Why it accelerates in flat directions and damps oscillation.
- RMSProp: divide by EMA of squared gradients. Per-parameter learning rates.
- Adam = momentum + RMSProp + bias correction. Why bias correction matters in the first few steps.
- Learning rate schedules: cosine, warmup, restarts.
- Dropout: randomly mask activations. Equivalent to training an ensemble of subnetworks; equivalent to noise injection.
- Batch normalization: normalize, then learn scale+shift `(γ, β)`. Why it speeds up convergence (smoother loss landscape, not "covariate shift" — the original claim was wrong).
- LayerNorm and why Transformers prefer it.
- Weight decay vs L2 — they are *not* the same in Adam (decoupled weight decay = AdamW).
- Diagrams: SGD vs momentum vs Adam paths on a non-convex 2D loss; dropout mask animation; before/after BN on activation distribution.
- Mind-map: optimization + regularization family.
- `from_scratch.py`: Adam, dropout, batchnorm, all in NumPy. Train module 10's MLP with each and compare.
- When it breaks: Adam can find sharper minima that generalize worse than SGD; BN with small batch sizes breaks; dropout at inference time = disaster.

References (preview):
- Kingma & Ba — *Adam* (2014).
- Ioffe & Szegedy — *Batch Normalization* (2015).
- Loshchilov & Hutter — *Decoupled Weight Decay Regularization* (AdamW, 2017).
- Santurkar et al. — *How Does Batch Normalization Help Optimization?* (2018) — the "covariate shift" debunk.
