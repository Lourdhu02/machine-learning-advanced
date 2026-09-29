# Machine Learning, From Scratch — course site

This is the online home of the **Machine Learning, From Scratch** course. Every module below is self-contained: derive the math, run the NumPy code, look at the diagram.

---

## The course

A theory-first course. You learn ML by **deriving the math**, **drawing the diagram**, and **coding the algorithm from scratch in NumPy** — not by gluing together scikit-learn.

Audience: comfortable Python, familiar with gradient descent and train/test, wants to *understand* every algorithm well enough to read papers and explain it on a whiteboard.

This is the theory counterpart to [gen-ai](https://github.com/Lourdhu02/gen-ai).

---

## What you'll know by the end

- Derive OLS, logistic regression, SVM, and backprop on paper from first principles.
- Explain why Ridge beats OLS on collinear data, why AdaBoost works, why the Transformer needs the scaling factor.
- Implement 16 algorithms from scratch in NumPy (no scikit-learn in the core loop) and 4 labs comparing them head-to-head.
- Read a paper like Vaswani et al. 2017 or Hochreiter & Schmidhuber 1997 and map it onto the code.
- Know when each algorithm works, when it breaks, and what to try next.

---

## Who this is for / not for

**For:** people who want the derivation, not just the API call. Students, self-taught engineers, researchers who need the foundations solid.

**Not for:** people who want a quick scikit-learn tutorial, or a production deployment guide. Those exist elsewhere. This course is about understanding.

---

## How to use this course

For each module:

1. Read the **Intuition** — one paragraph, no jargon.
2. Derive the **Math** on paper. Don't skip the intermediate steps.
3. Run the **From scratch** code: `python from_scratch.py`. Confirm the printed checks pass (`OK`).
4. Look at the **Diagram**. Regenerate it with `python diagram_*.py` if you changed anything.
5. Read **When to use / when it breaks** — this is the honest version.
6. Skim the **References** — the canonical papers and books.
7. (New) Read the **Advanced** section for deeper insight and connections.

Move to the next module. Do the labs once the relevant modules are done.

---

## Roadmap

| # | Module | Core idea | Math |
|---|---|---|---|
| 00 | [Math Foundations](course/00-math-foundations/) | Just-enough linear algebra, calculus, probability | Full |
| 01 | [Linear Regression](course/01-linear-regression/) | OLS closed-form and gradient descent, both derived | Full |
| 02 | [Logistic Regression](course/02-logistic-regression/) | Sigmoid, log-likelihood, cross-entropy | Full |
| 03 | [Regularization](course/03-regularization/) | Ridge, Lasso, ElasticNet — geometry of the penalty | Light |
| 04 | [SVM](course/04-svm/) | Margin → primal → dual → kernel trick | Full |
| 05 | [Decision Trees](course/05-decision-trees/) | Entropy, Gini, recursive splits | Mixed |
| 06 | [Ensembles](course/06-ensembles/) | Bagging, Random Forest, AdaBoost, Gradient Boosting | Mixed |
| — | [Lab A: Classical Shootout](course/lab-A-classical-shootout/) | Logistic vs SVM vs RF on one dataset | — |
| 07 | [Bayes & kNN](course/07-bayes-and-knn/) | Naive Bayes derivation + kNN baseline | Light |
| 08 | [Clustering](course/08-clustering/) | k-means + EM for GMM | Full |
| 09 | [Dim. Reduction](course/09-dim-reduction/) | PCA via SVD, t-SNE / UMAP intuition | Mixed |
| — | [Lab B: PCA vs t-SNE vs UMAP](course/lab-B-pca-tsne-umap/) | Three projections, side by side | — |
| 10 | [Neural Nets: MLP](course/10-neural-nets-mlp/) | Forward pass + backprop derived by chain rule | Full |
| 11 | [Training Deep Nets](course/11-training-deep-nets/) | SGD, momentum, Adam, dropout, batchnorm | Mixed |
| 12 | [CNNs](course/12-cnns/) | Convolution as weight-sharing, receptive fields | Mixed |
| — | [Lab C: MLP vs CNN](course/lab-C-mlp-vs-cnn/) | Same MNIST, same compute budget | — |
| 13 | [RNNs & LSTMs](course/13-rnns-lstms/) | BPTT derived, vanishing gradients, LSTM gates | Full |
| 14 | [Attention](course/14-attention/) | Scaled dot-product attention from scratch | Full |
| 15 | [Transformers](course/15-transformers/) | Block, multi-head, positional encodings, LLM intuition | Full |
| — | [Lab D: Attention from Scratch](course/lab-D-attention-from-scratch/) | NumPy attention vs PyTorch's | — |

---

## Selected diagrams

Every diagram in this course is a committed matplotlib script + PNG. Here are four that capture the core ideas:

### MSE loss surface and the gradient descent path

![loss surface and GD path](./course/01-linear-regression/diagram_loss_surface.png)
*Module 01 — the bowl-shaped MSE loss over (w, b) with a gradient-descent trajectory. The geometry of why gradient descent converges.*

### SVM margin

![SVM margin](./course/04-svm/diagram_margin.png)
*Module 04 — the maximum-margin separator. The support vectors (points on the margin) are the only ones that matter; the rest can be removed without changing the solution.*

### MLP training curves

![MLP training curves](./course/10-neural-nets-mlp/diagram_training_curves.png)
*Module 10 — training and validation loss as the network learns. The gap between them is overfitting; the module shows how width changes it.*

### Attention weights

![attention heatmap](./course/14-attention/diagram_attention_heatmap.png)
*Module 14 — what the model attends to. Each row is a query position, each column a key position; bright cells are where the model looks.*

---

## Advanced ML

Deep learning, NLP, CV — covered end-to-end by modules 7–15 and the four labs. A roadmap for part II (tokenization, modern CNNs, detection, segmentation, VAEs, GANs, diffusion, RL) is in [ADVANCED.md](ADVANCED.md).

---

## Start here

1. Finish [SETUP.md](SETUP.md).
2. Open [course/00-math-foundations/README.md](course/00-math-foundations/) and walk through the primer.
3. Move to module 01.

Read. Derive. Plot. Code. Repeat.

---

## Cite this course

If you use this material, cite it with `CITATION.cff` or:

> Lourdhu Raju. *Machine Learning, From Scratch: A theory-first course.* https://github.com/Lourdhu02/machine-learning-advanced, 2024. MIT License.

---

## License

MIT — see [LICENSE](LICENSE).
