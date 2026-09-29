# Lab A — Classical Shootout (Logistic vs SVM vs Random Forest)

> Run after module 06. Three different families of classifier — linear, kernel, ensemble — on the *same* dataset, side by side, no scikit-learn in the core loop.

This is the "every algorithm is somebody's best algorithm" lab. When one model decisively beats the others, the data's geometry is telling you something.

---

## What this lab does

`shootout.py` imports the from-scratch implementations from modules 02, 04, and 06, trains all three on two carefully chosen datasets, and reports:

- Train and test accuracy per classifier.
- Time-to-fit (rough — wall clock).
- A 2×3 grid of decision boundaries, one row per dataset, one column per classifier.

The two datasets are designed to make different classifiers win:

| Dataset | Geometry | Predicted winner |
|---|---|---|
| Linearly separable blobs | Two Gaussian clouds | Logistic Regression (lowest variance, fastest, well-calibrated) |
| Two interleaved moons | Non-linear, curved boundary | Kernel SVM (RBF) or Random Forest |

---

## Run

```powershell
python shootout.py
```

Expected output: an accuracy table to the terminal and a `diagram_shootout.png` file.

![side-by-side decision boundaries on two datasets](./diagram_shootout.png)

---

## How to read the result

- **Boundary shape** is the most informative thing. Logistic draws a single line. SVM with RBF draws a curve. Random Forest draws axis-aligned blocks.
- **Boundary smoothness** is the next thing. SVM gives smooth curves (it's optimizing margin in feature space). RF gives jagged blocks (each axis-aligned cut adds a step). Logistic is the smoothest because it's just a hyperplane.
- **Accuracy gap** between models is bigger on the moons (where the true boundary is curved) than on the blobs (where any decent classifier wins).

The lesson: *the model is a hypothesis about the geometry of the boundary*. Pick wrong, no amount of hyperparameter tuning saves you. Pick right and you barely need to tune at all.

---

## Why no neural net here?

Neural nets enter in module 10. The fair comparison "linear vs kernel vs ensemble vs neural net" comes in Lab C on a larger problem (MNIST). On these tiny 2D problems an MLP would either over-fit or just rediscover one of the boundaries above.

---

## References

### Papers
- **Cox, D.R. (1958).** *The regression analysis of binary sequences.* Journal of the Royal Statistical Society: Series B, 20(2), 215–232. [https://doi.org/10.1111/j.2517-6161.1958.tb00290.x] [The original logistic regression paper.]
- **Cortes, C. & Vapnik, V. (1995).** *Support-vector networks.* Machine Learning, 20(3), 273–297. [https://link.springer.com/article/10.1007/BF00994018] [The canonical SVM paper.]
- **Boser, B., Guyon, I. & Vapnik, V. (1992).** *A training algorithm for optimal margin classifiers.* COLT 1992, 144–152. [The original max-margin classifier.]
- **Breiman, L. (2001).** *Random forests.* Machine Learning, 45(1), 5–32. [https://doi.org/10.1023/A:1010950713441] [The Random Forest paper.]

> Lab A builds on modules 02 (logistic regression), 04 (SVM), and 06 (ensembles). See those modules' READMEs for the full paper lists.

## 8. Advanced

- **No free lunch theorem** (Wolpert & Macready, 1997): averaged over all possible data distributions, every classifier has the same error rate. This means there is no universally best algorithm — the best choice depends on the structure of the specific problem. This is why the shootout matters.
- **The comparison in this lab is on one dataset**; a fair comparison requires multiple datasets or a nested cross-validation to avoid overfitting to the dataset choice.
