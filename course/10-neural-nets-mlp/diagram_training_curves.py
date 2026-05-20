"""Cross-entropy loss and accuracy across training, for hidden widths 2, 8, 32.

Output: diagram_training_curves.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import MLPClassifier, make_moons


def main():
    X, y = make_moons(n=400, noise=0.22, seed=0)
    n_tr = 320
    Xt, Xv, yt, yv = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]

    widths = [2, 8, 32]
    colors = ["steelblue", "darkgreen", "crimson"]
    histories = []
    for h in widths:
        net = MLPClassifier(2, h, 2, seed=0)
        net.fit(Xt, yt, lr=0.2, n_epochs=200, batch_size=32,
                X_val=Xv, y_val=yv, verbose=False)
        histories.append(net.history_)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for h, color, hist in zip(widths, colors, histories):
        axes[0].plot(hist["loss"], color=color, lw=2, label=f"hidden = {h}")
        axes[1].plot(hist["acc"], color=color, lw=2, label=f"hidden = {h} (train)")
        axes[1].plot(hist["val_acc"], color=color, lw=1.2, ls="--",
                     label=f"hidden = {h} (test)")

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training cross-entropy")
    axes[0].set_title("Loss decreases faster for wider nets")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Wider nets fit easier; train-test gap is small here")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("diagram_training_curves.png", dpi=140)
    print("wrote diagram_training_curves.png")


if __name__ == "__main__":
    main()
