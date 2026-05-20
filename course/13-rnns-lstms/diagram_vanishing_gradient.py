"""Gradient magnitude across timesteps for a vanilla RNN vs LSTM cell-state path.

Output: diagram_vanishing_gradient.png
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import VanillaRNN, LSTM


def main():
    T = 30
    n_in, n_hidden = 3, 8
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, n_in)) * 0.5

    rnn = VanillaRNN(n_in, n_hidden, seed=0)
    _, c_rnn = rnn.forward(X)
    dh = np.ones(n_hidden)
    g_rnn = np.linalg.norm(rnn.backward(c_rnn, dh), axis=1)

    lstm = LSTM(n_in, n_hidden, seed=0)
    _, c_lstm = lstm.forward(X)
    g_lstm = lstm.gradient_survival(c_lstm, dh)

    # plot
    t_axis = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.semilogy(t_axis, g_rnn, color="crimson", lw=2.2, marker="o", markersize=4,
                label="Vanilla RNN")
    ax.semilogy(t_axis, g_lstm, color="steelblue", lw=2.2, marker="s", markersize=4,
                label="LSTM (cell-state path)")
    ax.set_xlabel("timestep t  (gradient flowing backward from t = T-1)")
    ax.set_ylabel("|gradient|  (log scale)")
    ax.set_title("Vanilla RNN gradients vanish exponentially.\n"
                 "LSTM's additive cell state preserves them.")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("diagram_vanishing_gradient.png", dpi=140)
    print("wrote diagram_vanishing_gradient.png")


if __name__ == "__main__":
    main()
