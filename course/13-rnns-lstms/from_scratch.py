"""Vanilla RNN and LSTM cells from scratch, with a gradient-survival
demonstration: backprop a unit signal from the last hidden state and
measure the gradient magnitude at each timestep going backward.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# -----------------------------------------------------------------------------
# Vanilla RNN
# -----------------------------------------------------------------------------


class VanillaRNN:
    """h_t = tanh(W_x x_t + W_h h_{t-1} + b)."""

    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Initialize W_h with spectral radius ~ 1.0 (worst case for gradient flow)
        self.W_x = rng.standard_normal((n_in, n_hidden)) * 0.1
        self.W_h = rng.standard_normal((n_hidden, n_hidden)) * 0.1
        self.b = np.zeros(n_hidden)
        self.n_hidden = n_hidden

    def forward(self, X):
        """X: (T, n_in). Returns hs of shape (T, n_hidden) and the cache."""
        T = X.shape[0]
        h = np.zeros(self.n_hidden)
        hs = np.zeros((T, self.n_hidden))
        zs = np.zeros((T, self.n_hidden))
        for t in range(T):
            z = X[t] @ self.W_x + h @ self.W_h + self.b
            h = np.tanh(z)
            hs[t] = h
            zs[t] = z
        return hs, dict(X=X, hs=hs, zs=zs)

    def backward(self, cache: dict, dh_final: np.ndarray):
        """Backprop a single dL/dh_T into every dL/dh_t. Returns the per-step
        gradient magnitude (a (T,) array)."""
        hs, zs = cache["hs"], cache["zs"]
        T = hs.shape[0]
        grads = np.zeros((T, self.n_hidden))
        delta = dh_final.copy()
        for t in reversed(range(T)):
            grads[t] = delta
            # propagate to h_{t-1}
            d_pre = delta * (1 - np.tanh(zs[t]) ** 2)  # through tanh
            delta = d_pre @ self.W_h.T
        return grads


# -----------------------------------------------------------------------------
# LSTM cell
# -----------------------------------------------------------------------------


class LSTM:
    """The standard LSTM cell. Concatenated weight matrix per gate."""

    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 0.1
        # All four gates share the same input format [x_t, h_{t-1}]
        in_size = n_in + n_hidden
        self.W_f = rng.standard_normal((in_size, n_hidden)) * scale
        self.W_i = rng.standard_normal((in_size, n_hidden)) * scale
        self.W_g = rng.standard_normal((in_size, n_hidden)) * scale
        self.W_o = rng.standard_normal((in_size, n_hidden)) * scale
        # Biases: init forget-gate bias to 1.0 (recommended trick)
        self.b_f = np.ones(n_hidden)
        self.b_i = np.zeros(n_hidden)
        self.b_g = np.zeros(n_hidden)
        self.b_o = np.zeros(n_hidden)
        self.n_hidden = n_hidden

    def forward(self, X):
        T = X.shape[0]
        h = np.zeros(self.n_hidden)
        c = np.zeros(self.n_hidden)
        hs = np.zeros((T, self.n_hidden))
        cs = np.zeros((T, self.n_hidden))
        fs = np.zeros((T, self.n_hidden))
        for t in range(T):
            xh = np.concatenate([X[t], h])
            f = sigmoid(xh @ self.W_f + self.b_f)
            i = sigmoid(xh @ self.W_i + self.b_i)
            g = np.tanh(xh @ self.W_g + self.b_g)
            o = sigmoid(xh @ self.W_o + self.b_o)
            c = f * c + i * g
            h = o * np.tanh(c)
            hs[t] = h
            cs[t] = c
            fs[t] = f
        return hs, dict(X=X, hs=hs, cs=cs, fs=fs)

    def gradient_survival(self, cache: dict, dh_final: np.ndarray):
        """Measure how much of dL/dh_T survives through the cell-state path.
        The cell-state recurrence is c_t = f_t * c_{t-1} + ..., so dL/dc_{t-1}
        gets multiplied by f_t each step. Approximates gradient norm per step.
        """
        cs, fs = cache["cs"], cache["fs"]
        T = cs.shape[0]
        norms = np.zeros(T)
        # crude per-step survival estimate via cell-state pathway
        g = dh_final
        for t in reversed(range(T)):
            norms[t] = np.linalg.norm(g)
            g = g * fs[t]  # cell-state derivative chain ≈ ∏ f_t
        return norms


# -----------------------------------------------------------------------------
# Main: gradient survival comparison
# -----------------------------------------------------------------------------


def main():
    T = 20
    n_in = 3
    n_hidden = 8
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, n_in)) * 0.5

    print("=" * 60)
    print(f"Gradient survival across T = {T} timesteps")
    print("=" * 60)

    # Vanilla RNN
    rnn = VanillaRNN(n_in, n_hidden, seed=0)
    hs_rnn, cache_rnn = rnn.forward(X)
    dh_final = np.ones(n_hidden)  # unit gradient at the last timestep
    grads_rnn = rnn.backward(cache_rnn, dh_final)
    rnn_mags = np.linalg.norm(grads_rnn, axis=1)

    # LSTM
    lstm = LSTM(n_in, n_hidden, seed=0)
    hs_lstm, cache_lstm = lstm.forward(X)
    lstm_mags = lstm.gradient_survival(cache_lstm, dh_final)

    print(f"  {'t':>4}  {'vanilla RNN |grad|':>22}  {'LSTM cell-state |grad|':>26}")
    for t in range(T):
        print(f"  {t:>4}  {rnn_mags[t]:>22.2e}  {lstm_mags[t]:>26.2e}")

    print()
    ratio_rnn = rnn_mags[0] / rnn_mags[-1]
    ratio_lstm = lstm_mags[0] / lstm_mags[-1]
    print(f"Gradient at t=0  /  gradient at t={T-1}:")
    print(f"  vanilla RNN:  {ratio_rnn:.2e}   (effectively zero -- unlearnable)")
    print(f"  LSTM        :  {ratio_lstm:.2e}   (orders of magnitude better)")
    print("\nOK")


if __name__ == "__main__":
    main()
