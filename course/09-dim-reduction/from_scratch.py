"""PCA via SVD and a minimal t-SNE, both in NumPy.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# PCA
# -----------------------------------------------------------------------------


class PCA:
    """PCA via thin SVD on the centered data."""

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt
        self.singular_values_ = S
        total = (S ** 2).sum()
        self.explained_variance_ratio_ = (S ** 2) / total
        return self

    def transform(self, X: np.ndarray, k: int | None = None) -> np.ndarray:
        k = k if k is not None else (self.n_components or self.components_.shape[0])
        return (X - self.mean_) @ self.components_[:k].T


# -----------------------------------------------------------------------------
# t-SNE: minimal but real implementation
# -----------------------------------------------------------------------------


def _pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return np.clip(D, 0, None)


def _binary_search_sigma(D_i: np.ndarray, target_perplexity: float,
                         tol: float = 1e-5, max_iter: int = 50) -> np.ndarray:
    """For each row i, find sigma_i so that the Gaussian distribution over
    neighbours has the desired perplexity (= 2^entropy).
    """
    target = np.log2(target_perplexity)
    n = D_i.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        beta_lo, beta_hi = 1e-20, 1e20
        beta = 1.0  # beta = 1 / (2 sigma^2)
        Di = D_i[i].copy()
        Di[i] = np.inf
        for _ in range(max_iter):
            num = np.exp(-Di * beta)
            num[i] = 0
            sum_num = num.sum()
            if sum_num <= 0:
                beta /= 2
                continue
            Pi = num / sum_num
            # entropy
            Pi_clip = np.clip(Pi, 1e-12, None)
            H = -np.sum(Pi_clip * np.log2(Pi_clip))
            if abs(H - target) < tol:
                break
            if H > target:
                beta_lo = beta
                beta = (beta * 2) if beta_hi > 1e19 else (beta + beta_hi) / 2
            else:
                beta_hi = beta
                beta = (beta + beta_lo) / 2
        P[i] = Pi
    return P


def tsne(X: np.ndarray, n_iter: int = 500, perplexity: float = 30.0,
         lr: float = 200.0, momentum: float = 0.9, seed: int = 0) -> np.ndarray:
    """Minimal t-SNE. Returns Y of shape (n, 2)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    D = _pairwise_sq_dist(X)

    # P-matrix with binary-search bandwidth, symmetrized
    Pc = _binary_search_sigma(D, perplexity)
    P = (Pc + Pc.T) / (2 * n)
    P = np.clip(P, 1e-12, None)
    # early exaggeration multiplier (helps separate clusters early on)
    P_exag = P * 4

    Y = rng.standard_normal((n, 2)) * 1e-4
    velocity = np.zeros_like(Y)

    for it in range(n_iter):
        if it == 100:  # turn off exaggeration after a while
            P_exag = P

        # Q-matrix (Student-t with df=1)
        D_y = _pairwise_sq_dist(Y)
        num = 1.0 / (1.0 + D_y)
        np.fill_diagonal(num, 0)
        Q = np.clip(num / num.sum(), 1e-12, None)

        # Gradient of KL(P || Q) wrt Y
        # dC/dy_i = 4 sum_j (P_ij - Q_ij) * (1 + ||y_i - y_j||^2)^-1 * (y_i - y_j)
        PQ = (P_exag - Q) * num
        grad = 4 * ((np.diag(PQ.sum(axis=1)) - PQ) @ Y)

        velocity = momentum * velocity - lr * grad
        Y = Y + velocity
        Y = Y - Y.mean(axis=0)

    return Y


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------


def make_correlated_gaussian(n=300, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[3.0, 1.6], [1.6, 1.0]])
    L = np.linalg.cholesky(cov)
    return rng.standard_normal((n, 2)) @ L.T


def make_blobs_hd(n_per_cluster=50, d=50, n_clusters=4, sep=4.0, seed=0):
    """K well-separated isotropic Gaussian clusters in d dims."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for k in range(n_clusters):
        center = np.zeros(d)
        center[k] = sep  # each cluster on its own coordinate axis
        Xs.append(rng.standard_normal((n_per_cluster, d)) * 0.6 + center)
        ys.append(np.full(n_per_cluster, k, dtype=int))
    perm = rng.permutation(n_per_cluster * n_clusters)
    return np.vstack(Xs)[perm], np.concatenate(ys)[perm]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    np.set_printoptions(precision=3, suppress=True)

    print("=" * 60)
    print("PCA on 2D correlated Gaussian")
    print("=" * 60)
    X2 = make_correlated_gaussian(n=400)
    pca2 = PCA().fit(X2)
    print(f"  singular values         : {pca2.singular_values_}")
    print(f"  explained variance ratio: {pca2.explained_variance_ratio_}")
    print(f"  principal components    : {pca2.components_}")

    print()
    print("=" * 60)
    print("PCA + t-SNE on 50-D, 4-cluster data")
    print("=" * 60)
    X, y = make_blobs_hd(n_per_cluster=40, d=50, n_clusters=4, sep=4.0)
    pca = PCA().fit(X)
    Z_pca = pca.transform(X, k=2)
    print(f"  PCA explained variance (top 4 components):"
          f" {pca.explained_variance_ratio_[:4]}")
    print(f"  -> cumulative top-2: {pca.explained_variance_ratio_[:2].sum():.3f}")

    print("  running t-SNE (this is the slow part)...")
    Y_tsne = tsne(X, n_iter=400, perplexity=20.0)
    print(f"  t-SNE final embedding shape: {Y_tsne.shape}")

    # sanity: in 2D, the average distance between same-cluster points should
    # be smaller than the average distance between different-cluster points.
    def cluster_quality(Z, y):
        n = Z.shape[0]
        D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        same = D[(y[:, None] == y[None]) & ~np.eye(n, dtype=bool)].mean()
        diff = D[y[:, None] != y[None]].mean()
        return float(same), float(diff), float(diff / same)

    sq_pca = cluster_quality(Z_pca, y)
    sq_tsne = cluster_quality(Y_tsne, y)
    print(f"  PCA:    same-cluster avg dist={sq_pca[0]:.2f}  diff={sq_pca[1]:.2f}  ratio={sq_pca[2]:.2f}")
    print(f"  t-SNE:  same-cluster avg dist={sq_tsne[0]:.2f}  diff={sq_tsne[1]:.2f}  ratio={sq_tsne[2]:.2f}")
    assert sq_tsne[2] > sq_pca[2], "t-SNE should separate clusters at least as well as PCA"
    print("\nOK")


if __name__ == "__main__":
    main()
