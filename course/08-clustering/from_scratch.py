"""KMeans (Lloyd + k-means++) and GaussianMixture (EM) from scratch.

Run: python from_scratch.py
"""

from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# k-means with k-means++ init
# -----------------------------------------------------------------------------


class KMeans:
    def __init__(self, n_clusters: int = 3, n_iter: int = 100,
                 tol: float = 1e-6, seed: int = 0):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed

    def _init_centroids(self, X: np.ndarray, rng) -> np.ndarray:
        n = X.shape[0]
        centroids = [X[rng.integers(0, n)]]
        for _ in range(1, self.n_clusters):
            dists = np.min(
                ((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(-1),
                axis=1,
            )
            probs = dists / dists.sum()
            centroids.append(X[rng.choice(n, p=probs)])
        return np.array(centroids)

    def fit(self, X: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.seed)
        self.centroids_ = self._init_centroids(X, rng)
        self.J_history_ = []

        for _ in range(self.n_iter):
            # assignment step
            d2 = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(-1)
            labels = np.argmin(d2, axis=1)
            # objective
            J = float(d2[np.arange(X.shape[0]), labels].sum())
            self.J_history_.append(J)
            # update step
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any()
                else self.centroids_[k]
                for k in range(self.n_clusters)
            ])
            if np.max(np.abs(new_centroids - self.centroids_)) < self.tol:
                self.centroids_ = new_centroids
                self.labels_ = labels
                break
            self.centroids_ = new_centroids
            self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        d2 = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(-1)
        return np.argmin(d2, axis=1)


# -----------------------------------------------------------------------------
# Gaussian Mixture Model with EM
# -----------------------------------------------------------------------------


def _log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log N(x; mean, cov) for each row of X."""
    d = X.shape[1]
    diff = X - mean
    cov_reg = cov + 1e-6 * np.eye(d)
    inv = np.linalg.inv(cov_reg)
    log_det = float(np.linalg.slogdet(cov_reg)[1])
    quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return -0.5 * (d * np.log(2 * np.pi) + log_det + quad)


class GaussianMixture:
    """Full-covariance EM."""

    def __init__(self, n_components: int = 3, n_iter: int = 100,
                 tol: float = 1e-6, seed: int = 0):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X: np.ndarray) -> "GaussianMixture":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        K = self.n_components

        # init: use KMeans for cluster centers, identity for cov
        km = KMeans(n_clusters=K, seed=self.seed).fit(X)
        self.means_ = km.centroids_.copy()
        self.covariances_ = np.array([np.eye(d) for _ in range(K)])
        self.weights_ = np.full(K, 1.0 / K)

        self.loglik_history_ = []
        prev = -np.inf
        for _ in range(self.n_iter):
            # E-step: compute log-responsibilities (stable via log-sum-exp)
            log_unnorm = np.empty((n, K))
            for k in range(K):
                log_unnorm[:, k] = np.log(self.weights_[k] + 1e-12) \
                                    + _log_gaussian(X, self.means_[k], self.covariances_[k])
            # log-sum-exp normalization
            m = log_unnorm.max(axis=1, keepdims=True)
            log_norm = m.ravel() + np.log(np.exp(log_unnorm - m).sum(axis=1))
            log_resp = log_unnorm - log_norm[:, None]
            resp = np.exp(log_resp)

            loglik = float(log_norm.sum())
            self.loglik_history_.append(loglik)

            # M-step
            Nk = resp.sum(axis=0)
            self.weights_ = Nk / n
            self.means_ = (resp.T @ X) / Nk[:, None]
            for k in range(K):
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k][:, None] * diff).T @ diff / Nk[k]

            if abs(loglik - prev) < self.tol:
                break
            prev = loglik

        self.responsibilities_ = resp
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n, _ = X.shape
        K = self.n_components
        log_unnorm = np.empty((n, K))
        for k in range(K):
            log_unnorm[:, k] = np.log(self.weights_[k] + 1e-12) \
                                + _log_gaussian(X, self.means_[k], self.covariances_[k])
        return np.argmax(log_unnorm, axis=1)


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------


def make_spherical(n=300, seed=0):
    rng = np.random.default_rng(seed)
    per = n // 3
    centers = np.array([[-2.5, -1.0], [2.5, -1.0], [0.0, 2.5]])
    X = np.vstack([
        rng.standard_normal((per, 2)) * 0.5 + centers[k]
        for k in range(3)
    ])
    y = np.concatenate([k * np.ones(per, dtype=int) for k in range(3)])
    return X, y


def make_elongated(n=300, seed=0):
    """Three differently-tilted, elongated Gaussians."""
    rng = np.random.default_rng(seed)
    per = n // 3
    Xs = []
    angles = [0.0, np.pi / 3, -np.pi / 4]
    centers = np.array([[-3.0, 0.0], [3.0, -1.0], [0.0, 3.5]])
    for k in range(3):
        z = rng.standard_normal((per, 2)) * np.array([1.4, 0.3])
        R = np.array([[np.cos(angles[k]), -np.sin(angles[k])],
                      [np.sin(angles[k]),  np.cos(angles[k])]])
        Xs.append(z @ R.T + centers[k])
    X = np.vstack(Xs)
    y = np.concatenate([k * np.ones(per, dtype=int) for k in range(3)])
    return X, y


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Spherical clusters: k-means and GMM both work")
    print("=" * 60)
    Xs, _ = make_spherical()
    km = KMeans(n_clusters=3, seed=0).fit(Xs)
    print(f"  KMeans  final J = {km.J_history_[-1]:.2f}  in {len(km.J_history_)} iters")
    gmm = GaussianMixture(n_components=3, seed=0).fit(Xs)
    print(f"  GMM     final log-lik = {gmm.loglik_history_[-1]:.2f}  in {len(gmm.loglik_history_)} iters")

    print()
    print("=" * 60)
    print("Elongated clusters: k-means struggles, GMM nails it")
    print("=" * 60)
    Xe, _ = make_elongated()
    km = KMeans(n_clusters=3, seed=0).fit(Xe)
    print(f"  KMeans  final J = {km.J_history_[-1]:.2f}")
    gmm = GaussianMixture(n_components=3, seed=0).fit(Xe)
    print(f"  GMM     final log-lik = {gmm.loglik_history_[-1]:.2f}")

    print()
    print("=" * 60)
    print("EM log-likelihood per iter (must be monotonic non-decreasing)")
    print("=" * 60)
    for t, ll in enumerate(gmm.loglik_history_[:10]):
        marker = "  +" if t == 0 or ll >= gmm.loglik_history_[t - 1] - 1e-9 else "  !"
        print(f"  iter {t:>2}: log-lik = {ll:>10.3f}{marker}")
    if len(gmm.loglik_history_) > 10:
        print(f"  ... ({len(gmm.loglik_history_)} total iters)")
    diffs = np.diff(gmm.loglik_history_)
    assert (diffs > -1e-6).all(), "EM log-likelihood went backwards"
    print("\nOK (monotonic)")


if __name__ == "__main__":
    main()
