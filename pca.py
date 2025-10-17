import logging
import math
import numpy as np
from tqdm import tqdm


class PCAModel:
    """Memory-efficient PCA using a two-pass streaming algorithm."""

    def __init__(self, k=None, ev=None, whiten=False, eps=1e-6):
        self.k = k
        self.ev_ratio = ev
        self.whiten = whiten
        self.eps = eps
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.pca_params = {}

    def fit(self, feature_generator, feature_dim: int, total_tokens: int):
        logging.info("Starting PCA fit...")
        # Pass 1: Compute mean
        self.mean_ = np.zeros(feature_dim, dtype=np.float64)
        for batch in tqdm(
            feature_generator(),
            total=math.ceil(
                total_tokens / (total_tokens / len(next(feature_generator())))
            ),
            desc="PCA Pass 1/2 (Mean)",
        ):
            self.mean_ += np.sum(batch, axis=0)
        self.mean_ /= total_tokens

        # Pass 2: Compute covariance
        cov_matrix = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        for batch in tqdm(
            feature_generator(),
            total=math.ceil(
                total_tokens / (total_tokens / len(next(feature_generator())))
            ),
            desc="PCA Pass 2/2 (Cov)",
        ):
            batch_centered = batch - self.mean_
            cov_matrix += batch_centered.T @ batch_centered
        cov_matrix /= total_tokens - 1

        logging.info("Performing eigendecomposition...")
        evals, evecs = np.linalg.eigh(cov_matrix)

        sorted_indices = np.argsort(evals)[::-1]
        self.explained_variance_ = evals[sorted_indices]
        evecs = evecs[:, sorted_indices]

        if self.ev_ratio is not None and self.k is None:
            cumulative_variance = np.cumsum(self.explained_variance_) / np.sum(
                self.explained_variance_
            )
            self.k = np.searchsorted(cumulative_variance, self.ev_ratio) + 1
            logging.info(
                f"PCA: selected k={self.k} components to explain {self.ev_ratio*100:.2f}% of variance."
            )

        if self.k is None:
            self.k = evecs.shape[1]
        else:
            self.k = min(self.k, evecs.shape[1])

        P = evecs[:, : self.k].T
        W = evecs[:, : self.k]
        if self.whiten:
            W *= 1.0 / np.sqrt(self.explained_variance_[: self.k] + self.eps)

        self.pca_params = {
            "mu": self.mean_.astype(np.float32),
            "W": W.astype(np.float32),
            "P": P.astype(np.float32),
            "evals": self.explained_variance_.astype(np.float32),
            "k": self.k,
            "whiten": self.whiten,
            "eps": self.eps,
            "cov_Z_inv": np.linalg.inv(
                np.diag(self.explained_variance_[: self.k])
            ).astype(np.float32),
        }
        return self.pca_params
