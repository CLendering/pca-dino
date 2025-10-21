import logging
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


class KernelPCAModel:
    """Kernel PCA for anomaly detection using scikit-learn."""

    def __init__(self, k=None, kernel="rbf", gamma=None, whiten=False, eps=1e-6):
        self.k = k
        self.kernel = kernel
        self.gamma = gamma
        self.whiten = whiten
        self.eps = eps
        self.scaler = None
        self.kpca = None
        self.pca_params = {}

    def fit(self, features: np.ndarray):
        logging.info("Starting Kernel PCA fit...")

        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        self.kpca = KernelPCA(
            n_components=self.k,
            kernel=self.kernel,
            gamma=self.gamma,
            copy_X=False,
            # fit_inverse_transform=True, # Needs scikit-learn 0.24+
        )

        logging.info(f"Fitting KernelPCA with kernel='{self.kernel}'...")
        self.kpca.fit(features_scaled)

        # In Kernel PCA, reconstruction is not as straightforward as in standard PCA.
        # The pre-image problem is non-trivial. For anomaly detection, we can
        # measure the distance in the feature space.
        # A common method is to calculate the squared distance of a sample
        # to its projection onto the principal components.

        # For the purpose of this implementation, we will store the fitted model
        # and the training data features required for transformation.
        # The actual scoring will need to handle the transformation.

        self.pca_params = {
            "scaler": self.scaler,
            "kpca": self.kpca,
            "k": self.k,
            "whiten": self.whiten,  # Note: Whitening is handled differently in KPCA
            "eps": self.eps,
            "train_features_scaled": features_scaled,
        }
        logging.info("Kernel PCA fit complete.")
        return self.pca_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"PCAModel will use device: {device}")


class PCAModel:
    """
    Memory-efficient PCA using a two-pass streaming algorithm.
    This version is accelerated using PyTorch for GPU computation.
    Based on https://github.com/dnhkng/PCAonGPU
    """

    def __init__(self, k=None, ev=None, whiten=False, eps=1e-6):
        self.k = k
        self.ev_ratio = ev
        self.whiten = whiten
        self.eps = eps
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.pca_params = {}
        self.device = device
        self.dtype = torch.float64

    def fit(
        self, feature_generator, feature_dim: int, total_tokens: int, num_batches: int
    ):
        logging.info(f"Starting PCA fit on {self.device}...")

        # Pass 1: Compute mean
        self.mean_ = torch.zeros(feature_dim, dtype=self.dtype, device=self.device)

        for batch in tqdm(
            feature_generator(),
            total=num_batches,
            desc="PCA Pass 1/2 (Mean)",
        ):
            batch_gpu = torch.from_numpy(batch).to(self.device, dtype=self.dtype)
            self.mean_ += torch.sum(batch_gpu, axis=0)

        self.mean_ /= total_tokens

        # Pass 2: Compute covariance
        cov_matrix = torch.zeros(
            (feature_dim, feature_dim), dtype=self.dtype, device=self.device
        )

        for batch in tqdm(
            feature_generator(),
            total=num_batches,
            desc="PCA Pass 2/2 (Cov)",
        ):
            batch_gpu = torch.from_numpy(batch).to(self.device, dtype=self.dtype)
            batch_centered = batch_gpu - self.mean_
            cov_matrix += torch.matmul(batch_centered.T, batch_centered)

        cov_matrix /= total_tokens - 1

        logging.info("Performing eigendecomposition on GPU...")
        evals, evecs = torch.linalg.eigh(cov_matrix)

        sorted_indices = torch.argsort(evals, descending=True)
        self.explained_variance_ = evals[sorted_indices]
        evecs = evecs[:, sorted_indices]

        if self.ev_ratio is not None and self.k is None:
            cumulative_variance = torch.cumsum(
                self.explained_variance_, dim=0
            ) / torch.sum(self.explained_variance_)
            self.k = (
                torch.searchsorted(
                    cumulative_variance,
                    torch.tensor([self.ev_ratio], dtype=self.dtype, device=self.device),
                ).item()
                + 1
            )
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
            W *= 1.0 / torch.sqrt(self.explained_variance_[: self.k] + self.eps)

        self.pca_params = {
            "mu": self.mean_.cpu().numpy().astype(np.float64),
            "W": W.cpu().numpy().astype(np.float64),
            "P": P.cpu().numpy().astype(np.float64),
            "evals": self.explained_variance_.cpu().numpy().astype(np.float64),
            "k": self.k,
            "whiten": self.whiten,
            "eps": self.eps,
            "cov_Z_inv": torch.linalg.inv(
                torch.diag(self.explained_variance_[: self.k])
            )
            .cpu()
            .numpy()
            .astype(np.float64),
        }
        return self.pca_params
