import logging
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


class KernelPCAModel:
    def __init__(self, k=None, kernel="rbf", gamma=None, eps=1e-6):
        self.k = k
        self.kernel = kernel
        self.gamma = gamma
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
            copy_X=False,  # Saves memory by avoiding an extra copy
        )

        logging.info(f"Fitting KernelPCA with kernel='{self.kernel}'...")
        self.kpca.fit(features_scaled)
        self.pca_params = {
            "scaler": self.scaler,
            "kpca": self.kpca,
            "k": self.k,
            "eps": self.eps,
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
                f"PCA: selected k={self.k} components to explain {self.ev_ratio * 100:.2f}% of variance."
            )

        if self.k is None:
            self.k = evecs.shape[1]
        else:
            self.k = min(self.k, evecs.shape[1])

        self.components_ = evecs[:, : self.k]  # [D, k], unscaled eigenvectors
        self.eigvals_ = self.explained_variance_[: self.k]  # [k]
        self.mu_ = self.mean_

        self.pca_params = {
            "mu": self.mu_.cpu().numpy().astype(np.float64),  # [D]
            "components": self.components_.cpu().numpy().astype(np.float64),  # [D, k]
            "eigvals": self.eigvals_.cpu().numpy().astype(np.float64),  # [k]
            "sqrt_eig": np.sqrt(
                self.eigvals_.cpu().numpy().astype(np.float64) + self.eps
            ),
            "k": self.k,
            "whiten": self.whiten,
            "eps": self.eps,
            # For Mahalanobis in PC space (unwhitened Z has diag cov = eigvals):
            "cov_Z_inv": np.diag(
                1.0 / (self.eigvals_.cpu().numpy().astype(np.float64) + self.eps)
            ),
        }
        return self.pca_params


def get_pc_projection_map(
    tokens_reshaped: np.ndarray, pca: dict, pc_index: int = 0
):
    """
    Projects tokens onto a single principal component (e.g., PC1) to create a
    'normality' map, as done in AnomalyDINO.
    Returns a 1D score (projection value) for each token.
    """
    if "kpca" in pca:
        logging.warning(
            "PCA projection masking is not supported for Kernel PCA. Returning empty mask."
        )
        return np.zeros(tokens_reshaped.shape[0], dtype=np.float32)

    if pc_index >= pca["k"]:
        raise ValueError(
            f"pc_index {pc_index} is out of bounds for PCA with k={pca['k']}"
        )

    mu = np.asarray(pca["mu"], dtype=tokens_reshaped.dtype)
    # Get only the component we care about
    C = np.asarray(
        pca["components"][:, pc_index : pc_index + 1], dtype=tokens_reshaped.dtype
    )  # [D, 1]

    # Project centered tokens onto the component
    Z = (tokens_reshaped - mu) @ C  # [N, D] @ [D, 1] -> [N, 1]

    # AnomalyDINO uses the *absolute* projection value as the normality score
    return np.abs(Z.flatten())