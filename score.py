import cv2
import numpy as np
import logging


def _kernel_self_dot(X: np.ndarray, kpca) -> np.ndarray:
    """Calculates the dot product of each sample with itself in kernel space."""
    if kpca.kernel == "rbf" or kpca.kernel == "cosine":
        return np.ones(X.shape[0])
    elif kpca.kernel == "linear":
        return np.sum(X**2, axis=1)
    elif kpca.kernel == "poly":
        # gamma is obtained from the fitted kpca object
        gamma = kpca.gamma if kpca.gamma is not None else 1.0 / X.shape[1]
        return (gamma * np.sum(X**2, axis=1) + kpca.coef0) ** kpca.degree
    elif kpca.kernel == "sigmoid":
        gamma = kpca.gamma if kpca.gamma is not None else 1.0 / X.shape[1]
        return np.tanh(gamma * np.sum(X**2, axis=1) + kpca.coef0)
    else:
        logging.warning(
            f"Cannot compute k(x,x) for kernel '{kpca.kernel}'. Reconstruction error will be incomplete."
        )
        return 0


def calculate_anomaly_scores(X: np.ndarray, pca: dict, method: str, drop_k: int = 0):
    """Calculates anomaly scores using specified PCA method."""
    if "kpca" in pca:
        if method != "reconstruction":
            logging.warning(
                f"Kernel PCA only supports 'reconstruction' scoring method. Ignoring '{method}'."
            )

        scaler = pca["scaler"]
        kpca = pca["kpca"]
        X_scaled = scaler.transform(X)

        # Anomaly score is the reconstruction error in the kernel-induced feature space.
        # Error = ||phi(x) - proj(phi(x))||^2 = k(x,x) - ||proj(phi(x))||^2
        X_projected = kpca.transform(X_scaled)
        k_x_x = _kernel_self_dot(X_scaled, kpca)

        # The projection norm is the sum of squares of projected components
        projection_norm_sq = np.sum(X_projected[:, drop_k:] ** 2, axis=1)

        score = k_x_x - projection_norm_sq
        # Scores can be negative due to numerical precision, clip at 0.
        return np.maximum(0, score)

    if drop_k < 0:
        raise ValueError("drop_k must be non-negative.")
    if drop_k >= pca["k"]:
        raise ValueError(f"drop_k ({drop_k}) cannot be >= num components ({pca['k']}).")

    if method == "reconstruction":
        W_dropped = pca["W"][:, drop_k:]
        P_dropped = pca["P"][drop_k:, :]
        Z_dropped = (X - pca["mu"]) @ W_dropped
        X_recon = Z_dropped @ P_dropped + pca["mu"]
        return np.sum((X - X_recon) ** 2, axis=1)

    elif method == "mahalanobis":
        if drop_k > 0:
            logging.warning(
                "drop_k is not supported for Mahalanobis distance. Ignoring."
            )
        Z = (X - pca["mu"]) @ pca["W"]
        return np.einsum("ij,jk,ik->i", Z, pca["cov_Z_inv"], Z)
    elif method == "cosine":
        if drop_k > 0:
            logging.warning("drop_k is not supported for Cosine distance. Ignoring.")
        # The features are already normalized in the feature extractor
        X_recon = (X - pca["mu"]) @ pca["W"] @ pca["P"] + pca["mu"]
        return 1 - np.sum(X * X_recon, axis=1)

    else:
        raise ValueError(f"Unknown scoring method '{method}'.")


def post_process_map(anomaly_map: np.ndarray, res):
    """Resizes and blurs the anomaly map."""
    if isinstance(res, int):
        dsize = (res, res)
    else:
        # patching.py passes (height, width), cv2 wants (width, height)
        dsize = (res[1], res[0])
    map_resized = cv2.resize(anomaly_map, dsize, interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(map_resized, (5, 5), 0)
