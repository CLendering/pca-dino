import cv2
import numpy as np
import logging


def calculate_anomaly_scores(X: np.ndarray, pca: dict, method: str, drop_k: int = 0):
    """Calculates anomaly scores using specified PCA method."""
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
        Z = (X - pca["mu"]) @ pca["W"]
        X_recon = Z @ pca["P"] + pca["mu"]
        dot_product = np.einsum("ij,ij->i", X, X_recon)
        norms = np.linalg.norm(X, axis=1) * np.linalg.norm(X_recon, axis=1)
        cosine_sim = np.divide(
            dot_product,
            norms,
            out=np.zeros_like(dot_product, dtype=float),
            where=norms != 0,
        )
        return 1 - cosine_sim

    else:
        raise ValueError(f"Unknown scoring method '{method}'.")


def post_process_map(anomaly_map: np.ndarray, res: int):
    """Resizes and blurs the anomaly map."""
    map_resized = cv2.resize(anomaly_map, (res, res), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(map_resized, (5, 5), 0)
