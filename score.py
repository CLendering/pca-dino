import cv2
import numpy as np
import logging


def _kernel_self_dot(X: np.ndarray, kpca) -> np.ndarray:
    """Compute k(x,x) for several sklearn KPCA kernels."""
    if kpca.kernel in ("rbf", "cosine"):
        return np.ones(X.shape[0])
    elif kpca.kernel == "linear":
        return np.sum(X**2, axis=1) + kpca.coef0
    elif kpca.kernel == "poly":
        gamma = kpca.gamma if kpca.gamma is not None else 1.0 / X.shape[1]
        return (gamma * np.sum(X**2, axis=1) + kpca.coef0) ** kpca.degree
    elif kpca.kernel == "sigmoid":
        gamma = kpca.gamma if kpca.gamma is not None else 1.0 / X.shape[1]
        return np.tanh(gamma * np.sum(X**2, axis=1) + kpca.coef0)
    else:
        logging.warning(
            f"Cannot compute k(x,x) for kernel '{kpca.kernel}'. Reconstruction error will be approximate."
        )
        return np.zeros(X.shape[0], dtype=X.dtype)


def _row_l2(X: np.ndarray, eps: float) -> np.ndarray:
    """Normalize rows to unit length."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def pca_reconstruct(X: np.ndarray, pca: dict, drop_k: int = 0) -> np.ndarray:
    """Reconstruct X in original space using unscaled components."""
    mu = np.asarray(pca["mu"], dtype=X.dtype)
    C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
    X0 = X - mu
    Z = X0 @ C
    if drop_k > 0:
        if drop_k >= Z.shape[1]:
            Z[:] = 0.0  # All components are "normal", so zero them out
        else:
            Z[:, :drop_k] = 0.0  # Zero out the "normal" components
    X_recon = (Z @ C.T) + mu
    return X_recon


def calculate_anomaly_scores(X: np.ndarray, pca: dict, method: str, drop_k: int = 0):
    """
    Calculate anomaly scores using PCA/KernelPCA based on reconstruction criteria.
    Methods:
      - 'reconstruction': (Default) Squared L2 recon error in original space.
      - 'mahalanobis'   : Mahalanobis distance (variance-normalized error)
                          in "abnormal" PC subspace [drop_k...k].
      - 'euclidean'     : Squared Euclidean distance (absolute error)
                          in "abnormal" PC subspace [drop_k...k].
      - 'cosine'        : Angular reconstruction error in original space.
      - KPCA 'reconstruction' (if 'kpca' in pca)
    """
    # --------------------------- Kernel PCA branch ----------------------------
    if "kpca" in pca:
        if method != "reconstruction":
            logging.warning(
                f"Kernel PCA only supports 'reconstruction' scoring method. Using 'reconstruction'."
            )

        scaler = pca["scaler"]
        kpca = pca["kpca"]
        X_scaled = scaler.transform(X)

        X_proj = kpca.transform(X_scaled)
        k_x_x = _kernel_self_dot(X_scaled, kpca)

        if drop_k > 0:
            if drop_k >= X_proj.shape[1]:
                X_proj = np.zeros_like(X_proj)  # All components dropped
            else:
                # This logic is correct: error of reconstructing from abnormal components
                X_proj = X_proj[:, drop_k:]

        proj_norm_sq = np.sum(X_proj**2, axis=1)
        score = k_x_x - proj_norm_sq
        return np.maximum(0.0, score)

    # --------------------------- Standard PCA branch --------------------------
    if drop_k < 0:
        raise ValueError("drop_k must be non-negative.")
    if drop_k >= pca["k"]:
        logging.warning(f"drop_k ({drop_k}) is >= num components ({pca['k']}).")
        if method == "mahalanobis" or method == "euclidean":
            return np.zeros(X.shape[0], dtype=X.dtype)

    if method == "reconstruction":
        X_recon = pca_reconstruct(X, pca, drop_k=drop_k)
        return np.sum((X - X_recon) ** 2, axis=1)

    elif method == "mahalanobis":
        mu = np.asarray(pca["mu"], dtype=X.dtype)
        C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
        Z = (X - mu) @ C  # [N, k]

        if drop_k >= pca["k"]:
            return np.zeros(X.shape[0], dtype=X.dtype)

        # We only want the components from drop_k onwards
        Z_abnormal = Z[:, drop_k:]  # Z is [N, k - drop_k]
        eigvals_abnormal = np.asarray(pca["eigvals"][drop_k:], dtype=X.dtype)

        # Calculate Mahalanobis distance in this subspace
        # score = sum( (z_i^2 / lambda_i) ) for i=drop_k...k
        cov_inv = np.diag(1.0 / (eigvals_abnormal + pca["eps"]))
        return np.einsum("ij,jk,ik->i", Z_abnormal, cov_inv, Z_abnormal)

    elif method == "euclidean":
        mu = np.asarray(pca["mu"], dtype=X.dtype)
        C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
        Z = (X - mu) @ C  # [N, k]

        if drop_k >= pca["k"]:
            return np.zeros(X.shape[0], dtype=X.dtype)

        Z_abnormal = Z[:, drop_k:]  # Z is [N, k - drop_k]

        # Calculate Euclidean distance in this subspace (squared L2 norm)
        # score = sum( z_i^2 ) for i=drop_k...k
        return np.sum(Z_abnormal**2, axis=1)

    # ---- Cosine (Angular reconstruction error) ----
    elif method == "cosine":
        # 1. Reconstruct X from PCA space
        X_recon = pca_reconstruct(X, pca, drop_k=drop_k)

        # 2. L2-normalize the original vectors X
        X_norm = _row_l2(X, pca["eps"])

        # 3. L2-normalize the reconstructed vectors X_recon
        X_recon_norm = _row_l2(X_recon, pca["eps"])

        # 4. Compute cosine similarity (batched dot product)
        sim = np.einsum("ij,ij->i", X_norm, X_recon_norm)

        # 5. Score is 1 - similarity. Clip for numerical stability.
        sim = np.clip(sim, -1.0, 1.0)
        return 1.0 - sim

    else:
        raise ValueError(f"Unknown scoring method '{method}'.")


def post_process_map(anomaly_map: np.ndarray, res, blur: bool = True):
    """Resize + blur the anomaly map."""
    if anomaly_map.dtype != np.float32:
        anomaly_map = anomaly_map.astype(np.float32)

    dsize = (res, res) if isinstance(res, int) else (res[1], res[0])
    map_resized = cv2.resize(anomaly_map, dsize, interpolation=cv2.INTER_LINEAR)

    if isinstance(res, int):
        scalar_res = res
    else:
        scalar_res = min(res)

    k_size = int(scalar_res / 50)
    if k_size % 2 == 0:
        k_size += 1
    k_size = max(3, k_size)
    # A common rule of thumb:
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8

    if blur:
        return cv2.GaussianBlur(map_resized, (k_size, k_size), sigma)
    else:
        return map_resized
