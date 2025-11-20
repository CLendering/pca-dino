# logical.py
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from features import FeatureExtractor


EPS = 1e-8


def _collect_tokens_for_image(
    img_path: str,
    extractor: FeatureExtractor,
    image_res: int,
    layers: List[int],
    agg_method: str,
    grouped_layers: List[List[int]],
    docrop: bool,
    use_clahe: bool,
    dino_saliency_layer: int,
):
    """Extracts spatial tokens for a single image."""
    pil_img = Image.open(img_path).convert("RGB")
    tokens, (h_p, w_p), _ = extractor.extract_tokens(
        [pil_img],
        image_res,
        layers,
        agg_method,
        grouped_layers,
        docrop,
        use_clahe=use_clahe,
        dino_saliency_layer=dino_saliency_layer,
    )
    # tokens: [1, h_p, w_p, C]
    tokens_spatial = tokens[0]  # [h_p, w_p, C]
    return pil_img.size, tokens_spatial, (h_p, w_p)


def _fit_token_clusters(
    train_paths: List[str],
    extractor: FeatureExtractor,
    args,
    layers: List[int],
    grouped_layers: List[List[int]],
) -> Dict[str, Any]:
    """First pass: fit MiniBatchKMeans on all training tokens."""
    if not train_paths:
        return {}

    logging.info(
        f"[Logical] Fitting MiniBatchKMeans with K={args.logical_num_parts} on "
        f"{len(train_paths)} train images..."
    )

    mbk = MiniBatchKMeans(
        n_clusters=args.logical_num_parts,
        batch_size=args.logical_cluster_batch_size,
        n_init="auto",
        random_state=42,
    )

    for path in train_paths:
        _, tokens_spatial, (h_p, w_p) = _collect_tokens_for_image(
            path,
            extractor,
            args.image_res,
            layers,
            args.agg_method,
            grouped_layers,
            args.docrop,
            args.use_clahe,
            args.dino_saliency_layer,
        )
        h, w, C = tokens_spatial.shape
        feats = tokens_spatial.reshape(-1, C)

        # Optional subsampling per image
        max_tokens = args.logical_max_tokens_per_image
        if max_tokens is not None and feats.shape[0] > max_tokens:
            idx = np.random.choice(
                feats.shape[0], size=max_tokens, replace=False
            )
            feats = feats[idx]

        mbk.partial_fit(feats.astype(np.float32))

    centers = mbk.cluster_centers_.astype(np.float32)
    logging.info("[Logical] Clustering complete.")
    return {
        "kmeans": mbk,
        "centers": centers,
        "K": centers.shape[0],
        "feat_dim": centers.shape[1],
    }


def _accumulate_training_stats_for_logical(
    train_paths: List[str],
    extractor: FeatureExtractor,
    args,
    layers: List[int],
    grouped_layers: List[List[int]],
    cluster_model: Dict[str, Any],
) -> Dict[str, Any]:
    """Second pass: estimate counts/position/adjacency/feature stats per part."""
    mbk = cluster_model["kmeans"]
    centers = cluster_model["centers"]
    K = cluster_model["K"]

    # --- Per-part accumulators ---
    # Counts (per image)
    counts_sum = np.zeros(K, dtype=np.float64)
    counts_sq_sum = np.zeros(K, dtype=np.float64)

    # Positions (per image, per part)
    pos_count = np.zeros(K, dtype=np.int64)
    pos_sum = np.zeros((K, 2), dtype=np.float64)
    pos_outer_sum = np.zeros((K, 2, 2), dtype=np.float64)

    # Features (per image, per part)
    feat_count = np.zeros(K, dtype=np.int64)
    feat_sum = None
    feat_sq_sum = None

    # Adjacency
    adj_counts = np.zeros((K, K), dtype=np.float64)

    # For feature arrays, we only know dim after seeing first image
    feat_dim = cluster_model["feat_dim"]

    feat_sum = np.zeros((K, feat_dim), dtype=np.float64)
    feat_sq_sum = np.zeros((K, feat_dim), dtype=np.float64)

    num_images = 0

    logging.info(
        f"[Logical] Accumulating counts/pos/adj/feature stats over {len(train_paths)} images..."
    )

    for path in train_paths:
        _, tokens_spatial, (h_p, w_p) = _collect_tokens_for_image(
            path,
            extractor,
            args.image_res,
            layers,
            args.agg_method,
            grouped_layers,
            args.docrop,
            args.use_clahe,
            args.dino_saliency_layer,
        )
        h, w, C = tokens_spatial.shape
        feats = tokens_spatial.reshape(-1, C).astype(np.float32)

        # Cluster assignments
        labels = mbk.predict(feats)  # [N]
        labels = labels.astype(np.int64)
        num_images += 1

        # --- Counts ---
        counts = np.bincount(labels, minlength=K).astype(np.float64)
        counts_sum += counts
        counts_sq_sum += counts**2

        # --- Positions ---
        # grid coords in [0,1]
        ys = np.repeat(np.arange(h), w)
        xs = np.tile(np.arange(w), h)
        xs_norm = xs / max(w - 1, 1)
        ys_norm = ys / max(h - 1, 1)

        for c in range(K):
            mask = labels == c
            if not np.any(mask):
                continue
            xs_c = xs_norm[mask]
            ys_c = ys_norm[mask]
            u = np.array([xs_c.mean(), ys_c.mean()], dtype=np.float64)

            pos_count[c] += 1
            pos_sum[c] += u
            pos_outer_sum[c] += np.outer(u, u)

            # --- Per-part feature prototype for this image ---
            feats_c = feats[mask]
            z = feats_c.mean(axis=0).astype(np.float64)
            feat_count[c] += 1
            feat_sum[c] += z
            feat_sq_sum[c] += z**2

        # --- Adjacency ---
        label_map = labels.reshape(h, w)

        # horizontal adjacency
        if w > 1:
            left = label_map[:, :-1].flatten()
            right = label_map[:, 1:].flatten()
            np.add.at(adj_counts, (left, right), 1.0)
            np.add.at(adj_counts, (right, left), 1.0)

        # vertical adjacency
        if h > 1:
            up = label_map[:-1, :].flatten()
            down = label_map[1:, :].flatten()
            np.add.at(adj_counts, (up, down), 1.0)
            np.add.at(adj_counts, (down, up), 1.0)

    if num_images == 0:
        logging.warning("[Logical] No training images for logical model.")
        return {}

    # --- Finalize statistics ---

    # Counts
    counts_mean = counts_sum / float(num_images)
    counts_var = counts_sq_sum / float(num_images) - counts_mean**2
    counts_var = np.maximum(counts_var, EPS)
    counts_std = np.sqrt(counts_var)

    # Positions (per part)
    pos_mu = np.zeros((K, 2), dtype=np.float64)
    pos_cov_inv = np.zeros((K, 2, 2), dtype=np.float64)

    for c in range(K):
        if pos_count[c] <= 1:
            pos_mu[c] = np.array([0.5, 0.5], dtype=np.float64)
            pos_cov_inv[c] = np.eye(2, dtype=np.float64)
            continue

        mu_c = pos_sum[c] / float(pos_count[c])
        outer_mean = pos_outer_sum[c] / float(pos_count[c])
        cov_c = outer_mean - np.outer(mu_c, mu_c)
        cov_c += np.eye(2, dtype=np.float64) * 1e-4  # regularize

        try:
            cov_inv = np.linalg.inv(cov_c)
        except np.linalg.LinAlgError:
            logging.warning(f"[Logical] Covariance for part {c} not invertible. Using identity.")
            cov_inv = np.eye(2, dtype=np.float64)

        pos_mu[c] = mu_c
        pos_cov_inv[c] = cov_inv

    # Features (per part)
    feat_mu = np.zeros((K, feat_dim), dtype=np.float64)
    feat_var_inv = np.ones((K, feat_dim), dtype=np.float64)

    for c in range(K):
        if feat_count[c] <= 1:
            # Leave mu=0, var_inv=1
            continue
        mu_c = feat_sum[c] / float(feat_count[c])
        var_c = feat_sq_sum[c] / float(feat_count[c]) - mu_c**2
        var_c = np.maximum(var_c, 1e-6)
        feat_mu[c] = mu_c
        feat_var_inv[c] = 1.0 / var_c

    # Adjacency probabilities
    adj_probs = np.zeros((K, K), dtype=np.float64)
    row_sums = adj_counts.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    adj_probs[nonzero] = adj_counts[nonzero] / row_sums[nonzero]
    # For zero rows, assign uniform distribution
    adj_probs[~nonzero] = 1.0 / K

    logging.info("[Logical] Training stats for logical branch computed.")

    return {
        "centers": centers,
        "K": K,
        "feat_dim": feat_dim,
        "counts_mean": counts_mean,
        "counts_std": counts_std,
        "pos_mu": pos_mu,
        "pos_cov_inv": pos_cov_inv,
        "feat_mu": feat_mu,
        "feat_var_inv": feat_var_inv,
        "adj_probs": adj_probs,
        "eps": EPS,
    }


def fit_logical_model(
    train_paths: List[str],
    extractor: FeatureExtractor,
    args,
    layers: List[int],
    grouped_layers: List[List[int]],
) -> Dict[str, Any]:
    """
    Top-level entry: builds a training-free logical model
    (part clustering + composition statistics) from k-shot normals.
    """
    if not train_paths:
        logging.warning("[Logical] Empty train_paths; skipping logical model.")
        return {}

    if args.patch_size is not None:
        logging.warning(
            "[Logical] Logical composition branch currently only supports full-image mode "
            "(--patch_size None). Disabling logical branch."
        )
        return {}

    cluster_model = _fit_token_clusters(
        train_paths, extractor, args, layers, grouped_layers
    )
    if not cluster_model:
        return {}

    stats = _accumulate_training_stats_for_logical(
        train_paths, extractor, args, layers, grouped_layers, cluster_model
    )
    if not stats:
        return {}

    # Store weights from CLI (for fusion)
    stats["weights"] = {
        "local": float(args.logical_w_local),
        "count": float(args.logical_w_count),
        "pos": float(args.logical_w_pos),
        "adj": float(args.logical_w_adj),
        "feat": float(args.logical_w_feat),
    }

    logging.info("[Logical] Logical model successfully fitted.")
    return stats


def _assign_clusters(feats: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-center assignment using squared Euclidean distance."""
    # feats: [N, C], centers: [K, C]
    # dist^2 = ||x - c||^2
    diff = feats[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff**2, axis=2)  # [N, K]
    labels = np.argmin(dist2, axis=1)
    return labels.astype(np.int64)


def compute_logical_scores_for_image(
    tokens_spatial: np.ndarray,
    logical_model: Dict[str, Any],
) -> Dict[str, float]:
    """
    Computes logical anomaly scores (counts, positions, adjacency, feature-consistency)
    for a single image given its spatial tokens [H_p, W_p, C].
    """
    if not logical_model:
        return {
            "S_count": 0.0,
            "S_pos": 0.0,
            "S_adj": 0.0,
            "S_feat": 0.0,
        }

    centers = logical_model["centers"]
    K = logical_model["K"]
    eps = logical_model.get("eps", EPS)

    h, w, C = tokens_spatial.shape
    feats = tokens_spatial.reshape(-1, C).astype(np.float64)

    labels = _assign_clusters(feats.astype(np.float32), centers.astype(np.float32))
    labels = labels.astype(np.int64)

    # --- Counts ---
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    mu_c = logical_model["counts_mean"]
    std_c = logical_model["counts_std"]

    valid = std_c > eps
    z = np.zeros_like(counts)
    z[valid] = (counts[valid] - mu_c[valid]) / (std_c[valid] + eps)
    S_count = float(np.sum(z[valid] ** 2)) if np.any(valid) else 0.0

    # --- Positions ---
    ys = np.repeat(np.arange(h), w)
    xs = np.tile(np.arange(w), h)
    xs_norm = xs / max(w - 1, 1)
    ys_norm = ys / max(h - 1, 1)

    pos_mu = logical_model["pos_mu"]
    pos_cov_inv = logical_model["pos_cov_inv"]

    S_pos_sum = 0.0
    n_pos_terms = 0

    for c in range(K):
        mask = labels == c
        if not np.any(mask):
            continue
        xs_c = xs_norm[mask]
        ys_c = ys_norm[mask]
        u = np.array([xs_c.mean(), ys_c.mean()], dtype=np.float64)

        mu = pos_mu[c]
        cov_inv = pos_cov_inv[c]
        diff_vec = u - mu
        val = diff_vec @ cov_inv @ diff_vec
        S_pos_sum += float(val)
        n_pos_terms += 1

    S_pos = S_pos_sum / max(n_pos_terms, 1)

    # --- Adjacency ---
    label_map = labels.reshape(h, w)

    adj_counts_img = np.zeros((K, K), dtype=np.float64)
    if w > 1:
        left = label_map[:, :-1].flatten()
        right = label_map[:, 1:].flatten()
        np.add.at(adj_counts_img, (left, right), 1.0)
        np.add.at(adj_counts_img, (right, left), 1.0)
    if h > 1:
        up = label_map[:-1, :].flatten()
        down = label_map[1:, :].flatten()
        np.add.at(adj_counts_img, (up, down), 1.0)
        np.add.at(adj_counts_img, (down, up), 1.0)

    adj_probs_base = logical_model["adj_probs"]
    adj_probs_img = np.zeros((K, K), dtype=np.float64)
    row_sums = adj_counts_img.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    adj_probs_img[nonzero] = adj_counts_img[nonzero] / row_sums[nonzero]

    diff_adj = adj_probs_img - adj_probs_base
    S_adj = float(np.sum((diff_adj**2) / (adj_probs_base + eps)))

    # --- Feature consistency ---
    feat_mu = logical_model["feat_mu"]
    feat_var_inv = logical_model["feat_var_inv"]

    S_feat_sum = 0.0
    n_feat_terms = 0

    for c in range(K):
        mask = labels == c
        if not np.any(mask):
            continue
        feats_c = feats[mask]
        z_c = feats_c.mean(axis=0)
        mu_c_part = feat_mu[c]
        inv_var_c = feat_var_inv[c]

        diff_vec = z_c - mu_c_part
        val = np.sum((diff_vec**2) * inv_var_c)
        S_feat_sum += float(val)
        n_feat_terms += 1

    S_feat = S_feat_sum / max(n_feat_terms, 1)

    return {
        "S_count": S_count,
        "S_pos": S_pos,
        "S_adj": S_adj,
        "S_feat": S_feat,
    }


def compute_combined_logical_score(
    tokens_spatial: np.ndarray,
    logical_model: Dict[str, Any],
) -> float:
    """
    Compute the scalar logical-composition image score S_log, i.e.,
    the weighted sum over count / pos / adjacency / feature-consistency
    terms using logical_model["weights"] (excluding the local weight).
    """
    if not logical_model:
        return 0.0

    logical_scores = compute_logical_scores_for_image(tokens_spatial, logical_model)
    w = logical_model.get("weights", {})
    w_count = w.get("count", 1.0)
    w_pos = w.get("pos", 1.0)
    w_adj = w.get("adj", 1.0)
    w_feat = w.get("feat", 1.0)

    s_log = (
        w_count * logical_scores["S_count"]
        + w_pos * logical_scores["S_pos"]
        + w_adj * logical_scores["S_adj"]
        + w_feat * logical_scores["S_feat"]
    )
    return float(s_log)


def fuse_image_score(
    local_score: float,
    tokens_spatial: np.ndarray,
    logical_model: Dict[str, Any],
) -> float:
    """
    Fuse local SubspaceAD score with logical scores.

    If logical_model["fusion_state"] is present, we:
      - z-normalize texture and logical scores using validation stats:
            z_tex = (local - mu_tex) / (sigma_tex + eps)
            z_log = (S_log - mu_log) / (sigma_log + eps)
      - fuse in standardized space:
            fused = w_local * z_tex + z_log

    Otherwise, we fall back to the original unnormalized fusion:
            fused = w_local * local + S_log
    """
    if not logical_model:
        return float(local_score)

    # Combined logical branch score S_log (already weighted by count/pos/adj/feat)
    s_log = compute_combined_logical_score(tokens_spatial, logical_model)

    w = logical_model.get("weights", {})
    w_local = w.get("local", 1.0)

    fusion_state = logical_model.get("fusion_state", None)

    if fusion_state is not None:
        tex_mu = float(fusion_state.get("tex_mu", 0.0))
        tex_std = float(fusion_state.get("tex_std", 1.0))
        log_mu = float(fusion_state.get("log_mu", 0.0))
        log_std = float(fusion_state.get("log_std", 1.0))

        # Guard against degenerate std
        tex_std = tex_std if tex_std > 0.0 else 1.0
        log_std = log_std if log_std > 0.0 else 1.0

        z_tex = (float(local_score) - tex_mu) / (tex_std + EPS)
        z_log = (float(s_log) - log_mu) / (log_std + EPS)

        fused = w_local * z_tex + z_log
    else:
        # Fallback: original linear fusion in raw-score space
        fused = w_local * float(local_score) + s_log

    return float(fused)
