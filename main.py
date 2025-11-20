import os
import math
import logging
import time
from pathlib import Path
import numpy as np
import random
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
)
from sklearn.decomposition import PCA
from anomalib.metrics.aupro import _AUPRO as TM_AUPRO

from args import get_args, parse_layer_indices, parse_grouped_layers
from utils import (
    setup_logging,
    save_config,
    min_max_norm,
    pick_threshold_with_fallback,
    generate_run_name,
)
from dataclass import get_dataset_handler
from features import FeatureExtractor
from pca import PCAModel, KernelPCAModel
from score import calculate_anomaly_scores, post_process_map, aggregate_image_score
from viz import save_visualization, save_overlay_for_intro
from specular import specular_mask_torch, filter_specular_anomalies
from patching import process_image_patched, get_patch_coords
from augmentations import get_augmentation_transform

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Logical composition branch
from logical import (
    fit_logical_model,
    fuse_image_score,
    compute_combined_logical_score,
)


def fit_pca_model(train_paths, extractor, aug_transform, args, layers, grouped_layers):
    """
    Fits a PCA (or KernelPCA) model on features from the training set.
    Returns PCA parameters and feature map dimensions.
    """
    if args.patch_size:
        if args.bg_mask_method == "pca_normality":
            raise ValueError("pca_normality mask is not compatible with --patch_size.")

        # Get feature dimensions from a sample patch
        temp_img = Image.open(train_paths[0]).convert("RGB")
        temp_patch = temp_img.crop((0, 0, args.patch_size, args.patch_size))
        temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
            [temp_patch],
            args.image_res,
            layers,
            args.agg_method,
            grouped_layers,
            args.docrop,
            use_clahe=args.use_clahe,
            dino_saliency_layer=args.dino_saliency_layer,
        )
        feature_dim = temp_tokens.shape[-1]
        tokens_per_patch = h_p * w_p

        # Calculate total tokens for PCA streaming
        total_patches = 0
        num_batches = 0
        num_aug_multiplier = (1 + args.aug_count) if aug_transform else 1

        for path in train_paths:
            img = Image.open(path).convert("RGB")
            patch_coords = get_patch_coords(
                img.height, img.width, args.patch_size, args.patch_overlap
            )
            total_patches += len(patch_coords) * num_aug_multiplier
            num_batches += (
                math.ceil(len(patch_coords) / args.batch_size) * num_aug_multiplier
            )
        total_tokens = total_patches * tokens_per_patch

        logging.info(
            f"Feature dim: {feature_dim}, Tokens per patch: {tokens_per_patch}, "
            f"Total train patches (w/ aug): {total_patches}, Total train tokens: {total_tokens}"
        )

        def feature_generator_patched():
            for path in train_paths:
                pil_img = Image.open(path).convert("RGB")
                images_to_process = [pil_img]
                if aug_transform:
                    for _ in range(args.aug_count):
                        images_to_process.append(aug_transform(pil_img))

                for img in images_to_process:
                    patch_coords = get_patch_coords(
                        img.height, img.width, args.patch_size, args.patch_overlap
                    )
                    for i in range(0, len(patch_coords), args.batch_size):
                        coord_batch = patch_coords[i : i + args.batch_size]
                        patch_batch = [img.crop(c) for c in coord_batch]
                        (
                            tokens_batch,
                            _,
                            saliency_masks_batch,
                        ) = extractor.extract_tokens(
                            patch_batch,
                            args.image_res,
                            layers,
                            args.agg_method,
                            grouped_layers,
                            args.docrop,
                            use_clahe=args.use_clahe,
                            dino_saliency_layer=args.dino_saliency_layer,
                        )
                        tokens_flat = tokens_batch.reshape(-1, feature_dim)

                        if args.bg_mask_method == "dino_saliency":
                            masks_flat = saliency_masks_batch.reshape(-1)
                            try:
                                if args.mask_threshold_method == "percentile":
                                    threshold = np.percentile(
                                        masks_flat, args.percentile_threshold * 100
                                    )
                                    foreground_tokens = tokens_flat[
                                        masks_flat >= threshold
                                    ]
                                else:  # otsu
                                    norm_mask = cv2.normalize(
                                        masks_flat,
                                        None,
                                        0,
                                        255,
                                        cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U,
                                    )
                                    _, binary_mask = cv2.threshold(
                                        norm_mask,
                                        0,
                                        255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                    )
                                    foreground_tokens = tokens_flat[
                                        binary_mask.flatten() > 0
                                    ]

                                if foreground_tokens.shape[0] > 0:
                                    yield foreground_tokens
                                else:
                                    yield tokens_flat
                            except Exception:
                                yield tokens_flat
                        else:
                            yield tokens_flat

        feature_generator = feature_generator_patched

    else:  # Full-image PCA
        temp_img = Image.open(train_paths[0]).convert("RGB")
        temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
            [temp_img],
            args.image_res,
            layers,
            args.agg_method,
            grouped_layers,
            args.docrop,
            use_clahe=args.use_clahe,
            dino_saliency_layer=args.dino_saliency_layer,
        )
        feature_dim = temp_tokens.shape[-1]
        num_aug_multiplier = (1 + args.aug_count) if aug_transform else 1
        total_train_images = len(train_paths) * num_aug_multiplier
        total_tokens = total_train_images * h_p * w_p

        logging.info(
            f"Feature dim: {feature_dim}, Tokens per image: {h_p * w_p}, "
            f"Total train images (w/ aug): {total_train_images}, Total train tokens: {total_tokens}"
        )

        def feature_generator_full():
            all_imgs_to_process = []
            for path in train_paths:
                pil_img = Image.open(path).convert("RGB")
                all_imgs_to_process.append(pil_img)
                if aug_transform:
                    for _ in range(args.aug_count):
                        all_imgs_to_process.append(aug_transform(pil_img))

            for i in range(0, len(all_imgs_to_process), args.batch_size):
                img_batch = all_imgs_to_process[i : i + args.batch_size]
                (
                    tokens_batch,
                    _,
                    saliency_masks_batch,
                ) = extractor.extract_tokens(
                    img_batch,
                    args.image_res,
                    layers,
                    args.agg_method,
                    grouped_layers,
                    args.docrop,
                    use_clahe=args.use_clahe,
                    dino_saliency_layer=args.dino_saliency_layer,
                )
                tokens_flat = tokens_batch.reshape(-1, feature_dim)

                if args.bg_mask_method == "dino_saliency":
                    masks_flat = saliency_masks_batch.reshape(-1)
                    try:
                        if args.mask_threshold_method == "percentile":
                            threshold = np.percentile(
                                masks_flat, args.percentile_threshold * 100
                            )
                            foreground_tokens = tokens_flat[masks_flat >= threshold]
                        else:  # otsu
                            norm_mask = cv2.normalize(
                                masks_flat,
                                None,
                                0,
                                255,
                                cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U,
                            )
                            _, binary_mask = cv2.threshold(
                                norm_mask,
                                0,
                                255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                            )
                            foreground_tokens = tokens_flat[binary_mask.flatten() > 0]

                        if foreground_tokens.shape[0] > 0:
                            yield foreground_tokens
                        else:
                            yield tokens_flat
                    except Exception:
                        yield tokens_flat
                else:
                    yield tokens_flat

        num_batches = math.ceil(total_train_images / args.batch_size)
        feature_generator = feature_generator_full

    # --- Fit the chosen PCA model ---
    if args.use_kernel_pca:
        if args.bg_mask_method == "pca_normality":
            raise ValueError("pca_normality mask is not compatible with Kernel PCA.")

        logging.info("Collecting all features for Kernel PCA...")
        all_train_tokens = np.concatenate(
            list(
                tqdm(
                    feature_generator(),
                    desc="Feature Collection",
                    total=num_batches,
                )
            )
        )
        pca_model = KernelPCAModel(
            k=args.pca_dim,
            kernel=args.kernel_pca_kernel,
            gamma=args.kernel_pca_gamma,
        )
        pca_params = pca_model.fit(all_train_tokens)
    else:
        pca_model = PCAModel(k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten)
        pca_params = pca_model.fit(
            feature_generator,
            feature_dim,
            total_tokens,
            num_batches,
        )

    return pca_params, h_p, w_p, feature_dim


def collect_branch_scores(
    paths,
    extractor,
    pca_params,
    args,
    h_p,
    w_p,
    feature_dim,
    category,
    logical_model,
):
    """
    One validation pass to collect:
      - texture (PCA/SubspaceAD) image scores
      - logical-composition image scores

    Used to compute z-normalization stats for fusion.
    Only meaningful in full-image mode with a non-empty logical_model.
    """
    tex_scores = []
    log_scores = []

    if not paths:
        return tex_scores, log_scores

    if not logical_model:
        logging.warning(
            f"[Logical] collect_branch_scores called for {category} but logical_model is empty."
        )
        return tex_scores, log_scores

    if args.patch_size is not None:
        logging.warning(
            f"[Logical] collect_branch_scores called with patch mode for {category}, "
            "but logical branch only supports full-image mode. Skipping logical stats."
        )
        return tex_scores, log_scores

    logging.info(
        f"[Logical] Collecting branch scores on {len(paths)} validation images for {category}..."
    )
    eval_iter = tqdm(paths, desc=f"Collecting branch scores {category}")

    layers = parse_layer_indices(args.layers)
    grouped_layers = (
        parse_grouped_layers(args.grouped_layers) if args.agg_method == "group" else []
    )

    for i in range(0, len(paths), args.batch_size):
        path_batch = paths[i : i + args.batch_size]
        pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]

        # Full-image feature extraction
        (
            tokens,
            (h_p, w_p),
            saliency_masks_batch,
        ) = extractor.extract_tokens(
            pil_imgs,
            args.image_res,
            layers,
            args.agg_method,
            grouped_layers,
            args.docrop,
            use_clahe=args.use_clahe,
            dino_saliency_layer=args.dino_saliency_layer,
        )
        b, _, _, c = tokens.shape
        tokens_reshaped = tokens.reshape(b * h_p * w_p, c)

        scores = calculate_anomaly_scores(
            tokens_reshaped, pca_params, args.score_method, args.drop_k
        )
        anomaly_maps = scores.reshape(b, h_p, w_p)

        # Background mask logic (same as in run_evaluation)
        mask_for_viz = None
        background_mask = np.zeros_like(anomaly_maps, dtype=bool)

        if args.bg_mask_method == "dino_saliency":
            mask_for_viz = saliency_masks_batch
            for j in range(b):
                saliency_map = saliency_masks_batch[j]
                try:
                    if args.mask_threshold_method == "percentile":
                        threshold = np.percentile(
                            saliency_map, args.percentile_threshold * 100
                        )
                        background_mask[j] = saliency_map < threshold
                    else:  # otsu
                        norm_mask = cv2.normalize(
                            saliency_map,
                            None,
                            0,
                            255,
                            cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U,
                        )
                        _, binary_mask = cv2.threshold(
                            norm_mask,
                            0,
                            255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                        )
                        background_mask[j] = binary_mask == 0
                except Exception as e:
                    logging.warning(f"[Logical] Saliency mask failed (val) img {j}: {e}")

        elif args.bg_mask_method == "pca_normality":
            threshold = 10.0
            kernel_size = 3
            border = 0.2
            grid_size = (h_p, w_p)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_for_viz = np.zeros_like(anomaly_maps)

            for j in range(b):
                img_features = tokens[j].reshape(-1, c)
                try:
                    pca = PCA(n_components=1, svd_solver="randomized")
                    first_pc = pca.fit_transform(img_features.astype(np.float32))
                    mask = first_pc > threshold
                    mask_2d = mask.reshape(grid_size)

                    h_start, h_end = int(grid_size[0] * border), int(
                        grid_size[0] * (1 - border)
                    )
                    w_start, w_end = int(grid_size[1] * border), int(
                        grid_size[1] * (1 - border)
                    )
                    m = mask_2d[h_start:h_end, w_start:w_end]

                    if m.sum() <= m.size * 0.35:
                        mask = -first_pc > threshold
                        mask_2d = mask.reshape(grid_size)

                    mask_processed = cv2.dilate(
                        mask_2d.astype(np.uint8), kernel
                    ).astype(bool)
                    mask_processed = cv2.morphologyEx(
                        mask_processed.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                    ).astype(bool)

                    background_mask[j] = ~mask_processed
                    mask_for_viz[j] = mask_processed.astype(np.float32)
                except Exception as e:
                    logging.warning(f"[Logical] PCA mask failed (val) img {j}: {e}")

        anomaly_maps[background_mask] = 0.0

        # Per-image scores
        for j in range(anomaly_maps.shape[0]):
            anomaly_map_pre_specular = post_process_map(
                anomaly_maps[j], args.image_res
            )
            anomaly_map_final = anomaly_map_pre_specular

            if args.use_specular_filter:
                img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                _, _, conf = specular_mask_torch(img_tensor, tau=args.specular_tau)
                conf = torch.nn.functional.interpolate(
                    conf,
                    size=anomaly_map_final.shape,
                    mode="bilinear",
                    align_corners=False,
                )
                conf_map = conf.squeeze().cpu().numpy()
                anomaly_map_final = (
                    filter_specular_anomalies(anomaly_map_final, conf_map)
                    .cpu()
                    .numpy()
                )

            # Texture / PCA image score
            local_img_score = aggregate_image_score(
                anomaly_map_final, args.img_score_agg
            )
            tex_scores.append(float(local_img_score))

            # Logical-composition image score
            tokens_spatial_j = tokens[j]  # [h_p, w_p, C]
            try:
                s_log = compute_combined_logical_score(
                    tokens_spatial_j,
                    logical_model,
                )
                log_scores.append(float(s_log))
            except Exception as e:
                logging.warning(
                    f"[Logical] Failed to compute logical score for val image "
                    f"{path_batch[j]}: {e}"
                )

        eval_iter.update(len(path_batch))

    return tex_scores, log_scores


def run_evaluation(
    paths,
    handler,
    extractor,
    pca_params,
    args,
    h_p,
    w_p,
    feature_dim,
    category,
    is_test_run=False,
    thr_img=None,
    thr_px=None,
    logical_model=None,
):
    """
    Runs evaluation on a set of images (validation or test).
    Returns a dictionary of results.
    """
    results = {
        "img_labels": [],
        "img_scores_f1": [],
        "img_scores_auroc": [],
        "px_labels_all": [],
        "px_scores_auroc_all": [],
        "px_scores_norm_all": [],
        "anom_gt_masks": [],
        "anom_maps_norm": [],
        "inference_times": [],
    }
    vis_saved_count = 0

    if not paths:
        return results

    desc = f"Testing {category}" if is_test_run else f"Validating {category}"
    eval_iter = tqdm(paths, desc=desc)

    for i in range(0, len(paths), args.batch_size):
        path_batch = paths[i : i + args.batch_size]
        pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
        is_anomaly_batch = [
            "good" not in str(p) and "Normal" not in str(p) for p in path_batch
        ]

        if is_test_run:
            torch.cuda.synchronize(DEVICE)
            start_time = time.perf_counter()

        # --- 1. COMPUTATION STAGE (Image -> Final Anomaly Map) ---
        final_anomaly_maps_for_batch = []
        saliency_maps_for_viz_batch = None
        tokens_batch_for_logical = None  # for logical fusion (full-image only)

        if args.patch_size:
            (
                anomaly_maps_batch,
                saliency_maps_batch,
            ) = process_image_patched(
                pil_imgs, extractor, pca_params, args, DEVICE, h_p, w_p, feature_dim
            )
            saliency_maps_for_viz_batch = saliency_maps_batch

            for j, anomaly_map_pre_specular in enumerate(anomaly_maps_batch):
                anomaly_map_final = anomaly_map_pre_specular
                if args.use_specular_filter:
                    img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                    _, _, conf = specular_mask_torch(img_tensor, tau=args.specular_tau)
                    conf = torch.nn.functional.interpolate(
                        conf,
                        size=anomaly_map_pre_specular.shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                    conf_map = conf.squeeze().cpu().numpy()
                    anomaly_map_final = (
                        filter_specular_anomalies(anomaly_map_pre_specular, conf_map)
                        .cpu()
                        .numpy()
                    )
                final_anomaly_maps_for_batch.append(anomaly_map_final)

        else:  # Full-image mode
            (
                tokens,
                (h_p, w_p),
                saliency_masks_batch,
            ) = extractor.extract_tokens(
                pil_imgs,
                args.image_res,
                parse_layer_indices(args.layers),
                args.agg_method,
                parse_grouped_layers(args.grouped_layers)
                if args.agg_method == "group"
                else [],
                args.docrop,
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )
            b, _, _, c = tokens.shape
            tokens_reshaped = tokens.reshape(b * h_p * w_p, c)

            scores = calculate_anomaly_scores(
                tokens_reshaped, pca_params, args.score_method, args.drop_k
            )
            anomaly_maps = scores.reshape(b, h_p, w_p)

            mask_for_viz = None
            background_mask = np.zeros_like(anomaly_maps, dtype=bool)

            if args.bg_mask_method == "dino_saliency":
                mask_for_viz = saliency_masks_batch
                for j in range(b):
                    saliency_map = saliency_masks_batch[j]
                    try:
                        if args.mask_threshold_method == "percentile":
                            threshold = np.percentile(
                                saliency_map, args.percentile_threshold * 100
                            )
                            background_mask[j] = saliency_map < threshold
                        else:  # otsu
                            norm_mask = cv2.normalize(
                                saliency_map,
                                None,
                                0,
                                255,
                                cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U,
                            )
                            _, binary_mask = cv2.threshold(
                                norm_mask,
                                0,
                                255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                            )
                            background_mask[j] = binary_mask == 0
                    except Exception as e:
                        logging.warning(f"Saliency mask failed for image {j}: {e}")

            elif args.bg_mask_method == "pca_normality":
                threshold = 10.0
                kernel_size = 3
                border = 0.2
                grid_size = (h_p, w_p)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_for_viz = np.zeros_like(anomaly_maps)

                for j in range(b):
                    img_features = tokens[j].reshape(-1, c)
                    try:
                        pca = PCA(n_components=1, svd_solver="randomized")
                        first_pc = pca.fit_transform(img_features.astype(np.float32))
                        mask = first_pc > threshold
                        mask_2d = mask.reshape(grid_size)

                        h_start, h_end = int(grid_size[0] * border), int(
                            grid_size[0] * (1 - border)
                        )
                        w_start, w_end = int(grid_size[1] * border), int(
                            grid_size[1] * (1 - border)
                        )
                        m = mask_2d[h_start:h_end, w_start:w_end]

                        if m.sum() <= m.size * 0.35:
                            mask = -first_pc > threshold
                            mask_2d = mask.reshape(grid_size)

                        mask_processed = cv2.dilate(
                            mask_2d.astype(np.uint8), kernel
                        ).astype(bool)
                        mask_processed = cv2.morphologyEx(
                            mask_processed.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                        ).astype(bool)

                        background_mask[j] = ~mask_processed
                        mask_for_viz[j] = mask_processed.astype(np.float32)
                    except Exception as e:
                        logging.warning(f"PCA mask failed for image {j}: {e}")

            anomaly_maps[background_mask] = 0.0
            saliency_maps_for_viz_batch = mask_for_viz
            tokens_batch_for_logical = tokens  # [B, h_p, w_p, C]

            for j in range(anomaly_maps.shape[0]):
                anomaly_map_pre_specular = post_process_map(
                    anomaly_maps[j], args.image_res
                )
                anomaly_map_final = anomaly_map_pre_specular
                if args.use_specular_filter:
                    img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                    _, _, conf = specular_mask_torch(
                        img_tensor, tau=args.specular_tau
                    )
                    conf = torch.nn.functional.interpolate(
                        conf,
                        size=anomaly_map_final.shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                    conf_map = conf.squeeze().cpu().numpy()
                    anomaly_map_final = (
                        filter_specular_anomalies(anomaly_map_final, conf_map)
                        .cpu()
                        .numpy()
                    )
                final_anomaly_maps_for_batch.append(anomaly_map_final)

        if is_test_run:
            torch.cuda.synchronize(DEVICE)
            end_time = time.perf_counter()
            results["inference_times"].append(end_time - start_time)

        # --- 2. METRICS & VISUALIZATION STAGE (Not Timed) ---
        for j, anomaly_map_final in enumerate(final_anomaly_maps_for_batch):
            is_anomaly = is_anomaly_batch[j]
            path = path_batch[j]
            pil_img = pil_imgs[j]

            # Local (SubspaceAD) image-level score
            local_img_score = aggregate_image_score(
                anomaly_map_final, args.img_score_agg
            )

            # Fuse with logical composition branch if available (full-image only)
            fused_img_score = local_img_score
            if (logical_model is not None) and bool(logical_model) and (
                tokens_batch_for_logical is not None
            ):
                try:
                    tokens_spatial_j = tokens_batch_for_logical[j]  # [h_p, w_p, C]
                    fused_img_score = fuse_image_score(
                        local_img_score,
                        tokens_spatial_j,
                        logical_model,
                    )
                except Exception as e:
                    logging.warning(
                        f"Logical fusion failed for image {path}: {e}. "
                        "Using local score only."
                    )
                    fused_img_score = local_img_score

            results["img_labels"].append(1 if is_anomaly else 0)
            results["img_scores_auroc"].append(float(fused_img_score))
            if thr_img is not None:
                results["img_scores_f1"].append(1 if fused_img_score >= thr_img else 0)

            anomaly_map_normalized = min_max_norm(anomaly_map_final)
            H, W = anomaly_map_normalized.shape

            # Ground Truth Mask Handling
            if args.patch_size:
                gt_mask = handler.get_ground_truth_mask(path, pil_img.size)
                gt_mask = (
                    np.array(
                        Image.fromarray(
                            (gt_mask.astype(np.uint8) * 255)
                        ).resize((W, H), resample=Image.NEAREST)
                    )
                    > 127
                ).astype(np.uint8)
            else:
                gt_path_str = handler.get_ground_truth_path(path)
                if not gt_path_str or not os.path.exists(gt_path_str):
                    gt_mask = np.zeros((H, W), dtype=np.uint8)
                else:
                    gt_mask_pil = Image.open(gt_path_str).convert("L")
                    if args.docrop:
                        resize_res = int(args.image_res / 0.875)
                        gt_mask_pil = TF.resize(
                            gt_mask_pil,
                            (resize_res, resize_res),
                            interpolation=TF.InterpolationMode.NEAREST,
                        )
                        gt_mask_pil = TF.center_crop(
                            gt_mask_pil, (args.image_res, args.image_res)
                        )
                    gt_mask_pil = TF.resize(
                        gt_mask_pil,
                        (H, W),
                        interpolation=TF.InterpolationMode.NEAREST,
                    )
                    gt_mask = (np.array(gt_mask_pil) > 0).astype(np.uint8)

            results["px_labels_all"].extend(gt_mask.flatten().astype(np.uint8))
            results["px_scores_auroc_all"].extend(
                anomaly_map_final.flatten().astype(np.float32)
            )
            results["px_scores_norm_all"].extend(
                anomaly_map_normalized.flatten().astype(np.float32)
            )

            if is_anomaly:
                results["anom_gt_masks"].append(gt_mask)
                results["anom_maps_norm"].append(anomaly_map_normalized)

                if is_test_run and args.save_intro_overlays:
                    save_overlay_for_intro(
                        path, pil_img, anomaly_map_normalized, args.outdir, category
                    )

                if is_test_run and vis_saved_count < args.vis_count:
                    vis_img = pil_img
                    if args.docrop and not args.patch_size:
                        resize_res = int(args.image_res / 0.875)
                        vis_img = TF.resize(
                            vis_img,
                            (resize_res, resize_res),
                            interpolation=TF.InterpolationMode.BICUBIC,
                        )
                        vis_img = TF.center_crop(
                            vis_img, (args.image_res, args.image_res)
                        )

                    saliency_map_for_viz = None
                    raw_mask_map = (
                        saliency_maps_for_viz_batch[j]
                        if saliency_maps_for_viz_batch is not None
                        else None
                    )

                    if raw_mask_map is not None:
                        try:
                            if args.bg_mask_method == "pca_normality":
                                binary_mask = raw_mask_map
                            elif args.bg_mask_method == "dino_saliency":
                                if args.mask_threshold_method == "percentile":
                                    threshold_val = np.percentile(
                                        raw_mask_map, args.percentile_threshold * 100
                                    )
                                    binary_mask = (
                                        raw_mask_map >= threshold_val
                                    ).astype(np.float32)
                                else:  # otsu
                                    norm_mask = cv2.normalize(
                                        raw_mask_map,
                                        None,
                                        0,
                                        255,
                                        cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U,
                                    )
                                    _, binary_mask_u8 = cv2.threshold(
                                        norm_mask,
                                        0,
                                        255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                    )
                                    binary_mask = (binary_mask_u8 > 0).astype(
                                        np.float32
                                    )

                            saliency_map_for_viz = post_process_map(
                                binary_mask, anomaly_map_normalized.shape, blur=False
                            )
                        except Exception as e:
                            logging.warning(f"Saliency viz processing failed: {e}")

                    save_visualization(
                        path,
                        vis_img,
                        gt_mask,
                        anomaly_map_normalized,
                        args.outdir,
                        category,
                        vis_saved_count,
                        saliency_mask=saliency_map_for_viz,
                    )
                    vis_saved_count += 1

        eval_iter.update(len(path_batch))

    return results


def main():
    args = get_args()

    # --- Setup ---
    run_name = generate_run_name(args)
    args.outdir = os.path.join(args.outdir, run_name)
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir, not args.no_log_file)
    save_config(args)

    # --- Initialize Augmentation ---
    aug_transform = None
    if args.k_shot is not None and args.aug_count > 0 and args.aug_list:
        aug_transform = get_augmentation_transform(args.aug_list, args.image_res)
        if not aug_transform.transforms:
            aug_transform = None

    # --- Parse layer arguments ---
    layers = parse_layer_indices(args.layers)
    grouped_layers = (
        parse_grouped_layers(args.grouped_layers) if args.agg_method == "group" else []
    )

    # --- Initialize model ---
    extractor = FeatureExtractor(args.model_ckpt)

    # --- Get categories ---
    if args.categories:
        categories = args.categories
    else:
        categories = sorted(
            [
                f.name
                for f in Path(args.dataset_path).iterdir()
                if f.is_dir() and f.name != "split_csv"
            ]
        )

    # --- Main Loop ---
    all_results_df = pd.DataFrame(
        columns=[
            "Category",
            "Image AUROC",
            "Image AUPR",
            "Pixel AUROC",
            "AU-PRO",
            "Image F1",
            "Pixel F1",
        ]
    )

    for category in categories:
        logging.info(f"--- Processing Category: {category} ---")
        handler = get_dataset_handler(args.dataset_name, args.dataset_path, category)
        train_paths = handler.get_train_paths()
        val_paths = handler.get_validation_paths()
        test_paths = handler.get_test_paths()

        current_aug_transform = aug_transform
        if category.lower() in args.no_aug_categories:
            if current_aug_transform is not None:
                logging.info(f"Disabling augmentations for the '{category}' category.")
            current_aug_transform = None

        if args.debug_limit is not None:
            logging.warning(f"--- DEBUG MODE: Limiting to {args.debug_limit} images ---")
            if val_paths:
                val_paths = val_paths[: args.debug_limit]
            if test_paths:
                test_paths = test_paths[: args.debug_limit]

        if not train_paths:
            logging.warning(f"No training images found for {category}. Skipping.")
            continue

        if args.batched_zero_shot:
            logging.info(
                f"--- Batched 0-Shot Mode: Fitting PCA on {len(test_paths)} test images ---"
            )
            train_paths = test_paths.copy()
            val_paths = None

        if args.k_shot is not None:
            logging.info(f"--- K-SHOT: Sampling {args.k_shot} training images ---")
            random.shuffle(train_paths)
            train_paths = train_paths[: args.k_shot]

        # 1. Fit PCA Model
        pca_params, h_p, w_p, feature_dim = fit_pca_model(
            train_paths, extractor, current_aug_transform, args, layers, grouped_layers
        )

        # 1.5 Fit Logical Composition Model (optional, full-image only)
        logical_model = None
        if getattr(args, "use_logical_branch", False):
            logical_model = fit_logical_model(
                train_paths,
                extractor,
                args,
                layers,
                grouped_layers,
            )
        else:
            logical_model = None

        # 1.75 Compute fusion (z-score) stats on validation set for logical branch
        if logical_model and val_paths:
            tex_scores_val, log_scores_val = collect_branch_scores(
                val_paths,
                extractor,
                pca_params,
                args,
                h_p,
                w_p,
                feature_dim,
                category,
                logical_model,
            )
            if len(tex_scores_val) > 0 and len(log_scores_val) > 0:
                tex_scores_arr = np.array(tex_scores_val, dtype=np.float64)
                log_scores_arr = np.array(log_scores_val, dtype=np.float64)

                fusion_state = {
                    "tex_mu": float(tex_scores_arr.mean()),
                    "tex_std": float(tex_scores_arr.std() + 1e-6),
                    "log_mu": float(log_scores_arr.mean()),
                    "log_std": float(log_scores_arr.std() + 1e-6),
                }
                logical_model["fusion_state"] = fusion_state
                logging.info(
                    "[Logical] Fusion stats for %s: "
                    "tex_mu=%.4f tex_std=%.4f log_mu=%.4f log_std=%.4f",
                    category,
                    fusion_state["tex_mu"],
                    fusion_state["tex_std"],
                    fusion_state["log_mu"],
                    fusion_state["log_std"],
                )
            else:
                logging.warning(
                    "[Logical] Could not compute fusion stats for %s (empty scores).",
                    category,
                )

        # 2. Determine PR-optimal F1 thresholds
        thr_img, thr_px = None, None
        if val_paths:
            logging.info(f"Validating on {len(val_paths)} images...")
            val_results = run_evaluation(
                val_paths,
                handler,
                extractor,
                pca_params,
                args,
                h_p,
                w_p,
                feature_dim,
                category,
                is_test_run=False,
                logical_model=logical_model,
            )

            target_img_fpr = getattr(args, "target_img_fpr", 0.05)
            target_px_fpr = getattr(args, "target_px_fpr", 0.05)

            thr_img, how_img = pick_threshold_with_fallback(
                val_results["img_labels"],
                val_results["img_scores_auroc"],
                target_img_fpr,
            )
            thr_px, how_px = pick_threshold_with_fallback(
                val_results["px_labels_all"],
                val_results["px_scores_norm_all"],
                target_px_fpr,
            )
            logging.info(
                f"Chosen thresholds — Image: {thr_img if thr_img is not None else float('nan'):.6g} "
                f"({how_img}), Pixel: {thr_px if thr_px is not None else float('nan'):.6g} ({how_px})"
            )
        else:
            logging.warning("No validation set found. F1 scores will be N/A.")

        # 3. Evaluate on Test Set
        logging.info(f"Evaluating on {len(test_paths)} test images...")
        test_results = run_evaluation(
            test_paths,
            handler,
            extractor,
            pca_params,
            args,
            h_p,
            w_p,
            feature_dim,
            category,
            is_test_run=True,
            thr_img=thr_img,
            thr_px=thr_px,
            logical_model=logical_model,
        )

        # 4. Report Timings
        if test_results["inference_times"]:
            times_arr = np.array(test_results["inference_times"])
            total_time = np.sum(times_arr)
            avg_time_per_image = total_time / len(test_paths)
            logging.info(f"--- Timing Results for {category} ---")
            logging.info(f"Total test images: {len(test_paths)}")
            logging.info(f"Total inference time: {total_time:.4f} s")
            logging.info(f"Avg. time per image: {avg_time_per_image:.6f} s")
            logging.info(f"Images per second (FPS): {1.0 / avg_time_per_image:.2f}")

        # 5. Calculate Metrics
        img_true = test_results["img_labels"]
        img_pred_auroc = test_results["img_scores_auroc"]
        img_auroc = (
            roc_auc_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else np.nan
        )
        img_aupr = (
            average_precision_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else np.nan
        )
        img_f1 = (
            f1_score(img_true, test_results["img_scores_f1"])
            if thr_img is not None
            else np.nan
        )

        px_true_arr = np.array(test_results["px_labels_all"], dtype=np.uint8)
        px_pred_arr_auroc = np.array(test_results["px_scores_auroc_all"])
        px_pred_arr_normalized = np.array(test_results["px_scores_norm_all"])
        has_pos = (px_true_arr == 1).any()
        has_neg = (px_true_arr == 0).any()

        px_auroc = (
            roc_auc_score(px_true_arr, px_pred_arr_auroc)
            if (has_pos and has_neg)
            else np.nan
        )
        px_f1 = (
            f1_score(px_true_arr, (px_pred_arr_normalized >= thr_px).astype(int))
            if (thr_px is not None and has_pos)
            else np.nan
        )

        au_pro = np.nan
        if test_results["anom_gt_masks"]:
            preds_np = np.stack(test_results["anom_maps_norm"]).astype(np.float32)
            gts_np = np.stack(test_results["anom_gt_masks"]).astype(np.uint8)
            preds_t = (
                torch.from_numpy(preds_np)
                .unsqueeze(1)
                .to(torch.float32)
                .to(DEVICE)
            )
            gts_t = (
                torch.from_numpy(gts_np).unsqueeze(1).to(torch.bool).to(DEVICE)
            )

            fpr_cap = getattr(args, "pro_integration_limit", 0.3)
            tm_metric = TM_AUPRO(fpr_limit=fpr_cap).to(DEVICE)
            au_pro = tm_metric(preds_t, gts_t).item()

        # 6. Log and store results
        logging.info(
            f"{category} Results | I-AUROC: {img_auroc:.4f} | I-AUPR: {img_aupr:.4f} | "
            f"P-AUROC: {px_auroc:.4f} | AU-PRO: {au_pro:.4f} | "
            f"I-F1: {img_f1:.4f} | P-F1: {px_f1:.4f}"
        )
        all_results_df.loc[len(all_results_df)] = [
            category,
            img_auroc,
            img_aupr,
            px_auroc,
            au_pro,
            img_f1,
            px_f1,
        ]

    # --- Final Report ---
    if not all_results_df.empty and len(all_results_df) > 1:
        mean_values = all_results_df.mean(numeric_only=True)
        mean_row = pd.DataFrame(
            [["Average"] + mean_values.tolist()], columns=all_results_df.columns
        )
        all_results_df = pd.concat([all_results_df, mean_row], ignore_index=True)

    logging.info("\n--- Benchmark Final Results ---")
    logging.info(
        "\n"
        + all_results_df.to_string(
            index=False, float_format="%.4f", na_rep="N/A"
        )
    )

    results_path = os.path.join(args.outdir, "benchmark_results.csv")
    all_results_df.to_csv(results_path, index=False, float_format="%.4f")
    logging.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()