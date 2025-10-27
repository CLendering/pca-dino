import os
import math
import logging
from pathlib import Path

import numpy as np
import random
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from anomalib.metrics.aupro import _AUPRO as TM_AUPRO

from args import get_args, parse_layer_indices, parse_grouped_layers
from utils import (
    setup_logging,
    save_config,
)
from dataclass import get_dataset_handler
from features import FeatureExtractor
from pca import PCAModel, KernelPCAModel
from score import calculate_anomaly_scores, post_process_map
from viz import save_visualization
from specular import specular_mask_torch, filter_specular_anomalies
from patching import process_image_patched, get_patch_coords
from augmentations import get_augmentation_transform


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


def _best_f1_threshold_from_scores(y_true, y_score):
    """Return threshold maximizing F1 on validation scores."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or y_score.size == 0 or (y_true.max() == y_true.min()):
        return None, 0.0
    p, r, t = precision_recall_curve(y_true, y_score)
    if t.size == 0:
        return None, 0.0
    f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])


def _quantile_threshold_from_negatives(y_true, y_score, target_fpr=0.01):
    """
    Fallback: pick threshold so that ~target_fpr of NEGATIVES exceed it.
    y_true in {0,1}, negatives are 0. Returns None if no negatives.
    """
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    neg = y_score[y_true == 0]
    if neg.size == 0:
        return None
    q = np.clip(1.0 - float(target_fpr), 0.0, 1.0)
    return float(np.quantile(neg, q, interpolation="linear"))


def _pick_threshold_with_fallback(y_true, y_score, target_fpr):
    """
    Try PR-optimal F1; if degenerate (single-class), fall back to negative-quantile.
    Returns (thr, how), where how ∈ {"pr", "quantile", "none"}.
    """
    thr_pr, _ = _best_f1_threshold_from_scores(y_true, y_score)
    if thr_pr is not None:
        return thr_pr, "pr"
    thr_q = _quantile_threshold_from_negatives(y_true, y_score, target_fpr)
    if thr_q is not None:
        return thr_q, "quantile"
    return None, "none"


def main():
    args = get_args()

    # --- Setup ---
    run_name = f"{args.dataset_name}_{args.agg_method}_layers{''.join(args.layers.split(','))}_res{args.image_res}_docrop{int(args.docrop)}"
    if args.patch_size:
        run_name += f"_patch{args.patch_size}"
    if args.use_kernel_pca:
        run_name += f"_kpca-{args.kernel_pca_kernel}"
    if args.use_specular_filter:
        run_name += "_spec-filt"

    run_name += f"_score-{args.score_method}"
    run_name += f"_clahe{int(args.use_clahe)}"
    run_name += f"_dropk{args.drop_k}"

    # Add k-shot and augmentation info to run name
    if args.k_shot is not None:
        run_name += f"_k{args.k_shot}"
        if args.aug_count > 0 and args.aug_list:
            # Create a short string for augs, e.g., "hrc"
            aug_str = "".join(sorted([a[0] for a in args.aug_list]))
            run_name += f"_aug{args.aug_count}x{aug_str}"

    args.outdir = os.path.join(args.outdir, run_name)
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir, not args.no_log_file)
    save_config(args)

    # --- Initialize Augmentation ---
    aug_transform = None
    if args.k_shot is not None and args.aug_count > 0 and args.aug_list:
        aug_transform = get_augmentation_transform(args.aug_list, args.image_res)
        if not aug_transform.transforms:  # Check if any valid transforms were created
            logging.warning(
                "Augmentation specified but no valid transforms were created. Disabling augmentations."
            )
            aug_transform = None  # Disable it

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
            [f.name for f in Path(args.dataset_path).iterdir() if f.is_dir()]
        )

    # --- Main Loop ---
    all_results = []
    for category in categories:
        logging.info(f"--- Processing Category: {category} ---")
        handler = get_dataset_handler(args.dataset_name, args.dataset_path, category)
        train_paths = handler.get_train_paths()
        val_paths = handler.get_validation_paths()
        test_paths = handler.get_test_paths()

        if args.debug_limit is not None:
            logging.warning(
                f"--- DEBUG MODE: Limiting validation and test sets to {args.debug_limit} images ---"
            )
            if val_paths:
                val_paths = val_paths[: args.debug_limit]
            if test_paths:
                test_paths = test_paths[: args.debug_limit]

        if not train_paths:
            logging.warning(f"No training images found for {category}. Skipping.")
            continue

        # --- K-Shot Sampling ---
        if args.k_shot is not None:
            if args.k_shot > len(train_paths):
                logging.warning(
                    f"Requested k_shot={args.k_shot} but only {len(train_paths)} training images available. Using all {len(train_paths)}."
                )
            else:
                logging.info(
                    f"--- K-SHOT: Randomly sampling {args.k_shot} training images ---"
                )
                random.shuffle(train_paths)
                train_paths = train_paths[: args.k_shot]
                for i, path in enumerate(train_paths):
                    logging.info(
                        f"  K-Shot image {i + 1}/{args.k_shot}: {Path(path).name}"
                    )

        # 1. Fit PCA Model
        if args.patch_size:
            # --- PCA on Patches ---
            temp_img = Image.open(train_paths[0]).convert("RGB")
            temp_patch = temp_img.crop((0, 0, args.patch_size, args.patch_size))
            temp_tokens, (h_p, w_p) = extractor.extract_tokens(
                [temp_patch],
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                is_cosine=(args.score_method == "cosine"),
                use_clahe=args.use_clahe,
            )
            feature_dim = temp_tokens.shape[-1]
            tokens_per_patch = h_p * w_p

            # Calculate total number of patches and tokens (with augmentations)
            total_patches = 0
            num_batches = 0
            # This multiplier accounts for the original image + N augmented images
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
                f"Base train patches: {total_patches // num_aug_multiplier}, "
                f"Total train patches (w/ aug): {total_patches}, Total train tokens: {total_tokens}"
            )

            def feature_generator_patched():
                for path in train_paths:
                    pil_img = Image.open(path).convert("RGB")

                    # Create a list of images to process: original + augmentations
                    images_to_process = [pil_img]
                    if aug_transform:
                        for _ in range(args.aug_count):
                            images_to_process.append(aug_transform(pil_img))

                    # Process each image (original + augmented)
                    for img in images_to_process:
                        patch_coords = get_patch_coords(
                            img.height,
                            img.width,
                            args.patch_size,
                            args.patch_overlap,
                        )
                        for i in range(0, len(patch_coords), args.batch_size):
                            coord_batch = patch_coords[i : i + args.batch_size]
                            patch_batch = [img.crop(c) for c in coord_batch]
                            tokens_batch, _ = extractor.extract_tokens(
                                patch_batch,
                                args.image_res,
                                layers,
                                args.agg_method,
                                grouped_layers,
                                args.docrop,
                                is_cosine=(args.score_method == "cosine"),
                                use_clahe=args.use_clahe,
                            )
                            yield tokens_batch.reshape(-1, feature_dim)

            feature_generator = feature_generator_patched

        else:
            # --- PCA on Full Images ---
            temp_img = Image.open(train_paths[0]).convert("RGB")
            temp_tokens, (h_p, w_p) = extractor.extract_tokens(
                [temp_img],
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                is_cosine=(args.score_method == "cosine"),
                use_clahe=args.use_clahe,
            )
            feature_dim = temp_tokens.shape[-1]

            # This multiplier accounts for the original image + N augmented images
            num_aug_multiplier = (1 + args.aug_count) if aug_transform else 1
            total_train_images = len(train_paths) * num_aug_multiplier
            total_tokens = total_train_images * h_p * w_p

            logging.info(
                f"Feature dim: {feature_dim}, Tokens per image: {h_p * w_p}, "
                f"Base train images: {len(train_paths)}, "
                f"Total train images (w/ aug): {total_train_images}, Total train tokens: {total_tokens}"
            )

            def feature_generator_full():
                all_imgs_to_process = []
                for path in train_paths:
                    pil_img = Image.open(path).convert("RGB")
                    all_imgs_to_process.append(pil_img)  # Add original
                    if aug_transform:
                        for _ in range(args.aug_count):
                            # Apply augmentation
                            all_imgs_to_process.append(aug_transform(pil_img))

                # Now process all_imgs_to_process in batches
                for i in range(0, len(all_imgs_to_process), args.batch_size):
                    img_batch = all_imgs_to_process[i : i + args.batch_size]
                    tokens_batch, _ = extractor.extract_tokens(
                        img_batch,
                        args.image_res,
                        layers,
                        args.agg_method,
                        grouped_layers,
                        args.docrop,
                        is_cosine=(args.score_method == "cosine"),
                        use_clahe=args.use_clahe,
                    )
                    yield tokens_batch.reshape(-1, feature_dim)

            num_batches = math.ceil(total_train_images / args.batch_size)
            feature_generator = feature_generator_full

        if args.use_kernel_pca:
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
            # Note: KernelPCAModel __init__ was fixed to remove 'whiten'
            pca_model = KernelPCAModel(
                k=args.pca_dim,
                kernel=args.kernel_pca_kernel,
                gamma=args.kernel_pca_gamma,
            )
            pca_params = pca_model.fit(all_train_tokens)
        else:
            pca_model = PCAModel(k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten)
            # Note: PCAModel.fit() was fixed to do 3 passes if needed
            pca_params = pca_model.fit(
                feature_generator,
                feature_dim,
                total_tokens,
                num_batches,
            )

        # 2. Determine PR-optimal F1 thresholds (if validation set exists)
        if val_paths:
            logging.info(
                f"Collecting validation stats on {len(val_paths)} images for PR-optimal F1 thresholds..."
            )
            val_img_scores, val_img_labels = [], []
            val_px_scores_normalized, val_px_gts = [], []
            val_iter = tqdm(val_paths, desc="Validating")
            for i in range(0, len(val_paths), args.batch_size):
                path_batch = val_paths[i : i + args.batch_size]
                pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
                is_anomaly_batch = ["good" not in p for p in path_batch]

                if args.patch_size:
                    anomaly_maps_batch = process_image_patched(
                        pil_imgs,
                        extractor,
                        pca_params,
                        args,
                        DEVICE,
                        h_p,
                        w_p,
                        feature_dim,
                    )
                    for j, anomaly_map_final in enumerate(anomaly_maps_batch):
                        # --- IMAGE METRICS (I-AUROC, I-F1) ---
                        # Use RAW map for global score
                        if args.img_score_agg == "max":
                            img_score = float(np.max(anomaly_map_final))
                        elif args.img_score_agg == "p99":
                            img_score = float(np.percentile(anomaly_map_final, 99))
                        else:
                            img_score = float(np.mean(anomaly_map_final))
                        val_img_scores.append(img_score)
                        val_img_labels.append(1 if is_anomaly_batch[j] else 0)

                        # --- PIXEL METRICS (AUPRO, P-F1) ---
                        # Use PER-IMAGE NORMALIZED map
                        am_min = np.min(anomaly_map_final)
                        am_max = np.max(anomaly_map_final)
                        if am_max > am_min:
                            anomaly_map_normalized = (anomaly_map_final - am_min) / (
                                am_max - am_min + 1e-8
                            )
                        else:
                            anomaly_map_normalized = np.zeros_like(
                                anomaly_map_final, dtype=np.float32
                            )

                        H, W = anomaly_map_normalized.shape
                        gt_mask = handler.get_ground_truth_mask(
                            path_batch[j], pil_imgs[j].size
                        )
                        gt_mask = (
                            np.array(
                                Image.fromarray(
                                    (gt_mask.astype(np.uint8) * 255)
                                ).resize((W, H), resample=Image.NEAREST)
                            )
                            > 127
                        )
                        val_px_gts.extend(gt_mask.flatten().astype(np.uint8))
                        val_px_scores_normalized.extend(
                            anomaly_map_normalized.flatten().astype(np.float32)
                        )

                else:
                    tokens, (h_p, w_p) = extractor.extract_tokens(
                        pil_imgs,
                        args.image_res,
                        layers,
                        args.agg_method,
                        grouped_layers,
                        args.docrop,
                        is_cosine=(args.score_method == "cosine"),
                        use_clahe=args.use_clahe,
                    )
                    b, _, _, c = tokens.shape
                    tokens_reshaped = tokens.reshape(b * h_p * w_p, c)

                    scores = calculate_anomaly_scores(
                        tokens_reshaped,
                        pca_params,
                        args.score_method,
                        args.drop_k,
                    )
                    anomaly_maps = scores.reshape(b, h_p, w_p)

                    for j in range(anomaly_maps.shape[0]):
                        anomaly_map_final = post_process_map(
                            anomaly_maps[j], args.image_res
                        )

                        if args.use_specular_filter:
                            img_tensor = (
                                TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                            )
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

                        # --- IMAGE METRICS (I-AUROC, I-F1) ---
                        # Use RAW map for global score
                        if args.img_score_agg == "max":
                            img_score = float(np.max(anomaly_map_final))
                        elif args.img_score_agg == "p99":
                            img_score = float(np.percentile(anomaly_map_final, 99))
                        else:
                            img_score = float(np.mean(anomaly_map_final))
                        val_img_scores.append(img_score)
                        val_img_labels.append(1 if is_anomaly_batch[j] else 0)

                        # --- PIXEL METRICS (AUPRO, P-F1) ---
                        # Use PER-IMAGE NORMALIZED map
                        am_min = np.min(anomaly_map_final)
                        am_max = np.max(anomaly_map_final)
                        if am_max > am_min:
                            anomaly_map_normalized = (anomaly_map_final - am_min) / (
                                am_max - am_min + 1e-8
                            )
                        else:
                            anomaly_map_normalized = np.zeros_like(
                                anomaly_map_final, dtype=np.float32
                            )

                        H, W = anomaly_map_normalized.shape
                        gt_mask = handler.get_ground_truth_mask(
                            path_batch[j], (args.image_res, args.image_res)
                        )
                        gt_mask = (
                            np.array(
                                Image.fromarray(
                                    (gt_mask.astype(np.uint8) * 255)
                                ).resize((W, H), resample=Image.NEAREST)
                            )
                            > 127
                        )
                        val_px_gts.extend(gt_mask.flatten().astype(np.uint8))
                        val_px_scores_normalized.extend(
                            anomaly_map_normalized.flatten().astype(np.float32)
                        )
                val_iter.update(len(path_batch))

            target_img_fpr = getattr(args, "target_img_fpr", 0.05)
            target_px_fpr = getattr(args, "target_px_fpr", 0.05)

            # Threshold for I-F1 (using raw image scores)
            thr_img, how_img = _pick_threshold_with_fallback(
                val_img_labels, val_img_scores, target_img_fpr
            )
            # Threshold for P-F1 (using per-image normalized pixel scores)
            val_px_scores_mm = np.array(val_px_scores_normalized)
            thr_px, how_px = _pick_threshold_with_fallback(
                val_px_gts, val_px_scores_mm, target_px_fpr
            )

            if how_img == "none":
                logging.warning(
                    "Validation image threshold degenerate and no negatives: image F1 will be NaN."
                )
            if how_px == "none":
                logging.warning(
                    "Validation pixel threshold degenerate and no negatives: pixel F1 will be NaN."
                )

            logging.info(
                f"Chosen thresholds — Image: {thr_img if thr_img is not None else float('nan'):.6g} "
                f"({how_img}), Pixel: {thr_px if thr_px is not None else float('nan'):.6g} ({how_px})"
            )

        else:
            logging.warning("No validation set found. F1 scores will be N/A.")
            thr_img, thr_px = None, None

        # 3. Evaluate on Test Set
        logging.info(f"Evaluating on {len(test_paths)} test images...")
        img_true, img_pred_f1 = [], []
        img_pred_auroc = []  # RAW scores for I-AUROC
        px_true_all = []
        px_pred_all_auroc = []  # RAW scores for P-AUROC
        px_pred_all_normalized = []  # NORMALIZED scores for P-F1
        anomalous_gt_masks = []
        anomalous_anomaly_maps = []  # NORMALIZED maps for AUPRO
        vis_saved_count = 0

        test_iter = tqdm(test_paths, desc=f"Testing {category}")
        for i in range(0, len(test_paths), args.batch_size):
            path_batch = test_paths[i : i + args.batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
            is_anomaly_batch = ["good" not in p for p in path_batch]

            if args.patch_size:
                anomaly_maps_batch = process_image_patched(
                    pil_imgs, extractor, pca_params, args, DEVICE, h_p, w_p, feature_dim
                )
                for j, anomaly_map_final in enumerate(anomaly_maps_batch):
                    is_anomaly = is_anomaly_batch[j]
                    path = path_batch[j]
                    pil_img = pil_imgs[j]

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

                    # --- IMAGE METRICS (I-AUROC, I-F1) ---
                    # Use RAW map for global score
                    if args.img_score_agg == "max":
                        img_score = np.max(anomaly_map_final)
                    elif args.img_score_agg == "p99":
                        img_score = np.percentile(anomaly_map_final, 99)
                    else:
                        img_score = np.mean(anomaly_map_final)

                    img_true.append(1 if is_anomaly else 0)
                    img_pred_auroc.append(float(img_score))
                    if thr_img is not None:
                        img_pred_f1.append(1 if img_score >= thr_img else 0)

                    # --- PIXEL METRICS (AUPRO, P-F1) ---
                    # Use PER-IMAGE NORMALIZED map
                    am_min = np.min(anomaly_map_final)
                    am_max = np.max(anomaly_map_final)
                    if am_max > am_min:
                        anomaly_map_normalized = (anomaly_map_final - am_min) / (
                            am_max - am_min + 1e-8
                        )
                    else:
                        anomaly_map_normalized = np.zeros_like(
                            anomaly_map_final, dtype=np.float32
                        )

                    H, W = anomaly_map_normalized.shape
                    gt_mask = handler.get_ground_truth_mask(path, pil_img.size)
                    gt_mask = (
                        np.array(
                            Image.fromarray(gt_mask.astype(np.uint8) * 255).resize(
                                (W, H), resample=Image.NEAREST
                            )
                        )
                        > 127
                    )

                    px_true_all.extend(gt_mask.flatten().astype(np.uint8))
                    # Store RAW scores for P-AUROC
                    px_pred_all_auroc.extend(
                        anomaly_map_final.flatten().astype(np.float32)
                    )
                    # Store NORMALIZED scores for P-F1
                    px_pred_all_normalized.extend(
                        anomaly_map_normalized.flatten().astype(np.float32)
                    )

                    if is_anomaly:
                        anomalous_gt_masks.append(gt_mask)
                        # Store NORMALIZED map for AUPRO
                        anomalous_anomaly_maps.append(anomaly_map_normalized)
                        if vis_saved_count < args.vis_count:
                            save_visualization(
                                path,
                                pil_img,
                                gt_mask,
                                anomaly_map_normalized,  # Use normalized for viz
                                args.outdir,
                                category,
                                vis_saved_count,
                            )
                            vis_saved_count += 1
            else:
                tokens, (h_p, w_p) = extractor.extract_tokens(
                    pil_imgs,
                    args.image_res,
                    layers,
                    args.agg_method,
                    grouped_layers,
                    args.docrop,
                    is_cosine=(args.score_method == "cosine"),
                    use_clahe=args.use_clahe,
                )
                b, _, _, c = tokens.shape
                tokens_reshaped = tokens.reshape(b * h_p * w_p, c)

                scores = calculate_anomaly_scores(
                    tokens_reshaped,
                    pca_params,
                    args.score_method,
                    args.drop_k,
                )
                anomaly_maps = scores.reshape(b, h_p, w_p)

                for j in range(anomaly_maps.shape[0]):
                    pil_img = pil_imgs[j]
                    is_anomaly = is_anomaly_batch[j]
                    path = path_batch[j]
                    anomaly_map_final = post_process_map(
                        anomaly_maps[j], args.image_res
                    )

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

                    # --- IMAGE METRICS (I-AUROC, I-F1) ---
                    # Use RAW map for global score
                    if args.img_score_agg == "max":
                        img_score = np.max(anomaly_map_final)
                    elif args.img_score_agg == "p99":
                        img_score = np.percentile(anomaly_map_final, 99)
                    else:
                        img_score = np.mean(anomaly_map_final)

                    img_true.append(1 if is_anomaly else 0)
                    img_pred_auroc.append(float(img_score))
                    if thr_img is not None:
                        img_pred_f1.append(1 if img_score >= thr_img else 0)

                    # --- PIXEL METRICS (AUPRO, P-F1) ---
                    # Use PER-IMAGE NORMALIZED map
                    am_min = np.min(anomaly_map_final)
                    am_max = np.max(anomaly_map_final)
                    if am_max > am_min:
                        anomaly_map_normalized = (anomaly_map_final - am_min) / (
                            am_max - am_min + 1e-8
                        )
                    else:
                        anomaly_map_normalized = np.zeros_like(
                            anomaly_map_final, dtype=np.float32
                        )

                    H, W = anomaly_map_normalized.shape
                    gt_mask = handler.get_ground_truth_mask(
                        path, (args.image_res, args.image_res)
                    )
                    gt_mask = (
                        np.array(
                            Image.fromarray(gt_mask.astype(np.uint8) * 255).resize(
                                (W, H), resample=Image.NEAREST
                            )
                        )
                        > 127
                    )

                    px_true_all.extend(gt_mask.flatten().astype(np.uint8))
                    # Store RAW scores for P-AUROC
                    px_pred_all_auroc.extend(
                        anomaly_map_final.flatten().astype(np.float32)
                    )
                    # Store NORMALIZED scores for P-F1
                    px_pred_all_normalized.extend(
                        anomaly_map_normalized.flatten().astype(np.float32)
                    )

                    if is_anomaly:
                        anomalous_gt_masks.append(gt_mask)
                        # Store NORMALIZED map for AUPRO
                        anomalous_anomaly_maps.append(anomaly_map_normalized)
                        if vis_saved_count < args.vis_count:
                            save_visualization(
                                path,
                                pil_img,
                                gt_mask,
                                anomaly_map_normalized,  # Use normalized for viz
                                args.outdir,
                                category,
                                vis_saved_count,
                            )
                            vis_saved_count += 1
            test_iter.update(len(path_batch))

        # 4. Calculate Metrics

        # --- Image AUROC (uses RAW scores) ---
        img_auroc = (
            roc_auc_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else np.nan
        )

        px_true_arr = np.array(px_true_all, dtype=np.uint8)
        px_pred_arr_auroc = np.array(px_pred_all_auroc)
        px_pred_arr_normalized = np.array(px_pred_all_normalized)
        has_pos = (px_true_arr == 1).any()
        has_neg = (px_true_arr == 0).any()

        # --- Pixel AUROC (uses RAW scores) ---
        px_auroc = (
            roc_auc_score(px_true_arr, px_pred_arr_auroc)
            if (has_pos and has_neg)
            else np.nan
        )

        # --- Image F1 (uses RAW scores) ---
        img_f1 = f1_score(img_true, img_pred_f1) if (thr_img is not None) else np.nan

        # --- Pixel F1 (uses NORMALIZED scores) ---
        if thr_px is not None and has_pos:
            px_f1 = f1_score(
                px_true_arr.astype(int),
                (px_pred_arr_normalized >= thr_px).astype(int),
            )
        else:
            px_f1 = np.nan

        # --- AUPRO (uses NORMALIZED scores) ---
        if len(anomalous_gt_masks) > 0:
            preds_np = np.stack(anomalous_anomaly_maps).astype(np.float32)  # [N,H,W]
            gts_np = np.stack(anomalous_gt_masks).astype(np.uint8)  # [N,H,W]

            # Maps are already per-image normalized, no global norm needed

            preds_t = (
                torch.from_numpy(preds_np).unsqueeze(1).to(torch.float32).to(DEVICE)
            )  # [N,1,H,W]
            gts_t = (
                torch.from_numpy(gts_np).unsqueeze(1).to(torch.bool).to(DEVICE)
            )  # [N,1,H,W]

            fpr_cap = getattr(args, "pro_integration_limit", 0.3)
            tm_metric = TM_AUPRO(fpr_limit=fpr_cap).to(DEVICE)
            au_pro = tm_metric(preds_t, gts_t).item()
        else:
            logging.warning(
                f"No anomalous images found in test set for {category}. AUPRO is not computable."
            )
            au_pro = np.nan

        # 5. Log and store results
        logging.info(
            f"{category} Results | I-AUROC: {img_auroc:.4f} | P-AUROC: {px_auroc:.4f} | "
            f"AU-PRO: {au_pro:.4f} | I-F1: {img_f1:.4f} | P-F1: {px_f1:.4f}"
        )
        all_results.append([category, img_auroc, px_auroc, au_pro, img_f1, px_f1])

    # --- Final Report ---
    df = pd.DataFrame(
        all_results,
        columns=[
            "Category",
            "Image AUROC",
            "Pixel AUROC",
            "AU-PRO",
            "Image F1",
            "Pixel F1",
        ],
    )
    if not df.empty and len(df) > 1:
        mean_values = df.mean(numeric_only=True)
        mean_row = pd.DataFrame(
            [["Average"] + mean_values.tolist()], columns=df.columns
        )
        df = pd.concat([df, mean_row], ignore_index=True)

    logging.info("\n--- Benchmark Final Results ---")
    logging.info("\n" + df.to_string(index=False, float_format="%.4f", na_rep="N/A"))

    results_path = os.path.join(args.outdir, "benchmark_results.csv")
    df.to_csv(results_path, index=False, float_format="%.4f")
    logging.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
