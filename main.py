import os
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from args import get_args, parse_layer_indices, parse_grouped_layers
from utils import setup_logging, save_config
from dataclass import get_dataset_handler
from features import FeatureExtractor
from pca import PCAModel
from score import calculate_anomaly_scores, post_process_map
from metrics import calculate_au_pro
from viz import save_visualization


torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = get_args()

    # --- Setup ---
    run_name = f"{args.dataset_name}_{args.agg_method}_layers{''.join(args.layers.split(','))}_res{args.image_res}_docrop{int(args.docrop)}"
    args.outdir = os.path.join(args.outdir, run_name)
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir, not args.no_log_file)
    save_config(args)

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

        if not train_paths:
            logging.warning(f"No training images found for {category}. Skipping.")
            continue

        # 1. Fit PCA Model
        # Determine feature dimension and total token count
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
        total_tokens = len(train_paths) * h_p * w_p

        logging.info(
            f"Feature dim: {feature_dim}, Tokens per image: {h_p*w_p}, Total train tokens: {total_tokens}"
        )

        def feature_generator():
            for i in range(0, len(train_paths), args.batch_size):
                path_batch = train_paths[i : i + args.batch_size]
                pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
                tokens_batch, _ = extractor.extract_tokens(
                    pil_imgs,
                    args.image_res,
                    layers,
                    args.agg_method,
                    grouped_layers,
                    args.docrop,
                    is_cosine=(args.score_method == "cosine"),
                    use_clahe=args.use_clahe,
                )
                yield tokens_batch.reshape(-1, feature_dim)

        pca_model = PCAModel(k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten)
        pca_params = pca_model.fit(feature_generator, feature_dim, total_tokens)

        # 2. Determine Adaptive Threshold (if validation set exists)
        threshold = None
        if val_paths:
            logging.info(
                f"Calculating adaptive threshold on {len(val_paths)} validation images..."
            )
            val_scores = []
            for i in tqdm(range(0, len(val_paths), args.batch_size), desc="Validating"):
                path_batch = val_paths[i : i + args.batch_size]
                pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
                tokens, _ = extractor.extract_tokens(
                    pil_imgs,
                    args.image_res,
                    layers,
                    args.agg_method,
                    grouped_layers,
                    args.docrop,
                    is_cosine=(args.score_method == "cosine"),
                    use_clahe=args.use_clahe,
                )
                scores = calculate_anomaly_scores(
                    tokens.reshape(-1, feature_dim),
                    pca_params,
                    args.score_method,
                    args.drop_k,
                )
                val_scores.extend(scores)
            threshold = np.percentile(val_scores, 99.5) if val_scores else 0.0
            logging.info(f"Adaptive threshold for F1-score set to: {threshold:.4f}")
        else:
            logging.warning("No validation set found. F1 scores will be N/A.")

        # 3. Evaluate on Test Set
        logging.info(f"Evaluating on {len(test_paths)} test images...")
        img_true, img_pred_auroc, img_pred_f1 = [], [], []
        px_true_all, px_pred_all = [], []
        anomalous_gt_masks, anomalous_anomaly_maps = [], []
        vis_saved_count = 0

        for path in tqdm(test_paths, desc=f"Testing {category}"):
            is_anomaly = "good" not in path
            pil_img = Image.open(path).convert("RGB")

            tokens, (h_p, w_p) = extractor.extract_tokens(
                [pil_img],
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                is_cosine=(args.score_method == "cosine"),
                use_clahe=args.use_clahe,
            )

            scores = calculate_anomaly_scores(
                tokens.reshape(-1, feature_dim),
                pca_params,
                args.score_method,
                args.drop_k,
            )
            anomaly_map = scores.reshape(h_p, w_p)
            anomaly_map_final = post_process_map(anomaly_map, args.image_res)

            # Aggregate pixel scores to image score
            if args.img_score_agg == "max":
                img_score = np.max(anomaly_map_final)
            elif args.img_score_agg == "p99":
                img_score = np.percentile(anomaly_map_final, 99)
            else:
                img_score = np.mean(anomaly_map_final)

            img_true.append(1 if is_anomaly else 0)
            img_pred_auroc.append(img_score)
            if threshold is not None:
                img_pred_f1.append(1 if img_score > threshold else 0)

            gt_mask = handler.get_ground_truth_mask(path, args.image_res)
            px_true_all.extend(gt_mask.flatten())
            px_pred_all.extend(anomaly_map_final.flatten())

            if is_anomaly:
                anomalous_gt_masks.append(gt_mask)
                anomalous_anomaly_maps.append(anomaly_map_final)
                if vis_saved_count < args.vis_count:
                    save_visualization(
                        path,
                        pil_img,
                        gt_mask,
                        anomaly_map_final,
                        args.outdir,
                        category,
                        vis_saved_count,
                    )
                    vis_saved_count += 1

        # 4. Calculate Metrics
        img_auroc = (
            roc_auc_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else 0.0
        )
        px_auroc = roc_auc_score(px_true_all, px_pred_all) if any(px_true_all) else 0.0

        img_f1 = f1_score(img_true, img_pred_f1) if threshold is not None else np.nan
        px_f1 = (
            f1_score(px_true_all, (np.array(px_pred_all) > threshold).astype(int))
            if threshold is not None and any(px_true_all)
            else np.nan
        )

        au_pro = calculate_au_pro(
            anomalous_gt_masks, anomalous_anomaly_maps, args.pro_integration_limit
        )

        # 5. Log and store results
        logging.info(
            f"{category} Results | I-AUROC: {img_auroc:.4f} | P-AUROC: {px_auroc:.4f} | AU-PRO: {au_pro:.4f} | I-F1: {img_f1:.4f} | P-F1: {px_f1:.4f}"
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
