import os
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from args import get_args, parse_layer_indices, parse_grouped_layers
from utils import setup_logging, save_config
from dataclass import get_dataset_handler
from features import FeatureExtractor
from pca import PCAModel, KernelPCAModel
from score import calculate_anomaly_scores, post_process_map
from metrics import calculate_au_pro
from viz import save_visualization
from specular import specular_mask_torch, filter_specular_anomalies
from patching import process_image_patched, get_patch_coords


torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


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

            # Calculate total number of patches and tokens
            total_patches = 0
            num_batches = 0
            for path in train_paths:
                img = Image.open(path).convert("RGB")
                patch_coords = get_patch_coords(
                    img.height, img.width, args.patch_size, args.patch_overlap
                )
                total_patches += len(patch_coords)
                num_batches += math.ceil(len(patch_coords) / args.batch_size)
            total_tokens = total_patches * tokens_per_patch

            logging.info(
                f"Feature dim: {feature_dim}, Tokens per patch: {tokens_per_patch}, Total train patches: {total_patches}, Total train tokens: {total_tokens}"
            )

            def feature_generator_patched():
                for path in train_paths:
                    pil_img = Image.open(path).convert("RGB")
                    patch_coords = get_patch_coords(
                        pil_img.height, pil_img.width, args.patch_size, args.patch_overlap
                    )
                    for i in range(0, len(patch_coords), args.batch_size):
                        coord_batch = patch_coords[i : i + args.batch_size]
                        patch_batch = [pil_img.crop(c) for c in coord_batch]
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
            total_tokens = len(train_paths) * h_p * w_p

            logging.info(
                f"Feature dim: {feature_dim}, Tokens per image: {h_p*w_p}, Total train tokens: {total_tokens}"
            )

            def feature_generator_full():
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

            num_batches = math.ceil(len(train_paths) / args.batch_size)
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
            pca_model = KernelPCAModel(
                k=args.pca_dim,
                kernel=args.kernel_pca_kernel,
                gamma=args.kernel_pca_gamma,
                whiten=args.whiten,
            )
            pca_params = pca_model.fit(all_train_tokens)
        else:
            pca_model = PCAModel(k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten)
            pca_params = pca_model.fit(
                feature_generator, feature_dim, total_tokens, num_batches
            )

        # 2. Determine Adaptive Threshold (if validation set exists)
        threshold = None
        if val_paths:
            logging.info(
                f"Calculating adaptive threshold on {len(val_paths)} validation images..."
            )
            val_scores = []
            val_iter = tqdm(val_paths, desc="Validating")
            for i in range(0, len(val_paths), args.batch_size):
                path_batch = val_paths[i : i + args.batch_size]
                pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]

                if args.patch_size:
                    anomaly_maps_final = process_image_patched(
                        pil_imgs, extractor, pca_params, args, DEVICE, h_p, w_p, feature_dim
                    )
                    for anomaly_map_final in anomaly_maps_final:
                        val_scores.extend(anomaly_map_final.flatten())
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
                    
                    b, _, c = tokens.shape
                    tokens_reshaped = tokens.reshape(b * h_p * w_p, c)
                    
                    scores = calculate_anomaly_scores(
                        tokens_reshaped,
                        pca_params,
                        args.score_method,
                        args.drop_k,
                    )
                    
                    anomaly_maps = scores.reshape(b, h_p, w_p)
                    
                    for j in range(anomaly_maps.shape[0]):
                        anomaly_map_final = post_process_map(anomaly_maps[j], args.image_res)

                        if args.use_specular_filter:
                            img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                            _, _, conf = specular_mask_torch(img_tensor, tau=args.specular_tau)
                            conf = torch.nn.functional.interpolate(
                                conf,
                                size=anomaly_map_final.shape,
                                mode='bilinear',
                                align_corners=False
                            )
                            conf_map = conf.squeeze().cpu().numpy()
                            anomaly_map_final = filter_specular_anomalies(anomaly_map_final, conf_map)

                        val_scores.extend(anomaly_map_final.flatten())
                val_iter.update(len(path_batch))

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
        
        test_iter = tqdm(test_paths, desc=f"Testing {category}")
        for i in range(0, len(test_paths), args.batch_size):
            path_batch = test_paths[i : i + args.batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
            is_anomaly_batch = ["good" not in p for p in path_batch]

            if args.patch_size:
                anomaly_maps_final = process_image_patched(
                    pil_imgs, extractor, pca_params, args, DEVICE, h_p, w_p, feature_dim
                )
                for j, anomaly_map_final in enumerate(anomaly_maps_final):
                    is_anomaly = is_anomaly_batch[j]
                    path = path_batch[j]
                    pil_img = pil_imgs[j]
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
                b, _, c = tokens.shape
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
                    anomaly_map_final = post_process_map(anomaly_maps[j], args.image_res)

                    if args.use_specular_filter:
                        img_tensor = TF.to_tensor(pil_img).unsqueeze(0).to(DEVICE)
                        _, _, conf = specular_mask_torch(img_tensor, tau=args.specular_tau)
                        conf = torch.nn.functional.interpolate(
                            conf,
                            size=anomaly_map_final.shape,
                            mode='bilinear',
                            align_corners=False
                        )
                        conf_map = conf.squeeze().cpu().numpy()
                        anomaly_map_final = filter_specular_anomalies(anomaly_map_final, conf_map)

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

                    if args.patch_size:
                        gt_mask = handler.get_ground_truth_mask(path, pil_img.size[::-1])
                    else:
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
            test_iter.update(len(path_batch))

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

