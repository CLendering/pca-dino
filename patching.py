import numpy as np
from args import parse_layer_indices, parse_grouped_layers
import logging
import cv2  # Import OpenCV
from features import FeatureExtractor
from score import calculate_anomaly_scores, post_process_map


def get_patch_coords(image_height, image_width, patch_size, overlap):
    """Calculates patch coordinates.

    Args:
        image_height (int): Height of the image.
        image_width (int): Width of the image.
        patch_size (int): Size of the square patches.
        overlap (float): Overlap ratio between patches.

    Returns:
        list: A list of (x1, y1, x2, y2) coordinates for each patch.
    """
    coords = []
    stride = int(patch_size * (1 - overlap))
    for y in range(0, image_height, stride):
        for x in range(0, image_width, stride):
            x1, y1 = x, y
            x2, y2 = min(x + patch_size, image_width), min(y + patch_size, image_height)
            if (x2 - x1) < patch_size or (y2 - y1) < patch_size:
                x1, y1 = max(0, x2 - patch_size), max(0, y2 - patch_size)
            coords.append((x1, y1, x2, y2))
    return coords


def process_image_patched(
    pil_imgs: list,
    extractor: FeatureExtractor,
    pca_params,
    args,
    device="cpu",
    h_p=None,
    w_p=None,
    feature_dim=None,
):
    """Processes a batch of images in patches and returns a list of stitched anomaly maps."""
    anomaly_maps_final = []
    saliency_maps_final = []  # Store stitched saliency maps

    if args.bg_mask_method == "pca_normality":
        logging.warning(
            "Patching mode is not compatible with 'pca_normality' masking. "
            "Falling back to 'dino_saliency' for masking."
        )

    for pil_img in pil_imgs:
        img_width, img_height = pil_img.size
        patch_coords = get_patch_coords(
            img_height, img_width, args.patch_size, args.patch_overlap
        )

        anomaly_map_full = np.zeros((img_height, img_width), dtype=np.float32)
        count_map = np.zeros((img_height, img_width), dtype=np.float32)

        saliency_map_full = np.zeros((img_height, img_width), dtype=np.float32)
        s_count_map = np.zeros((img_height, img_width), dtype=np.float32)

        # Process patches in batches
        for i in range(0, len(patch_coords), args.batch_size):
            coord_batch = patch_coords[i : i + args.batch_size]
            patch_batch = [pil_img.crop(c) for c in coord_batch]

            tokens, _, saliency_masks_batch = extractor.extract_tokens(
                patch_batch,
                args.image_res,
                parse_layer_indices(args.layers),
                args.agg_method,
                parse_grouped_layers(args.grouped_layers)
                if args.agg_method == "group"
                else [],
                args.docrop,
                is_cosine=(args.score_method == "cosine"),
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )

            scores = calculate_anomaly_scores(
                tokens.reshape(-1, feature_dim),
                pca_params,
                args.score_method,
                args.drop_k,
            )
            anomaly_maps_batch = scores.reshape(len(patch_batch), h_p, w_p)

            if args.bg_mask_method is not None:
                # In patching mode, we must use dino_saliency, as pca_normality is not possible.
                background_mask = np.zeros_like(anomaly_maps_batch, dtype=bool)
                for j in range(len(patch_batch)):
                    saliency_map = saliency_masks_batch[j]
                    try:
                        if args.mask_threshold_method == "percentile":
                            threshold = np.percentile(
                                saliency_map, args.percentile_threshold * 100
                            )
                            background_mask[j] = saliency_map < threshold
                        else:  # otsu
                            norm_mask = cv2.normalize(
                                saliency_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                            )
                            _, binary_mask = cv2.threshold(
                                norm_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                            )
                            background_mask[j] = binary_mask == 0
                    except Exception as e:
                        logging.warning(f"Saliency mask failed for patch {j}: {e}. Skipping mask.")
                anomaly_maps_batch[background_mask] = 0.0  # Zero out background


            for j, anomaly_map_patch in enumerate(anomaly_maps_batch):
                anomaly_map_patch_resized = post_process_map(
                    anomaly_map_patch,
                    (
                        coord_batch[j][3] - coord_batch[j][1],
                        coord_batch[j][2] - coord_batch[j][0],
                    ),
                )
                x1, y1, x2, y2 = coord_batch[j]
                anomaly_map_full[y1:y2, x1:x2] += anomaly_map_patch_resized
                count_map[y1:y2, x1:x2] += 1

                # Stitch saliency map
                saliency_map_patch = saliency_masks_batch[j]
                saliency_map_patch_resized = post_process_map(
                    saliency_map_patch,
                    (
                        coord_batch[j][3] - coord_batch[j][1],
                        coord_batch[j][2] - coord_batch[j][0],
                    ),
                    blur=False,  # Don't blur a mask
                )
                saliency_map_full[y1:y2, x1:x2] += saliency_map_patch_resized
                s_count_map[y1:y2, x1:x2] += 1

        # Average the scores in overlapping regions
        anomaly_map_final = np.divide(
            anomaly_map_full,
            count_map,
            out=np.zeros_like(anomaly_map_full),
            where=count_map != 0,
        )
        anomaly_maps_final.append(anomaly_map_final)

        saliency_map_final = np.divide(
            saliency_map_full,
            s_count_map,
            out=np.zeros_like(saliency_map_full),
            where=s_count_map != 0,
        )
        saliency_maps_final.append(saliency_map_final)

    return anomaly_maps_final, saliency_maps_final