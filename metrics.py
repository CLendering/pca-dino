import numpy as np
from sklearn.metrics import roc_curve, auc
from skimage.measure import label
from tqdm import tqdm


def calculate_au_pro(gt_masks: list, anomaly_maps: list, integration_limit: float):
    if not gt_masks:
        return 0.0

    # Flatten all masks and maps
    flat_gts = np.concatenate([m.flatten() for m in gt_masks])
    flat_preds = np.concatenate([am.flatten() for am in anomaly_maps])

    # Get ROC curve up to the integration limit
    fprs, _, thresholds = roc_curve(flat_gts, flat_preds)
    valid_indices = fprs <= integration_limit
    fprs_lim, thresholds_lim = fprs[valid_indices], thresholds[valid_indices]

    # Subsample thresholds for efficiency
    if len(thresholds_lim) > 500:
        indices = np.linspace(0, len(thresholds_lim) - 1, 500, dtype=int)
        thresholds_lim = thresholds_lim[indices]
        fprs_lim = fprs_lim[indices]

    # Calculate Per-Region Overlap (PRO) for each threshold
    pros = []
    for th in tqdm(
        thresholds_lim, desc=f"AU-PRO ({integration_limit:.2f})", leave=False, ncols=80
    ):
        pro_scores = []
        for gt_mask, anom_map in zip(gt_masks, anomaly_maps):
            if np.sum(gt_mask) == 0:
                continue
            pred_mask = (anom_map >= th).astype(np.uint8)
            labeled_gt, num_components = label(gt_mask, return_num=True, connectivity=2)
            for c in range(1, num_components + 1):
                component_mask = labeled_gt == c
                pro = np.sum(component_mask & pred_mask) / np.sum(component_mask)
                pro_scores.append(pro)
        pros.append(np.mean(pro_scores) if pro_scores else 0.0)

    return auc(fprs_lim, pros) / integration_limit if integration_limit > 0 else 0.0
