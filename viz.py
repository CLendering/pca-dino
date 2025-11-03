from PIL import Image
import numpy as np
import cv2
import os
from typing import Optional


def save_visualization(
    path: str,
    img: Image.Image,
    gt_mask: np.ndarray,
    anom_map: np.ndarray,
    outdir: str,
    category: str,
    vis_idx: int,
    saliency_mask: Optional[np.ndarray] = None,
):
    """Saves a multi-panel visualization of an anomaly."""

    img_np = np.array(img.resize((anom_map.shape[1], anom_map.shape[0])))

    heatmap = cv2.applyColorMap(
        cv2.normalize(anom_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        cv2.COLORMAP_JET,
    )

    if gt_mask.shape != anom_map.shape:
        print(
            f"GT had shape {gt_mask.shape}, while Anom map had shape {anom_map.shape}"
        )
        gt_mask = cv2.resize(
            gt_mask.astype(np.uint8),
            (anom_map.shape[1], anom_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    gt_mask_vis = cv2.cvtColor((gt_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def add_text(img, text):
        return cv2.putText(
            img.copy(),
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    panel1 = add_text(img_np, "Original")
    panel2 = add_text(gt_mask_vis, "Ground Truth")
    panel3 = add_text(heatmap, "Anomaly Map")

    if saliency_mask is not None:
        # --- THIS IS THE FIX ---
        # Convert binary 0/1 float mask to 0/255 uint8 image
        saliency_mask_vis = (saliency_mask * 255).astype(np.uint8)
        # --- END FIX ---
        
        saliency_mask_vis = cv2.cvtColor(saliency_mask_vis, cv2.COLOR_GRAY2RGB)
        panel4 = add_text(saliency_mask_vis, "Saliency Mask (FG)")
    else:
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        panel4 = add_text(overlay, "Overlay")

    combined_img = np.vstack([np.hstack([panel1, panel2]), np.hstack([panel3, panel4])])

    vis_dir = os.path.join(outdir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    out_path = os.path.join(vis_dir, f"{category}_example_{vis_idx}.png")
    Image.fromarray(combined_img).save(out_path)