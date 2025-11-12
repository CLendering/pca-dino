from PIL import Image
import numpy as np
import cv2
import os
from typing import Optional
from pathlib import Path


def save_overlay_for_intro(
    path: str,
    img: Image.Image,
    anom_map: np.ndarray,
    outdir: str,
    category: str,
    kernel_size: int = 5,
    overlay_intensity: float = 0.4,
):
    """
    Saves a convincing overlay by denoising the anomaly map
    before blending, removing small, isolated noise.
    """
    
    # --- 1. Normalize Map and Get Heatmap ---
    img_h, img_w = anom_map.shape
    img_np = np.array(img.resize((img_w, img_h)))
    
    # Ensure image is RGB for blending
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # Normalize anomaly map to 8-bit
    anom_map_norm_u8 = cv2.normalize(
        anom_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    
    # Create the full heatmap
    heatmap = cv2.applyColorMap(anom_map_norm_u8, cv2.COLORMAP_JET)

    # --- 2. Create a *Denoised* Mask for Anomalous Regions ---
    
    # Use Otsu's method to get an initial binary threshold
    otsu_threshold, binary_mask = cv2.threshold(
        anom_map_norm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Use Morphological Opening to remove small noise speckles
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    denoised_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # (Optional) Dilate the mask slightly so the heatmap bleeds over
    # the edges a bit, which looks better.
    denoised_mask = cv2.dilate(denoised_mask, kernel, iterations=1)
    
    # --- 3. Combine Image, Heatmap, and Denoised Mask ---
    
    # Create the blended overlay
    overlay = cv2.addWeighted(
        img_np, (1.0 - overlay_intensity), heatmap, overlay_intensity, 0
    )

    # Convert 1-channel mask to 3-channel for np.where
    mask_3d = cv2.cvtColor(denoised_mask, cv2.COLOR_GRAY2RGB)
    
    # Combine using the mask:
    # Where mask is > 0, use the 'overlay'.
    # Where mask is 0, use the original 'img_np'.
    final_image = np.where(mask_3d > 0, overlay, img_np)

    # --- 4. Save File ---
    vis_dir = os.path.join(outdir, "intro_overlays", category)
    os.makedirs(vis_dir, exist_ok=True)
    
    # *** THIS IS THE FIX ***
    # Create a unique filename like "contamination_001.png"
    p = Path(path)
    defect_type = p.parent.name
    original_filename = p.name
    unique_filename = f"{defect_type}_{original_filename}"
    # *** END FIX ***
    
    out_path = os.path.join(vis_dir, unique_filename)
    Image.fromarray(final_image).save(out_path)


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

    # anom_map is the 0-1 normalized float map
    img_np = np.array(img.resize((anom_map.shape[1], anom_map.shape[0])))

    # Convert 0-1 map to 0-255 for applyColorMap
    anom_map_u8 = (anom_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anom_map_u8, cv2.COLORMAP_JET)

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
    
    # Ensure all panels are RGB for stacking
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    panel1 = add_text(img_np, "Original")
    panel2 = add_text(gt_mask_vis, "Ground Truth")
    panel3 = add_text(heatmap, "Anomaly Map")

    if saliency_mask is not None:
        # saliency_mask is 0-1 float, convert to 8-bit
        saliency_mask_vis = (saliency_mask * 255).astype(np.uint8)
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