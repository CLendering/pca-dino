import torch
import kornia as K
import numpy as np


def specular_mask_torch(img_rgb, tau=0.6):
    """
    img_rgb: float tensor in [0,1], shape [B,3,H,W], sRGB
    returns:
      bin_mask [B,1,H,W] bool
      soft_spec [B,1,H,W] float
      conf      [B,1,H,W] = 1 - soft_spec  (use as confidence)
    """
    eps = 1e-6
    B, C, H, W = img_rgb.shape

    # linearize
    I_lin = torch.clamp(img_rgb, eps, 1.0) ** 2.2
    R, G, Bc = I_lin[:, 0:1], I_lin[:, 1:2], I_lin[:, 2:3]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * Bc  # [B,1,H,W]

    # HSV saturation from sRGB (kornia)
    hsv = K.color.rgb_to_hsv(img_rgb)
    S = hsv[:, 1:2]  # [B,1,H,W]

    # clipping cue (on sRGB)
    clip_flag = (img_rgb.max(dim=1, keepdim=True).values > 0.985).float()

    # bright cue
    kY, tY = 15.0, 0.85
    sY = torch.sigmoid(kY * (Y - tY))

    # desaturation cue
    kS, tS = 10.0, 0.25
    sS = torch.sigmoid(kS * (tS - S))

    # curvature cue on Y: LoG via Laplacian(Gaussian)
    Y_blur = K.filters.gaussian_blur2d(Y, (3, 3), (1.0, 1.0))
    lap = K.filters.laplacian(Y_blur, kernel_size=3)  # [B,1,H,W]
    # percentile per-image
    tk = torch.quantile(lap.view(B, -1), q=0.95, dim=1).view(B, 1, 1, 1) + 1e-6
    sK = torch.sigmoid(4.0 * (lap - tk) / tk)

    # combine
    w1, w2, w3, w4 = 0.5, 0.3, 0.2, 0.3
    Sspec = torch.clamp(w1 * sY + w2 * sS + w3 * sK + w4 * clip_flag, 0.0, 1.0)

    # binary + confidence
    bin_mask = Sspec > tau
    conf = 1.0 - Sspec  # confidence to KEEP pixel
    return bin_mask, Sspec, conf


def filter_specular_anomalies(anomaly_map, conf_map, blur_sigma=5.0):
    """
    Filters specular FPs by comparing a pixel's anomaly score to its
    non-specular neighborhood.
    """
    eps = 1e-6
    ksize = int(blur_sigma * 4 + 0.5) * 2 + 1

    if isinstance(conf_map, np.ndarray):
        conf_map = torch.from_numpy(conf_map)
    elif not isinstance(conf_map, torch.Tensor):
        raise TypeError(
            f"conf_map must be a torch.Tensor or np.ndarray. Got: {type(conf_map)}"
        )

    device = (
        conf_map.device
        if conf_map.is_cuda
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    conf_map = conf_map.to(device)

    if isinstance(anomaly_map, np.ndarray):
        anomaly_map = torch.from_numpy(anomaly_map).to(device)
    elif not isinstance(anomaly_map, torch.Tensor):
        raise TypeError(
            f"anomaly_map must be a torch.Tensor or np.ndarray. Got: {type(anomaly_map)}"
        )
    else:
        anomaly_map = anomaly_map.to(device)

    original_shape = anomaly_map.shape

    # Reshape anomaly_map to 4D
    if anomaly_map.dim() == 2:  # [H, W]
        anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif anomaly_map.dim() == 3:  # [B, H, W] or [C, H, W]
        anomaly_map = anomaly_map.unsqueeze(1)  # Assume [B, H, W] -> [B, 1, H, W]

    # Also reshape conf_map to 4D (it should be 4D, but robust code helps)
    if conf_map.dim() == 2:  # [H, W]
        conf_map = conf_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif conf_map.dim() == 3:  # [B, H, W] or [C, H, W]
        conf_map = conf_map.unsqueeze(1)  # Assume [B, H, W] -> [B, 1, H, W]

    # Final check
    if anomaly_map.dim() != 4 or conf_map.dim() != 4:
        raise ValueError(
            f"Could not convert inputs to 4D. "
            f"Got anomaly_map: {original_shape} -> {anomaly_map.shape} and "
            f"conf_map: {conf_map.shape}"
        )

    # 1. Get the anomaly map for *non-specular* regions.
    # (This is now a 4D * 4D operation)
    anomaly_map_non_spec = anomaly_map * conf_map

    # 2. Get the average non-specular anomaly score in the neighborhood.
    blur_kernel = (ksize, ksize), (blur_sigma, blur_sigma)

    # These Kornia calls now correctly receive 4D tensors
    sum_weighted_anomalies = K.filters.gaussian_blur2d(
        anomaly_map_non_spec, *blur_kernel
    )
    sum_weights = K.filters.gaussian_blur2d(conf_map, *blur_kernel)

    anomaly_map_non_spec_avg = sum_weighted_anomalies / (sum_weights + eps)

    # 3. Compute the "context score"
    context_score = (anomaly_map_non_spec_avg / (anomaly_map + eps)).clamp(0.0, 1.0)

    # 4. Linearly interpolate the suppression multiplier.
    suppression_multiplier = torch.lerp(
        conf_map,
        torch.tensor(1.0, device=device),  # Use the device we found
        context_score,
    )

    filtered_map = (anomaly_map * suppression_multiplier).clone().detach()

    if len(original_shape) == 2:
        return filtered_map.squeeze(0).squeeze(0)  # [1, 1, H, W] -> [H, W]
    elif len(original_shape) == 3:
        return filtered_map.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    else:
        return filtered_map