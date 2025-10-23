import torch
import kornia as K


def specular_mask_torch(img_rgb, tau=0.6):
    """
    img_rgb: float tensor in [0,1], shape [B,3,H,W], sRGB
    returns:
      bin_mask [B,1,H,W] bool (True == is specular)
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
    try:
        tk = torch.quantile(lap.view(B, -1), q=0.95, dim=1).view(B, 1, 1, 1) + 1e-6
    except IndexError:  # Handle empty tensor case
        tk = torch.tensor(1e-6, device=lap.device)
    sK = torch.sigmoid(4.0 * (lap - tk) / tk)

    # combine
    w1, w2, w3, w4 = 0.5, 0.3, 0.2, 0.3
    Sspec = torch.clamp(w1 * sY + w2 * sS + w3 * sK + w4 * clip_flag, 0.0, 1.0)

    # binary + confidence
    bin_mask = Sspec > tau  # True means "is specular"
    conf = 1.0 - Sspec  # confidence to KEEP pixel
    return bin_mask, Sspec, conf


@torch.no_grad()
def suppress_specular_fp(
    anomaly_map,
    spec_mask,
):
    # TODO: Figure this out.
    return anomaly_map
