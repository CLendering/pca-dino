import torch
import torch.nn.functional as F
import kornia as K
import sys
import os
from PIL import Image

def specular_mask_torch(img_rgb, tau=0.6):
    """
    img_rgb: float tensor in [0,1], shape [B,3,H,W], sRGB
    returns:
      bin_mask [B,1,H,W] bool
      soft_spec [B,1,H,W] float
      conf     [B,1,H,W] = 1 - soft_spec  (use as confidence)
    """
    eps = 1e-6
    B, C, H, W = img_rgb.shape

    # linearize
    I_lin = torch.clamp(img_rgb, eps, 1.0) ** 2.2
    R, G, Bc = I_lin[:,0:1], I_lin[:,1:2], I_lin[:,2:3]
    Y = 0.2126*R + 0.7152*G + 0.0722*Bc  # [B,1,H,W]

    # HSV saturation from sRGB (kornia)
    hsv = K.color.rgb_to_hsv(img_rgb)
    S = hsv[:,1:2]  # [B,1,H,W]

    # clipping cue (on sRGB)
    clip_flag = (img_rgb.max(dim=1, keepdim=True).values > 0.985).float()

    # bright cue
    kY, tY = 15.0, 0.85
    sY = torch.sigmoid(kY * (Y - tY))

    # desaturation cue
    kS, tS = 10.0, 0.25
    sS = torch.sigmoid(kS * (tS - S))

    # curvature cue on Y: LoG via Laplacian(Gaussian)
    Y_blur = K.filters.gaussian_blur2d(Y, (3,3), (1.0,1.0))
    lap = K.filters.laplacian(Y_blur, kernel_size=3)  # [B,1,H,W]
    # percentile per-image
    tk = torch.quantile(lap.view(B,-1), q=0.95, dim=1).view(B,1,1,1) + 1e-6
    sK = torch.sigmoid(4.0 * (lap - tk) / tk)

    # combine
    w1, w2, w3, w4 = 0.5, 0.3, 0.2, 0.3
    Sspec = torch.clamp(w1*sY + w2*sS + w3*sK + w4*clip_flag, 0.0, 1.0)

    # binary + confidence
    bin_mask = (Sspec > tau)
    conf = 1.0 - Sspec  # confidence to KEEP pixel
    return bin_mask, Sspec, conf


def filter_specular_anomalies(anomaly_map, conf_map):
    """
    Filters anomalies by multiplying with a confidence map.
    This reduces anomaly scores in regions with low confidence (i.e. specular regions).
    The confidence map is (1 - soft_specular_map).

    Args:
        anomaly_map (np.ndarray): Anomaly score map.
        conf_map (np.ndarray): Confidence map, where low values indicate specular regions.

    Returns:
        np.ndarray: Filtered anomaly map.
    """
    return anomaly_map * conf_map


if __name__ == "__main__":
    import torchvision.transforms.functional as TF
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"

    img = Image.open(img_path).convert("RGB")
    inp = TF.to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W], float in [0,1]

    bin_mask, soft_spec, conf = specular_mask_torch(inp)

    out_dir = "spec_outputs"
    os.makedirs(out_dir, exist_ok=True)

    soft_np = soft_spec[0, 0].cpu().detach().numpy()
    bin_np = bin_mask[0, 0].cpu().numpy().astype("uint8") * 255
    conf_np = conf[0, 0].cpu().detach().numpy()

    plt.imsave(os.path.join(out_dir, "soft_spec.png"), soft_np, cmap="viridis")
    plt.imsave(os.path.join(out_dir, "binary_mask.png"), bin_np, cmap="gray")
    plt.imsave(os.path.join(out_dir, "conf.png"), conf_np, cmap="viridis")

    print(f"Saved soft_spec, binary_mask, conf to {out_dir}")