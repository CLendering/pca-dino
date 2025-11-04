import logging
import torch
from transformers import AutoImageProcessor, AutoModel
import cv2
import numpy as np
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    """Encapsulates the feature extraction model and logic."""

    def __init__(self, model_ckpt: str):
        logging.info(f"Loading feature extraction model: {model_ckpt}...")
        self.processor = AutoImageProcessor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).eval().to(DEVICE)
        
        # --- FIX for DINOv3/Transformers ---
        # Force 'eager' attention implementation to get attention weights
        # This resolves the "TypeError: 'NoneType' object is not subscriptable"
        try:
            self.model.set_attn_implementation('eager')
            logging.info("Set model attention implementation to 'eager' to get attention weights.")
        except AttributeError:
            logging.warning("Could not set attention implementation. Saliency masking might fail.")
        # --- END FIX ---
            
        logging.info("Model loaded successfully.")

    @torch.no_grad()
    def extract_tokens(
        self,
        pil_imgs: list,
        res: int,
        layers: list,
        agg_method: str,
        grouped_layers: list = [],
        docrop: bool = False,
        is_cosine: bool = False,
        use_clahe: bool = False,
        dino_saliency_layer: int = 0, # New argument
    ):
        """Extracts and aggregates features from a batch of images."""
        if use_clahe:
            processed_imgs = []
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for img in pil_imgs:
                # Convert PIL image to numpy array (RGB)
                img_np = np.array(img)
                # Convert RGB to LAB
                img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                # Split LAB channels
                l_, a, b = cv2.split(img_lab)
                # Apply CLAHE to L-channel
                l_clahe = clahe.apply(l_)
                # Merge channels back
                img_lab_clahe = cv2.merge((l_clahe, a, b))
                # Convert LAB back to RGB
                img_rgb_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
                # Convert back to PIL image
                processed_imgs.append(Image.fromarray(img_rgb_clahe))
            pil_imgs = processed_imgs

        if docrop:
            resize_res = int(res / 0.875)
            do_resize = True
            size = {"height": resize_res, "width": resize_res}
            crop_size = {"height": res, "width": res}
        else:
            do_resize = True
            size = {"height": res, "width": res}
            crop_size = {"height": res, "width": res}

        inputs = self.processor(
            images=pil_imgs,
            return_tensors="pt",
            do_resize=do_resize,
            size=size,
            do_center_crop=docrop,
            crop_size=crop_size,
        ).to(DEVICE)

        outputs = self.model(
            **inputs, output_hidden_states=True, output_attentions=True
        )
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions  # Tuple of (B, Heads, N, N)
        
        if attentions is None:
             raise ValueError("Attention weights are None. This is likely because the model is using a fast attention implementation "
                              "(like Flash Attention). The 'eager' implementation was set, but this error persists. "
                              "Please check your transformers library version or model compatibility.")

        cfg = self.model.config
        ps = cfg.patch_size
        num_reg = getattr(cfg, "num_register_tokens", 0)
        drop_front = 1 + num_reg
        h_p, w_p = res // ps, res // ps
        N_expected = h_p * w_p

        # --- Saliency Mask Generation ---
        if dino_saliency_layer < 0: # Handle negative indexing
            dino_saliency_layer = len(attentions) + dino_saliency_layer
        
        if dino_saliency_layer >= len(attentions):
            logging.warning(f"DINO saliency layer {dino_saliency_layer} is out of bounds (0-{len(attentions)-1}). Defaulting to layer 0.")
            dino_saliency_layer = 0
            
        attn_map = attentions[dino_saliency_layer]  # Shape: (B, num_heads, N_tokens, N_tokens)
        
        if num_reg > 0:
            # DINOv3 uses register tokens. Their attention is a better saliency map.
            # Get attention from REGISTER tokens (1 to num_reg) to PATCH tokens
            reg_attn_to_patches = attn_map[
                :, :, 1:drop_front, drop_front : drop_front + N_expected
            ]
            # Average across all heads AND all register tokens
            saliency_mask = reg_attn_to_patches.mean(dim=(1, 2))  # Shape: (B, N_expected)
        else:
            # Fallback to CLS token (DINOv1 style)
            logging.info("No register tokens found. Using CLS token for saliency mask.")
            cls_attn_to_patches = attn_map[
                :, :, 0, drop_front : drop_front + N_expected
            ]
            # Average across all heads
            saliency_mask = cls_attn_to_patches.mean(dim=1)  # Shape: (B, N_expected)

        # Reshape to 2D saliency mask
        saliency_mask = saliency_mask.reshape(
            inputs.pixel_values.shape[0], h_p, w_p
        )
        # --- End Saliency ---


        def _spatial_from_seq(seq_tokens: torch.Tensor) -> torch.Tensor:
            B, N, C = seq_tokens.shape
            tokens = seq_tokens[:, drop_front : drop_front + N_expected, :]
            return tokens.reshape(B, h_p, w_p, C)

        if agg_method == "group":
            if not grouped_layers:
                raise ValueError(
                    "Grouped layers must be provided for 'group' aggregation."
                )
            all_layer_indices = sorted(
                list(set(idx for group in grouped_layers for idx in group))
            )
            layer_tensors = {
                li: _spatial_from_seq(hidden_states[li]) for li in all_layer_indices
            }
            fused_groups = [
                torch.stack([layer_tensors[li] for li in group], dim=0).mean(dim=0)
                for group in grouped_layers
            ]
            fused = torch.cat(fused_groups, dim=-1)
        else:
            feats = [_spatial_from_seq(hidden_states[li]) for li in layers]
            if agg_method == "concat":
                fused = torch.cat(feats, dim=-1)
            elif agg_method == "mean":
                fused = torch.stack(feats, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown aggregation method: '{agg_method}'")
            
        return fused.cpu().numpy(), (h_p, w_p), saliency_mask.cpu().numpy()