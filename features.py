import logging
import torch
from transformers import AutoImageProcessor, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    """Encapsulates the feature extraction model and logic."""

    def __init__(self, model_ckpt: str):
        logging.info(f"Loading feature extraction model: {model_ckpt}...")
        self.processor = AutoImageProcessor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).eval().to(DEVICE)
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
    ):
        """Extracts and aggregates features from a batch of images."""
        inputs = self.processor(
            images=pil_imgs,
            return_tensors="pt",
            # Always resize first to the desired input resolution
            do_resize=True,
            size={"height": res, "width": res},
            # Then, perform a center crop (typically for evaluation or a tighter view)
            do_center_crop=docrop,
            # The crop_size determines the final output size after cropping
            crop_size={"height": res, "width": res},
        ).to(DEVICE)

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        cfg = self.model.config
        ps = cfg.patch_size
        num_reg = getattr(cfg, "num_register_tokens", 0)
        drop_front = 1 + num_reg
        h_p, w_p = res // ps, res // ps
        N_expected = h_p * w_p

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
        else:  # 'concat' or 'mean'
            feats = [_spatial_from_seq(hidden_states[li]) for li in layers]
            if agg_method == "concat":
                fused = torch.cat(feats, dim=-1)
            elif agg_method == "mean":
                fused = torch.stack(feats, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown aggregation method: '{agg_method}'")

        if is_cosine:
            fused = torch.nn.functional.normalize(fused, p=2, dim=-1)

        return fused.cpu().numpy(), (h_p, w_p)
