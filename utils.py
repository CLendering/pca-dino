import argparse
import json
import logging
import os
import numpy as np
import torch


def setup_logging(outdir: str, save_log: bool = True):
    """Configures the logging for console and file output."""
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # File handler
    if save_log:
        log_file = os.path.join(outdir, "run.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)

    logging.info("Logging configured.")


def save_config(args: argparse.Namespace):
    """Saves the run configuration to a JSON file."""
    config_path = os.path.join(args.outdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Configuration saved to {config_path}")


def min_max_norm(x, eps=1e-8):
    is_torch = torch.is_tensor(x)

    if is_torch:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = torch.amin(x, dim=(-1, -2), keepdim=True)
        x_max = torch.amax(x, dim=(-1, -2), keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + eps)
        return x_norm.clamp(0.0, 1.0)
    else:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = np.min(x, axis=(-1, -2), keepdims=True)
        x_max = np.max(x, axis=(-1, -2), keepdims=True)
        x_norm = (x - x_min) / (x_max - x_min + eps)
        return np.clip(x_norm, 0.0, 1.0)
