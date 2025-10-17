import argparse
import json
import logging
import os


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
