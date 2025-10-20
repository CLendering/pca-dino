import argparse


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Anomaly Detection Benchmark Framework"
    )

    # --- Dataset Arguments ---
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["mvtec_ad", "mvtec_ad2"],
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Root path to the dataset."
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specify categories to run, e.g., 'bottle screw'. If None, runs all.",
    )

    # --- Model & Feature Extraction Arguments ---
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="facebook/dinov2-base",
        help="HuggingFace model checkpoint for feature extraction.",
    )
    parser.add_argument(
        "--image_res", type=int, default=256, help="Image resolution for the model."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="""Size of the square patches to process. If None, the image is processed
        in full resolution.""",
    )
    parser.add_argument(
        "--patch_overlap",
        type=float,
        default=0.0,
        help="Overlap ratio between patches.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for feature extraction."
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default="concat",
        choices=["concat", "mean", "group"],
        help="Feature aggregation method across layers.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="-1,-2,-3",
        help="Comma-separated layer indices for 'concat' or 'mean' aggregation.",
    )
    parser.add_argument(
        "--grouped_layers",
        type=str,
        default=None,
        help="Layer groups for 'group' agg. Format: '-1,-2:-3,-4'.",
    )
    parser.add_argument(
        "--docrop",
        action="store_true",
        help="Apply center cropping during preprocessing.",
    )
    parser.add_argument(
        "--use_clahe",
        action="store_true",
        help="Apply CLAHE to the images.",
    )

    # --- Anomaly Detection (PCA) Arguments ---
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="Number of principal components to keep. Overrides --pca_ev.",
    )
    parser.add_argument(
        "--pca_ev",
        type=float,
        default=0.99,
        help="Explained variance to retain for PCA. Used if --pca_dim is None.",
    )
    parser.add_argument("--whiten", action="store_true", help="Apply whitening in PCA.")

    # --- Kernel PCA Arguments ---
    parser.add_argument(
        "--use_kernel_pca",
        action="store_true",
        help="Use Kernel PCA instead of standard PCA.",
    )
    parser.add_argument(
        "--kernel_pca_kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid", "cosine"],
        help="Kernel to use for Kernel PCA.",
    )
    parser.add_argument(
        "--kernel_pca_gamma",
        type=float,
        default=None,
        help="Gamma for rbf, poly and sigmoid kernels. If None, it's set to 1/n_features.",
    )

    # --- Scoring & Evaluation Arguments ---
    parser.add_argument(
        "--score_method",
        type=str,
        default="reconstruction",
        choices=["reconstruction", "mahalanobis", "cosine"],
        help="Anomaly scoring method.",
    )
    parser.add_argument(
        "--drop_k",
        type=int,
        default=0,
        help="Number of initial principal components to drop during reconstruction scoring.",
    )
    parser.add_argument(
        "--img_score_agg",
        type=str,
        default="p99",
        choices=["max", "mean", "p99"],
        help="Aggregation for image-level scores from pixel maps.",
    )
    parser.add_argument(
        "--pro_integration_limit",
        type=float,
        default=0.05,
        help="Integration limit for AU-PRO calculation.",
    )

    # --- Specular Reflection Filter Arguments ---
    parser.add_argument(
        "--use_specular_filter",
        action="store_true",
        help="Enable the specular reflection filter as a post-processing step.",
    )
    parser.add_argument(
        "--specular_tau",
        type=float,
        default=0.6,
        help="Binarization threshold for the specular mask.",
    )
    parser.add_argument(
        "--specular_size_threshold_factor",
        type=float,
        default=1.5,
        help="Size threshold factor for filtering specular anomalies.",
    )

    # --- Logistics ---
    parser.add_argument(
        "--outdir",
        type=str,
        default="./results",
        help="Directory to save results, logs, and visualizations.",
    )
    parser.add_argument(
        "--vis_count",
        type=int,
        default=3,
        help="Number of anomalous examples to visualize per category.",
    )
    parser.add_argument(
        "--no_log_file",
        action="store_true",
        help="Do not save a log file to the output directory.",
    )

    args = parser.parse_args()
    return args


def parse_layer_indices(arg_str: str):
    """Parses a comma-separated string of integers."""
    return [int(x.strip()) for x in arg_str.split(",")]


def parse_grouped_layers(arg_str: str):
    """Parses grouped layer indices from format like '-1,-2:-3,-4'."""
    if not arg_str:
        return []
    return [parse_layer_indices(group) for group in arg_str.split(":")]
