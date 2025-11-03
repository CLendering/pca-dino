import glob
import os
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import logging


class BaseDatasetHandler:
    """Abstract base class for dataset handlers."""

    def __init__(self, root_path, category):
        self.root_path = Path(root_path)
        self.category = category
        self.category_path = self.root_path / category

    def get_train_paths(self):
        raise NotImplementedError

    def get_validation_paths(self):
        return []  # Default: no validation set

    def get_test_paths(self):
        raise NotImplementedError

    def get_ground_truth_path(self, test_path: str):
        raise NotImplementedError

    def get_ground_truth_mask(self, test_path: str, res: tuple):
        gt_path_str = self.get_ground_truth_path(test_path)
        if not gt_path_str or not os.path.exists(gt_path_str):
            return np.zeros((res[1], res[0]), dtype=np.uint8)

        mask = (
            Image.open(gt_path_str)
            .convert("L")
            .resize(res, Image.Resampling.NEAREST)  # res is (W, H)
        )
        return (np.array(mask) > 0).astype(np.uint8)  # returns (H, W) array


class MVTecADDataset(BaseDatasetHandler):
    """Handler for the original MVTec AD dataset structure."""

    def get_train_paths(self):
        return sorted(glob.glob(str(self.category_path / "train" / "good" / "*.png")))

    def get_test_paths(self):
        return sorted(glob.glob(str(self.category_path / "test" / "*" / "*.png")))

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        return str(
            self.category_path / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
        )


class MVTecAD2Dataset(BaseDatasetHandler):
    """Handler for the MVTec AD 2 dataset structure."""

    def get_train_paths(self):
        return sorted(glob.glob(str(self.category_path / "train" / "good" / "*.png")))

    def get_validation_paths(self):
        return sorted(
            glob.glob(str(self.category_path / "validation" / "good" / "*.png"))
        )

    def get_test_paths(self):
        return sorted(
            glob.glob(str(self.category_path / "test_public" / "*" / "*.png"))
        )

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        return str(
            self.category_path
            / "test_public"
            / "ground_truth"
            / p.parent.name
            / f"{p.stem}_mask.png"
        )


class VisADataset(BaseDatasetHandler):
    """
    Handler for the VisA dataset structure.
    This handler relies on a CSV split file located at:
    {root_path}/split_csv/2cls_fewshot.csv
    """

    def __init__(self, root_path, category):
        super().__init__(root_path, category)
        self.train_paths = []
        self.test_paths = []
        self.test_path_to_mask_map = {}

        # The split file is assumed to be in the root_path, parallel to categories
        split_file = self.root_path / "split_csv" / "1cls.csv"
        if not split_file.exists():
            logging.error(f"VisA split file not found at: {split_file}")
            raise FileNotFoundError(f"VisA split file not found at: {split_file}")

        try:
            df = pd.read_csv(split_file)
            # Filter for the current category
            self.df = df[df["object"] == self.category].copy()

            if self.df.empty:
                logging.warning(
                    f"No entries found for category '{self.category}' in {split_file}"
                )
                return

            # Create absolute paths by joining root_path with relative paths from CSV
            self.df["image_abs"] = self.df["image"].apply(lambda x: self.root_path / x)
            self.df["mask_abs"] = self.df["mask"].apply(
                # Handle empty mask paths (for normal images)
                lambda x: self.root_path / x if pd.notna(x) and x else None
            )

            # Get train paths (normal images only)
            self.train_paths = (
                self.df[(self.df["split"] == "train") & (self.df["label"] == "normal")][
                    "image_abs"
                ]
                .astype(str)
                .tolist()
            )

            # Get test paths (all images labeled 'test')
            test_df = self.df[self.df["split"] == "test"]
            self.test_paths = test_df["image_abs"].astype(str).tolist()

            # Create lookup map for test images (image_path -> mask_path)
            self.test_path_to_mask_map = pd.Series(
                test_df["mask_abs"].values,
                index=test_df["image_abs"].astype(str).values,
            ).to_dict()

            # Sort for consistency
            self.train_paths.sort()
            self.test_paths.sort()

        except Exception as e:
            logging.error(
                f"Failed to load or process VisA split file {split_file}: {e}"
            )
            raise

    def get_train_paths(self):
        """Returns a list of 'normal' training image paths from the split file."""
        return self.train_paths

    def get_test_paths(self):
        """Returns a list of 'test' image paths from the split file."""
        return self.test_paths

    def get_ground_truth_path(self, test_path: str):
        """Looks up the mask path for a given test image path."""
        mask_path = self.test_path_to_mask_map.get(str(test_path))

        # mask_path can be None or pd.NA if the test image is 'normal'
        if mask_path is None or pd.isna(mask_path):
            return None

        return str(mask_path)


def get_dataset_handler(name: str, root_path: str, category: str) -> BaseDatasetHandler:
    """Factory function to get the correct dataset handler."""
    if name == "mvtec_ad":
        return MVTecADDataset(root_path, category)
    elif name == "mvtec_ad2":
        return MVTecAD2Dataset(root_path, category)
    elif name == "visa":
        return VisADataset(root_path, category)
    elif name == "blade30":
        return Blade30Dataset(root_path, category)
    else:
        raise ValueError(f"Unknown dataset: {name}")


class Blade30Dataset(BaseDatasetHandler):
    """Handler for the original MVTec AD dataset structure."""

    def get_train_paths(self):
        return sorted(glob.glob(str(self.category_path / "train" / "good" / "*.jpg")))

    def get_test_paths(self):
        return sorted(glob.glob(str(self.category_path / "test" / "*" / "*.jpg")))

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        return str(
            self.category_path / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
        )
