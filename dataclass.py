import glob
import os
from pathlib import Path
import numpy as np
from PIL import Image


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


# TODO: Data handeling in metrics will not work for VisA, as there is no 'good'subfolder.
class VisADataset(BaseDatasetHandler):
    """Handler for the VisA dataset structure."""

    def get_train_paths(self):
        return sorted(
            glob.glob(str(self.category_path / "Data" / "Images" / "Good" / "*.JPG"))
        )

    def get_test_paths(self):
        return sorted(
            glob.glob(str(self.category_path / "Data" / "Images" / "Anomaly" / "*.JPG"))
        )

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        return str(self.category_path / "Data" / "Masks" / f"{p.stem}.png")


def get_dataset_handler(name: str, root_path: str, category: str) -> BaseDatasetHandler:
    """Factory function to get the correct dataset handler."""
    if name == "mvtec_ad":
        return MVTecADDataset(root_path, category)
    elif name == "mvtec_ad2":
        return MVTecAD2Dataset(root_path, category)
    elif name == "visa":
        return VisADataset(root_path, category)
    else:
        raise ValueError(f"Unknown dataset: {name}")
