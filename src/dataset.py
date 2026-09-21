from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import pandas as pd


REQUIRED_COLUMNS = [
    "image_id",
    "split",
    "noise_level",
    "scene_id",
    "gt_id",
    "iso",
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "noise_measure",
    "mean_luminance",
]


class Dataset:
    def __init__(self, metadata_path: str, data_dir: str):
        self.metadata_path = Path(metadata_path)
        self.data_dir = Path(data_dir)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path.resolve()}")

        self.df = pd.read_csv(self.metadata_path)
        missing = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Metadata missing mandatory columns: {missing}")

        self.df = self.df[~self.df["image_id"].astype(str).str.startswith("#")].copy()

    def get_split(self, split_name: str) -> pd.DataFrame:
        return self.df[self.df["split"] == split_name].copy()

    def resolve_image_path(self, row: pd.Series) -> Path:
        split = str(row["split"])
        image_id = str(row["image_id"])
        noise_level = str(row["noise_level"])

        if split == "development":
            return self.data_dir / "development" / f"{image_id}.jpg"
        if split == "evaluation":
            return self.data_dir / "evaluation" / noise_level / f"{image_id}.jpg"
        raise ValueError(f"Unsupported split: {split}")

    def resolve_gt_path(self, row: pd.Series) -> Path:
        return self.data_dir / "ground_truth" / f"{row['gt_id']}.png"

    def load_sample(self, row: pd.Series) -> Tuple[Any, Any]:
        img_path = self.resolve_image_path(row)
        gt_path = self.resolve_gt_path(row)

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground Truth not found: {gt_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        if gt is None:
            raise ValueError(f"Failed to read Ground Truth: {gt_path}")

        if img.shape[:2] != gt.shape[:2]:
            raise ValueError(
                f"Shape mismatch for {row['image_id']}: image={img.shape[:2]}, gt={gt.shape[:2]}"
            )

        return img, gt

    @staticmethod
    def row_to_metadata(row: pd.Series) -> Dict[str, Any]:
        return {key: row[key] for key in REQUIRED_COLUMNS if key in row}
