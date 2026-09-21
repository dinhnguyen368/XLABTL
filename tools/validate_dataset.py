import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils import setup_logger


REQUIRED_COLUMNS = [
    "image_id",
    "split",
    "noise_level",
    "scene_id",
    "gt_id",
    "iso",  # retained for metadata compatibility; not used adaptively
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "noise_measure",
    "mean_luminance",
]

VALID_SPLITS = {
    "development",
    "evaluation",
}

VALID_NOISE = {
    "light",
    "medium",
    "heavy",
}

ROI_COLUMNS = [
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
]

NUMERIC_COLUMNS = [
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "noise_measure",
    "mean_luminance",
]


def _fail(logger, message: str) -> None:
    logger.critical(message)
    raise SystemExit(1)


def resolve_image_path(
    data_dir: Path,
    row: pd.Series,
) -> Path:

    split = str(row["split"])
    level = str(row["noise_level"])
    image_id = str(row["image_id"])

    if split == "development":
        return (
            data_dir
            / "development"
            / f"{image_id}.jpg"
        )

    return (
        data_dir
        / "evaluation"
        / level
        / f"{image_id}.jpg"
    )


def validate(
    meta_path: str = "data/metadata.csv",
    data_dir: str = "data",
) -> None:

    logger = setup_logger(
        "ValidateDataset"
    )

    logger.info(
        "=== STRICT DATASET VALIDATION ==="
    )

    metadata_path = Path(meta_path)
    root = Path(data_dir)

    if not metadata_path.exists():
        _fail(
            logger,
            f"Missing metadata: "
            f"{metadata_path.resolve()}",
        )

    try:
        df = pd.read_csv(
            metadata_path
        )
    except Exception as exc:
        _fail(
            logger,
            f"Failed to read metadata CSV: {exc}",
        )

    # Schema MUST be checked before any column access.
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        _fail(
            logger,
            "Schema validation failed. "
            f"Missing columns: {missing}",
        )

    df = df[
        ~df["image_id"]
        .astype(str)
        .str.startswith("#")
    ].copy()

    if df.empty:
        _fail(
            logger,
            "Metadata contains no active data rows.",
        )

    if not set(
        df["split"].dropna().unique()
    ).issubset(VALID_SPLITS):

        _fail(
            logger,
            "Invalid split values. "
            f"Allowed={VALID_SPLITS}, "
            f"found={df['split'].unique()}",
        )

    if not set(
        df["noise_level"].dropna().unique()
    ).issubset(VALID_NOISE):

        _fail(
            logger,
            "Invalid noise_level values. "
            f"Allowed={VALID_NOISE}, "
            f"found={df['noise_level'].unique()}",
        )

    if (
        df["image_id"].isna().any()
        or df["scene_id"].isna().any()
        or df["gt_id"].isna().any()
    ):
        _fail(
            logger,
            "image_id, scene_id, and gt_id "
            "must not contain missing values.",
        )

    if not df["image_id"].is_unique:
        _fail(
            logger,
            "Duplicate image_id found.",
        )

    if df.duplicated(
        subset=[
            "scene_id",
            "noise_level",
        ]
    ).any():

        _fail(
            logger,
            "Duplicate "
            "(scene_id, noise_level) "
            "combination found.",
        )

    for column in NUMERIC_COLUMNS:

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if (
            converted.isna().any()
            or np.isinf(
                converted.to_numpy()
            ).any()
        ):
            _fail(
                logger,
                f"Column '{column}' contains "
                "NaN, Inf, or non-numeric values.",
            )

        df[column] = converted

    for column in ROI_COLUMNS:

        values = df[column].to_numpy(
            dtype=float
        )

        if not np.all(
            values == np.floor(values)
        ):
            _fail(
                logger,
                f"ROI column '{column}' "
                "must be integer-valued.",
            )

    if (
        df["roi_w"] <= 0
    ).any() or (
        df["roi_h"] <= 0
    ).any():

        _fail(
            logger,
            "ROI width and height must be > 0.",
        )

    if (
        df["roi_x"] < 0
    ).any() or (
        df["roi_y"] < 0
    ).any():

        _fail(
            logger,
            "ROI x/y coordinates must be >= 0.",
        )

    if (
        df["noise_measure"] < 0
    ).any():

        _fail(
            logger,
            "noise_measure cannot be negative.",
        )

    if (
        df["mean_luminance"] < 0
    ).any() or (
        df["mean_luminance"] > 255
    ).any():

        _fail(
            logger,
            "mean_luminance must lie in [0, 255].",
        )

    if len(df) != 30:
        _fail(
            logger,
            f"Dataset must contain exactly "
            f"30 images. Found {len(df)}.",
        )

    scenes = sorted(
        df["scene_id"]
        .unique()
        .tolist()
    )

    if len(scenes) != 10:
        _fail(
            logger,
            "Dataset must contain exactly "
            f"10 unique scenes. Found {len(scenes)}.",
        )

    dev_scenes = set(
        df.loc[
            df["split"] == "development",
            "scene_id",
        ].unique()
    )

    eval_scenes = set(
        df.loc[
            df["split"] == "evaluation",
            "scene_id",
        ].unique()
    )

    if (
        len(dev_scenes) != 5
        or len(eval_scenes) != 5
    ):
        _fail(
            logger,
            "Expected 5 development and "
            "5 evaluation scenes. "
            f"Found {len(dev_scenes)} / "
            f"{len(eval_scenes)}.",
        )

    if not dev_scenes.isdisjoint(
        eval_scenes
    ):
        _fail(
            logger,
            "A scene_id exists in both "
            "development and evaluation.",
        )

    errors = []

    for scene_id in scenes:

        scene_df = df[
            df["scene_id"] == scene_id
        ].copy()

        if len(scene_df) != 3:

            errors.append(
                f"{scene_id}: expected "
                f"3 rows, found "
                f"{len(scene_df)}"
            )

            continue

        if set(
            scene_df["noise_level"]
        ) != VALID_NOISE:

            errors.append(
                f"{scene_id}: must contain "
                "light/medium/heavy exactly"
            )

        if scene_df["split"].nunique() != 1:

            errors.append(
                f"{scene_id}: split must be "
                "identical across levels"
            )

        if scene_df["gt_id"].nunique() != 1:

            errors.append(
                f"{scene_id}: all levels must "
                "share exactly one gt_id"
            )

        if (
            scene_df[
                ROI_COLUMNS
            ]
            .drop_duplicates()
            .shape[0]
            != 1
        ):

            errors.append(
                f"{scene_id}: ROI coordinates "
                "must be identical across levels"
            )

        shapes = []

        for _, row in scene_df.iterrows():

            image_path = resolve_image_path(
                root,
                row,
            )

            if not image_path.exists():

                errors.append(
                    f"{scene_id}/"
                    f"{row['noise_level']}: "
                    f"missing image "
                    f"{image_path}"
                )

                continue

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR,
            )

            if image is None:

                errors.append(
                    f"{scene_id}/"
                    f"{row['noise_level']}: "
                    f"unreadable image "
                    f"{image_path}"
                )

                continue

            shapes.append(
                image.shape[:2]
            )

            x = int(row["roi_x"])
            y = int(row["roi_y"])
            w = int(row["roi_w"])
            h = int(row["roi_h"])

            height, width = image.shape[:2]

            if (
                x + w > width
                or y + h > height
            ):

                errors.append(
                    f"{row['image_id']}: "
                    f"ROI out of bounds for "
                    f"image {width}x{height}"
                )

        if (
            len(shapes) == 3
            and not (
                shapes[0]
                == shapes[1]
                == shapes[2]
            )
        ):

            errors.append(
                f"{scene_id}: "
                "Light/Medium/Heavy dimensions "
                f"differ: {shapes}"
            )

    if errors:

        for item in errors:
            logger.error(item)

        _fail(
            logger,
            "Dataset structural validation "
            f"failed with {len(errors)} issue(s).",
        )

    # Final image-to-GT shape/readability checks.
    for _, row in df.iterrows():

        image_path = resolve_image_path(
            root,
            row,
        )

        gt_path = (
            root
            / "ground_truth"
            / f"{row['gt_id']}.png"
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        gt = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:

            errors.append(
                f"Unreadable image: "
                f"{image_path}"
            )

            continue

        if gt is None:

            errors.append(
                f"Unreadable/missing GT: "
                f"{gt_path}"
            )

            continue

        if image.shape[:2] != gt.shape[:2]:

            errors.append(
                f"Shape mismatch "
                f"{row['image_id']}: "
                f"image={image.shape[:2]}, "
                f"gt={gt.shape[:2]}"
            )

    if errors:

        for item in errors:
            logger.error(item)

        _fail(
            logger,
            "Dataset file validation "
            f"failed with {len(errors)} issue(s).",
        )

    logger.info(
        "SUCCESS: Dataset satisfies schema, "
        "10-scene pairing, ROI, split, GT, "
        "dimensions, and noise-proxy constraints."
    )


if __name__ == "__main__":
    validate()