from pathlib import Path
import json
import csv
import cv2
import numpy as np


ROI_FILE = Path("outputs/configs/roi_coordinates.json")
OUTPUT_FILE = Path("data/metadata.csv")


SCENES = {
    "scene01": {
        "split": "development",
        "light": "dev_01_l",
        "medium": "dev_01_m",
        "heavy": "dev_01_h",
    },
    "scene02": {
        "split": "development",
        "light": "dev_02_l",
        "medium": "dev_02_m",
        "heavy": "dev_02_h",
    },
    "scene03": {
        "split": "development",
        "light": "dev_03_l",
        "medium": "dev_03_m",
        "heavy": "dev_03_h",
    },
    "scene04": {
        "split": "development",
        "light": "dev_04_l",
        "medium": "dev_04_m",
        "heavy": "dev_04_h",
    },
    "scene05": {
        "split": "development",
        "light": "dev_05_l",
        "medium": "dev_05_m",
        "heavy": "dev_05_h",
    },
    "scene06": {
        "split": "evaluation",
        "light": "eval_06_l",
        "medium": "eval_06_m",
        "heavy": "eval_06_h",
    },
    "scene07": {
        "split": "evaluation",
        "light": "eval_07_l",
        "medium": "eval_07_m",
        "heavy": "eval_07_h",
    },
    "scene08": {
        "split": "evaluation",
        "light": "eval_08_l",
        "medium": "eval_08_m",
        "heavy": "eval_08_h",
    },
    "scene09": {
        "split": "evaluation",
        "light": "eval_09_l",
        "medium": "eval_09_m",
        "heavy": "eval_09_h",
    },
    "scene10": {
        "split": "evaluation",
        "light": "eval_10_l",
        "medium": "eval_10_m",
        "heavy": "eval_10_h",
    },
}


NOISE_LEVELS = ["light", "medium", "heavy"]


def get_image_path(split, noise_level, image_id):
    if split == "development":
        return Path("data/development") / f"{image_id}.jpg"

    return (
        Path("data/evaluation")
        / noise_level
        / f"{image_id}.jpg"
    )


def analyze_roi(image_path, roi):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    x = int(roi["x"])
    y = int(roi["y"])
    w = int(roi["w"])
    h = int(roi["h"])

    height, width = image.shape[:2]

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(
            f"Invalid ROI for {image_path}: "
            f"x={x}, y={y}, w={w}, h={h}"
        )

    if x + w > width or y + h > height:
        raise ValueError(
            f"ROI outside image for {image_path}: "
            f"image={width}x{height}, "
            f"ROI=({x},{y},{w},{h})"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    roi_img = gray[y:y + h, x:x + w]

    if roi_img.size == 0:
        raise ValueError(f"Empty ROI: {image_path}")

    mean_luminance = float(np.mean(roi_img))

    blurred = cv2.GaussianBlur(
        roi_img,
        (5, 5),
        0
    )

    residual = (
        roi_img.astype(np.float32)
        - blurred.astype(np.float32)
    )

    noise_measure = float(np.std(residual))

    return mean_luminance, noise_measure


def main():
    if not ROI_FILE.exists():
        raise FileNotFoundError(
            f"ROI file not found: {ROI_FILE}"
        )

    with open(ROI_FILE, "r", encoding="utf-8") as f:
        rois = json.load(f)

    rows = []

    print("=== BUILD METADATA FROM IMAGES + ROI ===")
    print()

    for scene_id, scene in SCENES.items():

        if scene_id not in rois:
            raise ValueError(
                f"Missing ROI for {scene_id}"
            )

        roi = rois[scene_id]

        print(f"\n--- {scene_id} ---")

        for noise_level in NOISE_LEVELS:

            image_id = scene[noise_level]

            image_path = get_image_path(
                scene["split"],
                noise_level,
                image_id
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            mean_luminance, noise_measure = analyze_roi(
                image_path,
                roi
            )

            print(
                f"{image_id}.jpg | "
                f"{noise_level:6s} | "
                f"luminance={mean_luminance:.4f} | "
                f"noise={noise_measure:.4f}"
            )

            rows.append(
                {
                    "image_id": image_id,
                    "split": scene["split"],
                    "noise_level": noise_level,
                    "scene_id": scene_id,
                    "gt_id": f"{scene_id}_gt",

                    # CHƯA CÓ ISO THẬT
                    "iso": "",

                    "roi_x": roi["x"],
                    "roi_y": roi["y"],
                    "roi_w": roi["w"],
                    "roi_h": roi["h"],

                    "noise_measure": noise_measure,
                    "mean_luminance": mean_luminance,

                    "notes": (
                        "real-world capture; "
                        "ISO metadata unavailable"
                    ),
                }
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
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
        "notes",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print(f"Created: {OUTPUT_FILE}")
    print(f"Rows   : {len(rows)}")
    print("=" * 60)

    if len(rows) != 30:
        raise RuntimeError(
            f"Expected 30 rows, got {len(rows)}"
        )

    print()
    print(
        "NOTE: ISO column is intentionally empty because "
        "the current JPG files do not contain ISO metadata."
    )


if __name__ == "__main__":
    main()