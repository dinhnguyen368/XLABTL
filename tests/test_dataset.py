import cv2
import numpy as np
import pandas as pd
import pytest

from src.dataset import Dataset
from tools.validate_dataset import validate


HEADER = (
    "image_id,split,noise_level,scene_id,gt_id,iso,roi_x,roi_y,roi_w,roi_h,"
    "noise_measure,mean_luminance\n"
)


def test_dataset_missing_column(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("image_id,split\nimg1,development\n", encoding="utf-8")

    with pytest.raises(ValueError):
        Dataset(metadata, tmp_path)


def test_dataset_loader_and_shape(tmp_path):
    data_dir = tmp_path / "data"
    dev_dir = data_dir / "development"
    gt_dir = data_dir / "ground_truth"
    dev_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[10, :] = 255
    cv2.imwrite(str(dev_dir / "img1.jpg"), image)
    cv2.imwrite(str(gt_dir / "gt1.png"), gt)

    metadata = data_dir / "metadata.csv"
    metadata.write_text(
        HEADER + "img1,development,light,s1,gt1,100,0,0,10,10,1.0,120.0\n",
        encoding="utf-8",
    )

    dataset = Dataset(metadata, data_dir)
    img, loaded_gt = dataset.load_sample(dataset.df.iloc[0])

    assert img.shape[:2] == (32, 32)
    assert loaded_gt.shape == (32, 32)


def test_dataset_invalid_path(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        HEADER + "img1,development,light,s1,gt1,100,0,0,10,10,1.0,120.0\n",
        encoding="utf-8",
    )
    dataset = Dataset(metadata, tmp_path)

    with pytest.raises(FileNotFoundError):
        dataset.load_sample(dataset.df.iloc[0])


def test_validator_missing_column(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("image_id,split\nimg1,development\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        validate(str(metadata), str(tmp_path))
    assert exc.value.code == 1


def test_validator_duplicate_image_id(tmp_path):
    # The validator checks schema before accessing columns; this fixture includes all required fields.
    rows = []
    for level in ["light", "medium", "heavy"]:
        rows.append(f"img1,development,{level},s1,gt1,100,0,0,10,10,1.0,120.0\n")
    for level in ["light", "medium", "heavy"]:
        rows.append(f"img1,development,{level},s2,gt2,800,0,0,10,10,2.0,120.0\n")

    # Create the complete required structure count so duplicate ID is the first semantic failure.
    extra = []
    for scene_index in range(3, 11):
        for level in ["light", "medium", "heavy"]:
            split = "development" if scene_index <= 5 else "evaluation"
            extra.append(
                f"img{scene_index}_{level},{split},{level},s{scene_index},gt{scene_index},"
                f"100,0,0,10,10,1.0,120.0\n"
            )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(HEADER + "".join(rows + extra), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        validate(str(metadata), str(tmp_path))
    assert exc.value.code == 1


def _write_complete_invalid_metadata(tmp_path, bad_column, bad_value):
    rows = []
    image_counter = 1
    for scene_index in range(1, 11):
        split = "development" if scene_index <= 5 else "evaluation"
        for level in ["light", "medium", "heavy"]:
            iso = "100"
            rx, ry, rw, rh = "0", "0", "10", "10"
            noise = "1.0"
            lum = "120.0"
            if level == "medium":
                iso = "800"
            if level == "heavy":
                iso = "3200"
            values = {
                "iso": iso, "roi_x": rx, "roi_y": ry, "roi_w": rw, "roi_h": rh,
                "noise_measure": noise, "mean_luminance": lum
            }
            values[bad_column] = bad_value
            rows.append(
                f"img{image_counter}_{level},{split},{level},s{scene_index},gt{scene_index},"
                f"{values['iso']},{values['roi_x']},{values['roi_y']},{values['roi_w']},{values['roi_h']},"
                f"{values['noise_measure']},{values['mean_luminance']}\n"
            )
            image_counter += 1

    metadata = tmp_path / "metadata.csv"
    metadata.write_text(HEADER + "".join(rows), encoding="utf-8")
    return metadata


def test_validator_invalid_iso(tmp_path):
    metadata = _write_complete_invalid_metadata(tmp_path, "iso", "not_a_number")
    with pytest.raises(SystemExit) as exc:
        validate(str(metadata), str(tmp_path))
    assert exc.value.code == 1


def test_validator_invalid_roi(tmp_path):
    metadata = _write_complete_invalid_metadata(tmp_path, "roi_w", "0")
    with pytest.raises(SystemExit) as exc:
        validate(str(metadata), str(tmp_path))
    assert exc.value.code == 1
