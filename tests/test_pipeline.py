import numpy as np

from src.pipeline import evaluate_single_image


def test_pipeline_baseline_runs():
    img = np.random.default_rng(42).integers(0, 256, (50, 50, 3), dtype=np.uint8)
    gt = np.zeros((50, 50), dtype=np.uint8)
    config = {"canny_threshold1": 50, "canny_threshold2": 150}

    metrics = evaluate_single_image(img, gt, "baseline", config, tolerance=1)

    assert "strict_f1" in metrics
    assert 0.0 <= metrics["strict_f1"] <= 1.0


def test_pipeline_proposed_runs():
    img = np.random.default_rng(42).integers(0, 256, (50, 50, 3), dtype=np.uint8)
    gt = np.zeros((50, 50), dtype=np.uint8)
    config = {
        "bilateral_d": 3,
        "bilateral_sigma_color": 25,
        "bilateral_sigma_space": 25,
        "canny_threshold1": 50,
        "canny_threshold2": 150,
    }

    metrics = evaluate_single_image(img, gt, "proposed", config, tolerance=1)

    assert "strict_f1" in metrics
    assert 0.0 <= metrics["strict_f1"] <= 1.0
