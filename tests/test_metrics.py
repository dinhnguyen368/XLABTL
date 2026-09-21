import numpy as np

from src.metrics import compute_metrics


def test_metrics_perfect_prediction():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[10, :] = 255
    pred = gt.copy()

    metrics = compute_metrics(pred, gt, tolerance=0)

    assert np.isclose(metrics["strict_f1"], 1.0)
    assert np.isclose(metrics["strict_precision"], 1.0)
    assert np.isclose(metrics["strict_recall"], 1.0)


def test_metrics_empty_prediction():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[10, :] = 255
    pred = np.zeros((20, 20), dtype=np.uint8)

    metrics = compute_metrics(pred, gt, tolerance=0)

    assert np.isclose(metrics["strict_f1"], 0.0)


def test_metrics_known_case():
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[5, 0:5] = 255

    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[5, 0:2] = 255  # TP = 2
    pred[6, 0:2] = 255  # FP = 2

    metrics = compute_metrics(pred, gt, tolerance=0)

    assert np.isclose(metrics["strict_precision"], 0.5, atol=0.01)
    assert np.isclose(metrics["strict_recall"], 0.4, atol=0.01)


def test_metrics_tolerance_shift():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[10, :] = 255

    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[11, :] = 255

    metrics = compute_metrics(pred, gt, tolerance=1)

    assert np.isclose(metrics["strict_f1"], 0.0)
    assert np.isclose(metrics["tol_f1"], 1.0, atol=1e-5)
