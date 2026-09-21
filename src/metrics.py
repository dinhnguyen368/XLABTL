import cv2
import numpy as np


def _safe_prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return precision, recall, f1, iou


def compute_metrics(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2):
    """Compute strict metrics plus a supplementary tolerance metric.

    Strict F1 is the primary metric. Tolerance-based metrics are supplementary.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")

    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)

    tp_strict = int(np.logical_and(pred_bin == 1, gt_bin == 1).sum())
    fp_strict = int(np.logical_and(pred_bin == 1, gt_bin == 0).sum())
    fn_strict = int(np.logical_and(pred_bin == 0, gt_bin == 1).sum())
    p_strict, r_strict, f1_strict, iou_strict = _safe_prf(
        tp_strict, fp_strict, fn_strict
    )

    if tolerance > 0:
        k = int(tolerance) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        gt_dilated = cv2.dilate(gt_bin, kernel)
        pred_dilated = cv2.dilate(pred_bin, kernel)

        tp_tol = int(np.logical_and(pred_bin == 1, gt_dilated == 1).sum())
        fp_tol = int(np.logical_and(pred_bin == 1, gt_dilated == 0).sum())
        fn_tol = int(np.logical_and(gt_bin == 1, pred_dilated == 0).sum())
        p_tol, r_tol, f1_tol, iou_tol = _safe_prf(tp_tol, fp_tol, fn_tol)
    else:
        p_tol, r_tol, f1_tol, iou_tol = p_strict, r_strict, f1_strict, iou_strict

    return {
        "strict_precision": p_strict,
        "strict_recall": r_strict,
        "strict_f1": f1_strict,
        "strict_iou": iou_strict,
        "tol_precision": p_tol,
        "tol_recall": r_tol,
        "tol_f1": f1_tol,
        "tol_iou": iou_tol,
    }
