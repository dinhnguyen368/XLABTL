import cv2
import numpy as np


def apply_canny(img_gray: np.ndarray, threshold1: float, threshold2: float) -> np.ndarray:
    if img_gray.ndim != 2:
        raise ValueError("Canny expects a grayscale 2D image.")
    return cv2.Canny(img_gray, float(threshold1), float(threshold2))
