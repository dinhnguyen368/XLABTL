import cv2
import numpy as np


def apply_bilateral_filter(
    img_gray: np.ndarray,
    d: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    if img_gray.ndim != 2:
        raise ValueError("Bilateral filter expects a grayscale 2D image.")
    return cv2.bilateralFilter(
        img_gray,
        int(d),
        float(sigma_color),
        float(sigma_space),
    )
