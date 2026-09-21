import cv2
import numpy as np

from src.denoising import apply_bilateral_filter
from src.edge_detection import apply_canny


def run_proposed(img: np.ndarray, config: dict) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    denoised = apply_bilateral_filter(
        gray,
        config["bilateral_d"],
        config["bilateral_sigma_color"],
        config["bilateral_sigma_space"],
    )
    edges = apply_canny(denoised, config["canny_threshold1"], config["canny_threshold2"])
    return {"original": img, "gray": gray, "denoised": denoised, "edges": edges}
