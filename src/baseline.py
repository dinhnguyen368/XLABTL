import cv2
import numpy as np

from src.edge_detection import apply_canny


def run_baseline(img: np.ndarray, config: dict) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = apply_canny(gray, config["canny_threshold1"], config["canny_threshold2"])
    return {"original": img, "gray": gray, "edges": edges}
