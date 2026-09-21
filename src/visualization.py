from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_intermediate_results(img_id: str, out_dir: Path, results_dict: Dict, gt: np.ndarray) -> None:
    out_dir = Path(out_dir) / str(img_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, img in results_dict.items():
        if img is None:
            continue
        if key == "original" and img.ndim == 3:
            cv2.imwrite(str(out_dir / f"{key}.jpg"), img)
        else:
            cv2.imwrite(str(out_dir / f"{key}.png"), img)

    cv2.imwrite(str(out_dir / "ground_truth.png"), gt)

    if "edges" in results_dict:
        pred = (results_dict["edges"] > 127).astype(np.uint8)
        gt_bin = (gt > 127).astype(np.uint8)

        error_map = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        error_map[(pred == 1) & (gt_bin == 1)] = [255, 255, 255]
        error_map[(pred == 1) & (gt_bin == 0)] = [0, 0, 255]
        error_map[(pred == 0) & (gt_bin == 1)] = [255, 0, 0]
        cv2.imwrite(str(out_dir / "error_map.png"), error_map)


def plot_metric_vs_noise(csv_path: Path, out_path: Path, metric: str = "strict_f1", title: str = "Metric vs Noise") -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Cannot plot empty CSV: {csv_path}")

    noise_order = ["light", "medium", "heavy"]
    df["noise_level"] = pd.Categorical(df["noise_level"], categories=noise_order, ordered=True)
    df = df.sort_values("noise_level")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    if f"baseline_{metric}" in df.columns:
        plt.plot(df["noise_level"], df[f"baseline_{metric}"], marker="o", label="Baseline")
    if f"proposed_{metric}" in df.columns:
        plt.plot(df["noise_level"], df[f"proposed_{metric}"], marker="s", label="Proposed")

    plt.title(title)
    plt.xlabel("Noise Level (Acquisition Condition Proxy)")
    plt.ylabel(metric.upper())
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
