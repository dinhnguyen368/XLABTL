"""Single-image demo for the final project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_parameters import NoiseAdaptiveConfig
from src.baseline import run_baseline
from src.config import CONFIG
from src.dataset import Dataset
from src.metrics import compute_metrics
from src.proposed import run_proposed


def compute_noise_proxy(gray: np.ndarray, row) -> float:
    x, y = int(row["roi_x"]), int(row["roi_y"])
    w, h = int(row["roi_w"]), int(row["roi_h"])
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        raise ValueError("ROI is empty.")
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    residual = roi.astype(np.float32) - blurred.astype(np.float32)
    return float(np.std(residual))


def run_demo(image_id: str = "eval_09_h") -> None:
    dataset = Dataset(CONFIG["paths"]["metadata_csv"], CONFIG["paths"]["data_dir"])
    rows = dataset.df[dataset.df["image_id"].astype(str) == str(image_id)]
    if len(rows) != 1:
        raise ValueError(f"Không tìm thấy duy nhất image_id={image_id}.")
    row = rows.iloc[0]
    if str(row["split"]) != "evaluation":
        raise ValueError("Demo phải dùng ảnh Evaluation.")

    rules_path = ROOT / CONFIG["paths"]["configs_dir"] / "e2_adaptive_rules.json"
    if not rules_path.exists():
        raise FileNotFoundError("Chưa có frozen E2 rules. Hãy chạy: python main.py --mode E2")

    adaptive = NoiseAdaptiveConfig(rules_path)
    image, gt = dataset.load_sample(row)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    proxy = compute_noise_proxy(gray, row)
    rule = adaptive.get_rule(proxy)
    proposed_config = adaptive.get_config(proxy, CONFIG["proposed_default"])

    baseline = run_baseline(image, CONFIG["baseline"])
    proposed = run_proposed(image, proposed_config)

    tol = int(CONFIG["evaluation"]["metric_tolerance"])
    baseline_f1 = float(compute_metrics(baseline["edges"], gt, tolerance=tol)["strict_f1"])
    proposed_f1 = float(compute_metrics(proposed["edges"], gt, tolerance=tol)["strict_f1"])

    x, y = int(row["roi_x"]), int(row["roi_y"])
    w, h = int(row["roi_w"]), int(row["roi_h"])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_rgb = rgb.copy()
    cv2.rectangle(display_rgb, (x, y), (x+w, y+h), (255, 180, 0), 1)

    panels = [
        ("Original + ROI", display_rgb, None),
        ("Ground Truth", gt, "gray"),
        ("Baseline Canny", baseline["edges"], "gray"),
        ("Bilateral Filter", proposed["denoised"], "gray"),
        ("Proposed Canny", proposed["edges"], "gray"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, (title, data, cmap) in zip(axes, panels):
        if cmap == "gray":
            ax.imshow(data, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")

    delta = proposed_f1 - baseline_f1
    fig.suptitle(
        f"{image_id} | noise proxy={proxy:.4f} | rule={rule['name']} | "
        f"Baseline F1={baseline_f1:.4f} | Proposed F1={proposed_f1:.4f} | Delta={delta:+.4f}"
    )
    fig.tight_layout()

    out_path = ROOT / CONFIG["paths"]["figures_dir"] / f"demo_{image_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)

    print("=" * 70)
    print("DEMO SINGLE IMAGE")
    print("=" * 70)
    print(f"Image         : {image_id}")
    print(f"Scene         : {row['scene_id']}")
    print(f"Noise level   : {row['noise_level']}")
    print(f"Noise proxy   : {proxy:.6f}")
    print(f"Selected rule : {rule['name']}")
    print(f"Bilateral     : d={rule['d']}, sigma_color={rule['sigma_color']}, sigma_space={rule['sigma_space']}")
    print(f"Canny         : {CONFIG['baseline']['canny_threshold1']}/{CONFIG['baseline']['canny_threshold2']}")
    print(f"Baseline F1   : {baseline_f1:.6f}")
    print(f"Proposed F1   : {proposed_f1:.6f}")
    print(f"Delta         : {delta:+.6f}")
    print(f"Figure        : {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-image demo")
    parser.add_argument("--image", default="eval_09_h", help="Evaluation image ID, e.g. eval_09_h or eval_06_h")
    args = parser.parse_args()
    run_demo(args.image)