import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"FAIL: Cannot read image: {path}")
    return image


def process_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        raise ValueError("ROI is empty.")

    mean_luminance = float(np.mean(roi))
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    residual = roi.astype(np.float32) - blurred.astype(np.float32)
    noise_measure = float(np.std(residual))
    return mean_luminance, noise_measure


def analyze_paired_scene(light_path: str, medium_path: str, heavy_path: str) -> Dict:
    img_l = load_image(light_path)
    img_m = load_image(medium_path)
    img_h = load_image(heavy_path)

    if not (img_l.shape == img_m.shape == img_h.shape):
        raise ValueError(
            "FAIL: Light/Medium/Heavy image dimensions must be identical. "
            f"Found {img_l.shape[:2]}, {img_m.shape[:2]}, {img_h.shape[:2]}"
        )

    print("Select a homogeneous, non-textured background ROI on the Light image.")
    print("Press ENTER to confirm. The exact same coordinates will be used on Medium and Heavy.")
    roi_rect = cv2.selectROI("Select Background ROI (Light)", img_l, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi_rect)
    img_h_px, img_w_px = img_l.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("FAIL: ROI width and height must be > 0.")
    if x < 0 or y < 0 or x + w > img_w_px or y + h > img_h_px:
        raise ValueError("FAIL: ROI is outside image boundaries.")

    lum_l, noise_l = process_roi(img_l, x, y, w, h)
    lum_m, noise_m = process_roi(img_m, x, y, w, h)
    lum_h, noise_h = process_roi(img_h, x, y, w, h)

    diff_m = abs(lum_m - lum_l) / max(abs(lum_l), 1e-5) * 100.0
    diff_h = abs(lum_h - lum_l) / max(abs(lum_l), 1e-5) * 100.0

    result = {
        "roi_x": x,
        "roi_y": y,
        "roi_w": w,
        "roi_h": h,
        "light": {"mean_luminance": lum_l, "noise_measure": noise_l},
        "medium": {
            "mean_luminance": lum_m,
            "noise_measure": noise_m,
            "relative_luminance_difference_percent": diff_m,
        },
        "heavy": {
            "mean_luminance": lum_h,
            "noise_measure": noise_h,
            "relative_luminance_difference_percent": diff_h,
        },
    }

    print("\n" + "=" * 60)
    print("ROI-BASED NOISE ANALYSIS")
    print("=" * 60)
    print(f"ROI: x={x}, y={y}, w={w}, h={h}")
    print(f"Light  | luminance={lum_l:.2f} | noise_proxy={noise_l:.4f}")
    print(
        f"Medium | luminance={lum_m:.2f} | noise_proxy={noise_m:.4f} "
        f"| luminance_diff={diff_m:.2f}%"
    )
    print(
        f"Heavy  | luminance={lum_h:.2f} | noise_proxy={noise_h:.4f} "
        f"| luminance_diff={diff_h:.2f}%"
    )
    print("Note: noise_measure is an image-level high-frequency residual proxy, not pure sensor noise.")
    print("Operational luminance criterion: relative difference <= 10%.")
    print("Do NOT discard a scene solely because noise_proxy is non-monotonic.")
    print("=" * 60)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a paired Light/Medium/Heavy scene.")
    parser.add_argument("-l", "--light", required=True)
    parser.add_argument("-m", "--medium", required=True)
    parser.add_argument("-H", "--heavy", required=True)
    parser.add_argument("--output-json", default=None, help="Optional path for structured JSON output.")
    args = parser.parse_args()

    try:
        result = analyze_paired_scene(args.light, args.medium, args.heavy)
    except Exception as exc:
        print(str(exc))
        raise SystemExit(1)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved JSON analysis: {output_path}")


if __name__ == "__main__":
    main()
