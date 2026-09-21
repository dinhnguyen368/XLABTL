from pathlib import Path
from typing import Optional

from src.baseline import run_baseline
from src.metrics import compute_metrics
from src.proposed import run_proposed
from src.visualization import save_intermediate_results


def evaluate_single_image(
    img,
    gt,
    method: str,
    config: dict,
    tolerance: int,
    img_id: Optional[str] = None,
    save_dir: Optional[Path] = None,
):
    if method == "baseline":
        results = run_baseline(img, config)
    elif method == "proposed":
        results = run_proposed(img, config)
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics = compute_metrics(results["edges"], gt, tolerance=tolerance)

    if save_dir is not None and img_id is not None:
        save_intermediate_results(img_id, save_dir, results, gt)

    return metrics
