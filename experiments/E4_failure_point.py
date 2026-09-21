import hashlib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.adaptive_parameters import NoiseAdaptiveConfig
from src.config import CONFIG
from src.dataset import Dataset
from src.pipeline import evaluate_single_image
from src.utils import ensure_directories, setup_logger
from tools.validate_dataset import validate


def deterministic_rng(
    scene_id: str,
    sigma: float,
    global_seed: int,
) -> np.random.Generator:
    seed_string = f"{scene_id}|{sigma}|{global_seed}"
    digest = hashlib.sha256(
        seed_string.encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**32)
    return np.random.default_rng(seed)


def run_e4() -> None:
    validate()
    ensure_directories()

    logger = setup_logger(
        "E4",
        Path(CONFIG["paths"]["logs_dir"]) / "E4.log",
    )

    logger.info(
        "=== E4: CONTROLLED GAUSSIAN ROBUSTNESS STRESS TEST ==="
    )

    dataset = Dataset(
        CONFIG["paths"]["metadata_csv"],
        CONFIG["paths"]["data_dir"],
    )

    df_eval_light = dataset.get_split("evaluation")
    df_eval_light = df_eval_light[
        df_eval_light["noise_level"] == "light"
    ].copy()

    if len(df_eval_light) != 5:
        logger.critical(
            f"E4 expects 5 Light evaluation scenes, "
            f"found {len(df_eval_light)}."
        )
        raise SystemExit(1)

    rules_path = (
        Path(CONFIG["paths"]["configs_dir"])
        / "e2_adaptive_rules.json"
    )

    try:
        adaptive_config = NoiseAdaptiveConfig(rules_path)
        frozen_heavy_config = adaptive_config.get_heavy_config(
            CONFIG["proposed_default"]
        )
    except Exception as exc:
        logger.critical(
            f"Cannot load frozen Heavy rule: {exc}"
        )
        raise SystemExit(1)

    sigma_values = [0, 10, 25, 50, 75, 100, 125, 150]

    seed = int(CONFIG["project"]["random_seed"])

    retention_ratio = float(
        CONFIG["evaluation"]["failure_retention_ratio"]
    )

    if not (0 < retention_ratio < 1):
        logger.critical(
            "failure_retention_ratio must be in (0, 1)."
        )
        raise SystemExit(1)

    per_scene_rows = []
    mean_rows = []

    for sigma in sigma_values:
        base_scores = []
        proposed_scores = []

        for _, row in df_eval_light.iterrows():

            img, gt = dataset.load_sample(row)

            gray = (
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img.ndim == 3
                else img
            )

            if sigma == 0:
                noisy_gray = gray.copy()
            else:
                rng = deterministic_rng(
                    str(row["scene_id"]),
                    sigma,
                    seed,
                )

                gaussian = rng.normal(
                    0.0,
                    sigma,
                    gray.shape,
                ).astype(np.float32)

                noisy_gray = np.clip(
                    gray.astype(np.float32) + gaussian,
                    0,
                    255,
                ).astype(np.uint8)

            baseline_metrics = evaluate_single_image(
                noisy_gray,
                gt,
                "baseline",
                CONFIG["baseline"],
                CONFIG["evaluation"]["metric_tolerance"],
            )

            proposed_metrics = evaluate_single_image(
                noisy_gray,
                gt,
                "proposed",
                frozen_heavy_config,
                CONFIG["evaluation"]["metric_tolerance"],
            )

            base_f1 = float(
                baseline_metrics["strict_f1"]
            )

            proposed_f1 = float(
                proposed_metrics["strict_f1"]
            )

            base_scores.append(base_f1)
            proposed_scores.append(proposed_f1)

            per_scene_rows.append(
                {
                    "scene_id": row["scene_id"],
                    "injected_sigma": sigma,
                    "baseline_strict_f1": base_f1,
                    "proposed_strict_f1": proposed_f1,
                }
            )

        mean_rows.append(
            {
                "injected_sigma": sigma,
                "mean_baseline_strict_f1": float(
                    np.mean(base_scores)
                ),
                "mean_proposed_strict_f1": float(
                    np.mean(proposed_scores)
                ),
                "n_scenes": len(base_scores),
            }
        )

    tables_dir = Path(CONFIG["paths"]["tables_dir"])
    figures_dir = Path(CONFIG["paths"]["figures_dir"])

    df_per_scene = pd.DataFrame(per_scene_rows)
    df_mean = pd.DataFrame(mean_rows)

    df_per_scene.to_csv(
        tables_dir / "E4_per_scene.csv",
        index=False,
    )

    df_mean.to_csv(
        tables_dir / "E4_stress_test_means.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Relative retention criterion
    # --------------------------------------------------------

    clean_row = df_mean[
        df_mean["injected_sigma"] == 0
    ].iloc[0]

    baseline_clean = float(
        clean_row["mean_baseline_strict_f1"]
    )

    proposed_clean = float(
        clean_row["mean_proposed_strict_f1"]
    )

    baseline_threshold = (
        baseline_clean * retention_ratio
    )

    proposed_threshold = (
        proposed_clean * retention_ratio
    )

    base_fail = df_mean[
        (df_mean["injected_sigma"] > 0)
        & (
            df_mean["mean_baseline_strict_f1"]
            < baseline_threshold
        )
    ]

    proposed_fail = df_mean[
        (df_mean["injected_sigma"] > 0)
        & (
            df_mean["mean_proposed_strict_f1"]
            < proposed_threshold
        )
    ]

    joint_fail = df_mean[
        (df_mean["injected_sigma"] > 0)
        & (
            df_mean["mean_baseline_strict_f1"]
            < baseline_threshold
        )
        & (
            df_mean["mean_proposed_strict_f1"]
            < proposed_threshold
        )
    ]

    base_point = (
        int(base_fail["injected_sigma"].min())
        if not base_fail.empty
        else None
    )

    proposed_point = (
        int(proposed_fail["injected_sigma"].min())
        if not proposed_fail.empty
        else None
    )

    joint_point = (
        int(joint_fail["injected_sigma"].min())
        if not joint_fail.empty
        else None
    )

    summary = pd.DataFrame(
        {
            "method": [
                "baseline",
                "proposed",
                "joint",
            ],
            "clean_f1": [
                baseline_clean,
                proposed_clean,
                min(
                    baseline_clean,
                    proposed_clean,
                ),
            ],
            "retention_ratio": [
                retention_ratio,
                retention_ratio,
                retention_ratio,
            ],
            "failure_threshold_f1": [
                baseline_threshold,
                proposed_threshold,
                min(
                    baseline_threshold,
                    proposed_threshold,
                ),
            ],
            "first_tested_sigma_below_threshold": [
                base_point,
                proposed_point,
                joint_point,
            ],
        }
    )

    summary.to_csv(
        tables_dir / "E4_failure_points_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        df_mean["injected_sigma"],
        df_mean["mean_baseline_strict_f1"],
        marker="o",
        linestyle=":",
        label="Baseline",
    )

    plt.plot(
        df_mean["injected_sigma"],
        df_mean["mean_proposed_strict_f1"],
        marker="s",
        label="Proposed",
    )

    plt.axhline(
        baseline_threshold,
        linestyle="--",
        label=(
            f"Baseline {retention_ratio:.0%} retention"
        ),
    )

    plt.axhline(
        proposed_threshold,
        linestyle="--",
        label=(
            f"Proposed {retention_ratio:.0%} retention"
        ),
    )

    plt.xlabel("Injected Gaussian Noise Sigma")
    plt.ylabel("Mean Strict F1")
    plt.title(
        "E4: Controlled Gaussian Robustness Stress Test"
    )
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir / "E4_failure_curve.png",
        dpi=300,
    )

    plt.close()

    logger.info(
        f"Baseline clean F1 = {baseline_clean:.6f}"
    )
    logger.info(
        f"Proposed clean F1 = {proposed_clean:.6f}"
    )
    logger.info(
        f"Retention criterion = {retention_ratio:.2%}"
    )
    logger.info(
        f"Baseline first tested sigma below threshold: "
        f"{base_point}"
    )
    logger.info(
        f"Proposed first tested sigma below threshold: "
        f"{proposed_point}"
    )
    logger.info(
        f"Joint first tested sigma below threshold: "
        f"{joint_point}"
    )

    logger.info("E4 completed successfully.")


if __name__ == "__main__":
    run_e4()