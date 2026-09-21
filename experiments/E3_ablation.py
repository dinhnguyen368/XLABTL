from pathlib import Path

import pandas as pd

from src.adaptive_parameters import NoiseAdaptiveConfig
from src.config import CONFIG
from src.dataset import Dataset
from src.pipeline import evaluate_single_image
from src.utils import ensure_directories, setup_logger
from src.visualization import plot_metric_vs_noise
from tools.validate_dataset import validate


def run_e3() -> None:
    validate()
    ensure_directories()

    logger = setup_logger(
        "E3",
        Path(CONFIG["paths"]["logs_dir"]) / "E3.log",
    )

    logger.info(
        "=== E3: ABLATION STUDY ON EVALUATION SET ==="
    )

    rules_path = (
        Path(CONFIG["paths"]["configs_dir"])
        / "e2_adaptive_rules.json"
    )

    try:
        adaptive_config = NoiseAdaptiveConfig(
            rules_path
        )
    except SystemExit:
        raise
    except Exception as exc:
        logger.critical(
            f"Failed to load frozen rules: {exc}"
        )
        raise SystemExit(1)

    dataset = Dataset(
        CONFIG["paths"]["metadata_csv"],
        CONFIG["paths"]["data_dir"],
    )

    df_eval = dataset.get_split("evaluation")

    if len(df_eval) != 15:
        logger.critical(
            f"E3 expects 15 evaluation images, "
            f"found {len(df_eval)}."
        )
        raise SystemExit(1)

    results = []
    failures = []

    for _, row in df_eval.iterrows():

        try:
            img, gt = dataset.load_sample(row)

            baseline_metrics = evaluate_single_image(
                img,
                gt,
                "baseline",
                CONFIG["baseline"],
                CONFIG["evaluation"][
                    "metric_tolerance"
                ],
            )

            proposed_config = adaptive_config.get_config(
                float(row["noise_measure"]),
                CONFIG["proposed_default"],
            )

            proposed_metrics = evaluate_single_image(
                img,
                gt,
                "proposed",
                proposed_config,
                CONFIG["evaluation"][
                    "metric_tolerance"
                ],
                img_id=row["image_id"],
                save_dir=(
                    Path(
                        CONFIG["paths"][
                            "intermediate_dir"
                        ]
                    )
                    / "E3_comparison"
                ),
            )

            result = {
                "image_id": row["image_id"],
                "scene_id": row["scene_id"],
                "noise_level": row["noise_level"],
                "noise_measure": float(
                    row["noise_measure"]
                ),
                "baseline_strict_f1": float(
                    baseline_metrics["strict_f1"]
                ),
                "proposed_strict_f1": float(
                    proposed_metrics["strict_f1"]
                ),
                "delta_proposed_minus_baseline_strict_f1":
                    float(
                        proposed_metrics["strict_f1"]
                        - baseline_metrics["strict_f1"]
                    ),
            }

            results.append(result)

        except Exception as exc:
            logger.error(
                f"Failed {row['image_id']}: {exc}"
            )
            failures.append(
                str(row["image_id"])
            )

    if len(results) != len(df_eval):
        logger.critical(
            f"E3 FAIL HARD: processed "
            f"{len(results)}/{len(df_eval)} images. "
            f"Failures={failures}"
        )
        raise SystemExit(1)

    df = pd.DataFrame(results)

    tables_dir = Path(
        CONFIG["paths"]["tables_dir"]
    )

    df.to_csv(
        tables_dir / "E3_raw.csv",
        index=False,
    )

    summary = (
        df.groupby(
            "noise_level",
            observed=True,
        )
        .agg(
            baseline_strict_f1=(
                "baseline_strict_f1",
                "mean",
            ),
            proposed_strict_f1=(
                "proposed_strict_f1",
                "mean",
            ),
            mean_delta=(
                "delta_proposed_minus_baseline_strict_f1",
                "mean",
            ),
            n=("image_id", "count"),
        )
        .reset_index()
    )

    summary.to_csv(
        tables_dir / "E3_summary.csv",
        index=False,
    )

    plot_metric_vs_noise(
        tables_dir / "E3_summary.csv",
        Path(
            CONFIG["paths"]["figures_dir"]
        ) / "E3_ablation_strict_f1.png",
        metric="strict_f1",
        title=(
            "E3: Baseline vs "
            "Noise-Adaptive Bilateral + Canny"
        ),
    )

    logger.info(
        "E3 completed successfully."
    )


if __name__ == "__main__":
    run_e3()