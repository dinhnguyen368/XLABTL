import sys
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.dataset import Dataset
from src.pipeline import evaluate_single_image
from src.utils import ensure_directories, setup_logger
from tools.validate_dataset import validate


def run_e1() -> None:
    validate()
    ensure_directories()
    logger = setup_logger("E1", Path(CONFIG["paths"]["logs_dir"]) / "E1.log")
    logger.info("=== E1: PAIRED CHALLENGE EVIDENCE ===")

    dataset = Dataset(CONFIG["paths"]["metadata_csv"], CONFIG["paths"]["data_dir"])
    df_eval = dataset.get_split("evaluation")
    expected_count = len(df_eval)
    if expected_count != 15:
        logger.critical(f"E1 expects 15 evaluation images, found {expected_count}.")
        raise SystemExit(1)

    results = []
    failures = []
    for _, row in df_eval.iterrows():
        try:
            img, gt = dataset.load_sample(row)
            metrics = evaluate_single_image(
                img,
                gt,
                "baseline",
                CONFIG["baseline"],
                CONFIG["evaluation"]["metric_tolerance"],
                img_id=row["image_id"],
                save_dir=Path(CONFIG["paths"]["intermediate_dir"]) / "E1_baseline",
            )
            result = {
                "image_id": row["image_id"],
                "scene_id": row["scene_id"],
                "noise_level": row["noise_level"],
                **{f"baseline_{key}": value for key, value in metrics.items()},
            }
            results.append(result)
        except Exception as exc:
            logger.error(f"Failed {row['image_id']}: {exc}")
            failures.append(str(row["image_id"]))

    if len(results) != expected_count:
        logger.critical(
            f"E1 FAIL HARD: expected {expected_count}, processed {len(results)}. "
            f"Failed images: {failures}"
        )
        raise SystemExit(1)

    df = pd.DataFrame(results)
    tables_dir = Path(CONFIG["paths"]["tables_dir"])
    df.to_csv(tables_dir / "E1_raw.csv", index=False)

    pivot = df.pivot(index="scene_id", columns="noise_level", values="baseline_strict_f1")
    required_levels = {"light", "medium", "heavy"}
    if set(pivot.columns) != required_levels:
        logger.critical(f"E1 paired analysis missing noise levels: {set(pivot.columns)}")
        raise SystemExit(1)

    pivot["drop_L_to_M"] = pivot["light"] - pivot["medium"]
    pivot["drop_L_to_H"] = pivot["light"] - pivot["heavy"]
    pivot.to_csv(tables_dir / "E1_paired_analysis.csv")

    # Select the three lowest-heavy-F1 scenes for report illustration.
    worst = pivot.sort_values("heavy").head(3).reset_index()
    worst[["scene_id", "heavy"]].to_csv(tables_dir / "E1_three_failure_cases.csv", index=False)

    summary = df.groupby("noise_level", observed=True).mean(numeric_only=True).reset_index()
    summary.to_csv(tables_dir / "E1_summary.csv", index=False)

    logger.info("Positive delta = F1( Light ) - F1( heavier level ) > 0, indicating degradation.")
    logger.info("Negative delta indicates higher F1 at the heavier level and must be reported as observed.")
    logger.info("E1 completed successfully.")


if __name__ == "__main__":
    run_e1()
