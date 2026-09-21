import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CONFIG
from src.dataset import Dataset
from src.pipeline import evaluate_single_image
from src.utils import ensure_directories, setup_logger
from tools.validate_dataset import validate


def run_e2() -> None:
    validate()
    ensure_directories()

    logger = setup_logger(
        "E2",
        Path(CONFIG["paths"]["logs_dir"]) / "E2.log",
    )

    logger.info(
        "=== E2: NOISE-PROXY ADAPTIVE PARAMETER SELECTION (DEV ONLY) ==="
    )

    rules_path = (
        Path(CONFIG["paths"]["configs_dir"])
        / "e2_adaptive_rules.json"
    )

    if rules_path.exists():
        logger.critical(
            f"Frozen rule already exists: {rules_path}. "
            f"Delete it manually only if you explicitly intend to rerun E2."
        )
        raise SystemExit(1)

    dataset = Dataset(
        CONFIG["paths"]["metadata_csv"],
        CONFIG["paths"]["data_dir"],
    )

    df_dev = dataset.get_split("development")

    if len(df_dev) != 15:
        logger.critical(
            f"E2 expects 15 development images, found {len(df_dev)}."
        )
        raise SystemExit(1)

    grid = CONFIG["parameter_search"]

    combinations = list(
        product(
            grid["bilateral_d"],
            grid["bilateral_sigma_color"],
            grid["bilateral_sigma_space"],
        )
    )

    # Guard against an accidentally trivial parameter study.
    if len(set(grid["bilateral_d"])) < 3:
        logger.critical(
            "E2 requires at least 3 distinct bilateral_d values."
        )
        raise SystemExit(1)

    if len(set(grid["bilateral_sigma_color"])) < 3:
        logger.critical(
            "E2 requires at least 3 distinct sigma_color values."
        )
        raise SystemExit(1)

    if len(set(grid["bilateral_sigma_space"])) < 3:
        logger.critical(
            "E2 requires at least 3 distinct sigma_space values."
        )
        raise SystemExit(1)

    thresholds = grid["noise_thresholds"]

    t1 = float(thresholds["t1"])
    t2 = float(thresholds["t2"])

    if not (0 < t1 < t2):
        logger.critical(
            f"Invalid noise thresholds: t1={t1}, t2={t2}"
        )
        raise SystemExit(1)

    bins = {
        "low_noise": df_dev[
            df_dev["noise_measure"] <= t1
        ],
        "medium_noise": df_dev[
            (df_dev["noise_measure"] > t1)
            & (df_dev["noise_measure"] <= t2)
        ],
        "high_noise": df_dev[
            df_dev["noise_measure"] > t2
        ],
    }

    best_rules = {
        "thresholds": {
            "t1": t1,
            "t2": t2,
        }
    }

    tables_dir = Path(CONFIG["paths"]["tables_dir"])
    figures_dir = Path(CONFIG["paths"]["figures_dir"])
    intermediate_dir = Path(
        CONFIG["paths"]["intermediate_dir"]
    )

    for bin_name, df_bin in bins.items():

        if df_bin.empty:
            logger.critical(
                f"Noise bin {bin_name} is empty. E2 FAIL HARD."
            )
            raise SystemExit(1)

        logger.info(
            f"Searching {bin_name}: N={len(df_bin)}, "
            f"range="
            f"{df_bin['noise_measure'].min():.4f}"
            f"-"
            f"{df_bin['noise_measure'].max():.4f}"
        )

        representative_row = df_bin.iloc[0]

        representative_img, representative_gt = (
            dataset.load_sample(representative_row)
        )

        all_results = []

        best_f1 = -1.0
        best_params = None

        for d, sigma_color, sigma_space in combinations:

            config = CONFIG["proposed_default"].copy()

            config.update(
                {
                    "bilateral_d": int(d),
                    "bilateral_sigma_color": float(
                        sigma_color
                    ),
                    "bilateral_sigma_space": float(
                        sigma_space
                    ),
                }
            )

            scores = []
            failures = []

            for _, row in df_bin.iterrows():

                try:
                    img, gt = dataset.load_sample(row)

                    metrics = evaluate_single_image(
                        img,
                        gt,
                        "proposed",
                        config,
                        CONFIG["evaluation"][
                            "metric_tolerance"
                        ],
                    )

                    scores.append(
                        float(metrics["strict_f1"])
                    )

                except Exception as exc:
                    logger.error(
                        f"Error on {row['image_id']}: {exc}"
                    )
                    failures.append(
                        str(row["image_id"])
                    )

            if len(scores) != len(df_bin):
                logger.critical(
                    f"E2 FAIL HARD for {bin_name}: "
                    f"processed {len(scores)}/{len(df_bin)} "
                    f"images. Failures={failures}"
                )
                raise SystemExit(1)

            mean_f1 = float(sum(scores) / len(scores))

            all_results.append(
                {
                    "d": int(d),
                    "sigma_color": float(sigma_color),
                    "sigma_space": float(sigma_space),
                    "mean_strict_f1": mean_f1,
                }
            )

            combo_label = (
                f"d{int(d)}_"
                f"sc{int(sigma_color)}_"
                f"ss{int(sigma_space)}"
            )

            save_dir = (
                intermediate_dir
                / "E2"
                / bin_name
                / combo_label
            )

            evaluate_single_image(
                representative_img,
                representative_gt,
                "proposed",
                config,
                CONFIG["evaluation"][
                    "metric_tolerance"
                ],
                img_id=str(
                    representative_row["image_id"]
                ),
                save_dir=save_dir,
            )

            if mean_f1 > best_f1:
                best_f1 = mean_f1

                best_params = {
                    "d": int(d),
                    "sigma_color": float(
                        sigma_color
                    ),
                    "sigma_space": float(
                        sigma_space
                    ),
                    "dev_strict_f1": mean_f1,
                }

        if best_params is None:
            logger.critical(
                f"No best rule generated for {bin_name}."
            )
            raise SystemExit(1)

        best_rules[bin_name] = best_params

        df_results = pd.DataFrame(all_results)

        df_results.to_csv(
            tables_dir / f"E2_{bin_name}_params.csv",
            index=False,
        )

        plot_df = df_results.copy()

        plot_df["label"] = plot_df.apply(
            lambda row:
            f"d{int(row['d'])}-"
            f"c{int(row['sigma_color'])}-"
            f"s{int(row['sigma_space'])}",
            axis=1,
        )

        plot_df = (
            plot_df
            .sort_values("mean_strict_f1")
            .tail(15)
        )

        plt.figure(figsize=(11, 7))

        plt.barh(
            plot_df["label"],
            plot_df["mean_strict_f1"],
        )

        plt.xlim(0, 1.05)

        plt.xlabel(
            "Mean Strict F1 on Development Set"
        )

        plt.ylabel(
            "Bilateral configuration"
        )

        plt.title(
            f"E2 Parameter Study — Top 15 ({bin_name})"
        )

        plt.grid(
            axis="x",
            linestyle="--",
            alpha=0.6,
        )

        plt.tight_layout()

        plt.savefig(
            figures_dir
            / f"E2_param_study_{bin_name}.png",
            dpi=300,
        )

        plt.close()

    expected_keys = {
        "thresholds",
        "low_noise",
        "medium_noise",
        "high_noise",
    }

    if set(best_rules.keys()) != expected_keys:
        logger.critical(
            "E2 generated incomplete rule schema."
        )
        raise SystemExit(1)

    rules_path.write_text(
        json.dumps(best_rules, indent=4),
        encoding="utf-8",
    )

    logger.info(
        f"E2 completed. Frozen noise-adaptive rules saved at "
        f"{rules_path}"
    )


if __name__ == "__main__":
    run_e2()