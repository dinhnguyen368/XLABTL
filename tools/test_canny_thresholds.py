from pathlib import Path
import pandas as pd

from src.config import CONFIG
from src.dataset import Dataset
from src.pipeline import evaluate_single_image
from tools.validate_dataset import validate


CANDIDATES = [
    (20, 60),
    (30, 90),
    (40, 120),
    (50, 150),
    (60, 180),
    (75, 225),
]


def main():
    validate()

    dataset = Dataset(
        CONFIG["paths"]["metadata_csv"],
        CONFIG["paths"]["data_dir"],
    )

    df_dev = dataset.get_split("development")

    results = []

    print("\n=== CANNY THRESHOLD STUDY — DEVELOPMENT ONLY ===\n")

    for t1, t2 in CANDIDATES:
        scores = []
        edge_counts = []

        config = CONFIG["baseline"].copy()
        config["canny_threshold1"] = t1
        config["canny_threshold2"] = t2

        for _, row in df_dev.iterrows():
            img, gt = dataset.load_sample(row)

            metrics = evaluate_single_image(
                img,
                gt,
                "baseline",
                config,
                CONFIG["evaluation"]["metric_tolerance"],
            )

            scores.append(float(metrics["strict_f1"]))

            # Chỉ để tham khảo: số pixel edge
            import cv2
            gray = (
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img.ndim == 3
                else img
            )

            edges = cv2.Canny(
                gray,
                t1,
                t2,
            )

            edge_counts.append(
                int((edges > 0).sum())
            )

        mean_f1 = sum(scores) / len(scores)
        mean_edges = sum(edge_counts) / len(edge_counts)

        results.append(
            {
                "threshold1": t1,
                "threshold2": t2,
                "mean_strict_f1_dev": mean_f1,
                "mean_edge_pixels_dev": mean_edges,
            }
        )

        print(
            f"Canny {t1:>3}/{t2:<3} | "
            f"Dev Strict F1 = {mean_f1:.6f} | "
            f"Mean edge pixels = {mean_edges:.1f}"
        )

    df = pd.DataFrame(results)

    output = Path(
        CONFIG["paths"]["tables_dir"]
    ) / "canny_threshold_study_dev.csv"

    df.to_csv(output, index=False)

    best = df.loc[
        df["mean_strict_f1_dev"].idxmax()
    ]

    print("\n=== RESULT ===")
    print(
        f"Best by Development Strict F1: "
        f"{int(best['threshold1'])}/"
        f"{int(best['threshold2'])}"
    )
    print(
        f"Development Strict F1 = "
        f"{best['mean_strict_f1_dev']:.6f}"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()