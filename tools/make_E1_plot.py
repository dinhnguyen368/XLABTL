from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# E1: Plot baseline Strict F1 across Light / Medium / Heavy.
#
# Input:
# outputs/tables/E1_summary.csv
#
# Expected columns:
# noise_level
# baseline_strict_f1
#
# Output:
# outputs/figures/E1_baseline_noise_levels.png
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "outputs" / "tables" / "E1_summary.csv"
OUTPUT_PNG = PROJECT_ROOT / "outputs" / "figures" / "E1_baseline_noise_levels.png"

LEVEL_ORDER = ["light", "medium", "heavy"]


def main():
    print("=" * 70)
    print("CREATING E1 BASELINE NOISE-LEVEL PLOT")
    print("=" * 70)
    print(f"Input : {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find E1 CSV:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {"noise_level", "baseline_strict_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["noise_level"] = df["noise_level"].astype(str).str.strip().str.lower()
    df["baseline_strict_f1"] = pd.to_numeric(
        df["baseline_strict_f1"], errors="raise"
    )

    # Keep only the three report levels and order them consistently.
    df = df[df["noise_level"].isin(LEVEL_ORDER)].copy()

    if df.empty:
        raise ValueError(
            "No rows found for light/medium/heavy in E1_summary.csv."
        )

    df["order"] = df["noise_level"].map(
        {name: i for i, name in enumerate(LEVEL_ORDER)}
    )
    df = df.sort_values("order")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars = ax.bar(
        [x.capitalize() for x in df["noise_level"]],
        df["baseline_strict_f1"],
    )

    ax.set_title(
        "E1 – Baseline Performance Under Different Noise Levels",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Strict F1")
    ax.set_ylim(0, max(0.25, float(df["baseline_strict_f1"].max()) * 1.25))
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, df["baseline_strict_f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("=" * 70)
    print("DONE")
    print(f"Saved: {OUTPUT_PNG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
