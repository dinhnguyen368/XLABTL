from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# E3: Compare Baseline vs Proposed Strict F1.
#
# Input:
# outputs/tables/E3_summary.csv
#
# Expected columns:
# noise_level
# baseline_strict_f1
# proposed_strict_f1
#
# Output:
# outputs/figures/E3_baseline_vs_proposed.png
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "outputs" / "tables" / "E3_summary.csv"
OUTPUT_PNG = PROJECT_ROOT / "outputs" / "figures" / "E3_baseline_vs_proposed.png"

LEVEL_ORDER = ["light", "medium", "heavy"]


def main():
    print("=" * 70)
    print("CREATING E3 BASELINE VS PROPOSED PLOT")
    print("=" * 70)
    print(f"Input : {INPUT_CSV}")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find E3 CSV:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {
        "noise_level",
        "baseline_strict_f1",
        "proposed_strict_f1",
    }
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
    df["proposed_strict_f1"] = pd.to_numeric(
        df["proposed_strict_f1"], errors="raise"
    )

    df = df[df["noise_level"].isin(LEVEL_ORDER)].copy()

    if df.empty:
        raise ValueError(
            "No rows found for light/medium/heavy in E3_summary.csv."
        )

    df["order"] = df["noise_level"].map(
        {name: i for i, name in enumerate(LEVEL_ORDER)}
    )
    df = df.sort_values("order")

    x = range(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))

    baseline_bars = ax.bar(
        [i - width / 2 for i in x],
        df["baseline_strict_f1"],
        width=width,
        label="Baseline",
    )

    proposed_bars = ax.bar(
        [i + width / 2 for i in x],
        df["proposed_strict_f1"],
        width=width,
        label="Proposed",
    )

    ax.set_title(
        "E3 – Baseline vs Proposed Strict F1",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Strict F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels([name.capitalize() for name in df["noise_level"]])

    all_values = pd.concat(
        [df["baseline_strict_f1"], df["proposed_strict_f1"]]
    )
    ax.set_ylim(0, max(0.25, float(all_values.max()) * 1.25))

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    def add_labels(bars, values):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_labels(baseline_bars, df["baseline_strict_f1"])
    add_labels(proposed_bars, df["proposed_strict_f1"])

    baseline_mean = df["baseline_strict_f1"].mean()
    proposed_mean = df["proposed_strict_f1"].mean()
    delta = proposed_mean - baseline_mean

    summary_text = (
        f"Mean Strict F1: {baseline_mean:.5f} → {proposed_mean:.5f}\n"
        f"Absolute improvement: {delta:+.5f}"
    )

    fig.text(
        0.5,
        0.01,
        summary_text,
        ha="center",
        va="bottom",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0.07, 1, 1))

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("=" * 70)
    print("DONE")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Mean baseline : {baseline_mean:.5f}")
    print(f"Mean proposed : {proposed_mean:.5f}")
    print(f"Mean delta    : {delta:+.5f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
