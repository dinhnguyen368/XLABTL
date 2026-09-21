from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Create a 3-panel montage showing one evaluation scene
# at Light / Medium / Heavy noise levels.
#
# Expected project structure:
# data/
#   evaluation/
#     light/eval_06_l.jpg
#     medium/eval_06_m.jpg
#     heavy/eval_06_h.jpg
#
# Output:
# outputs/figures/dataset_light_medium_heavy.png
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENE_ID = 6

INPUTS = {
    "Light": PROJECT_ROOT / "data" / "evaluation" / "light" / f"eval_{SCENE_ID:02d}_l.jpg",
    "Medium": PROJECT_ROOT / "data" / "evaluation" / "medium" / f"eval_{SCENE_ID:02d}_m.jpg",
    "Heavy": PROJECT_ROOT / "data" / "evaluation" / "heavy" / f"eval_{SCENE_ID:02d}_h.jpg",
}

OUTPUT = PROJECT_ROOT / "outputs" / "figures" / "dataset_light_medium_heavy.png"


def main():
    print("=" * 70)
    print("CREATING LIGHT / MEDIUM / HEAVY DATASET MONTAGE")
    print(f"Scene: {SCENE_ID}")
    print("=" * 70)

    images = []
    for label, path in INPUTS.items():
        print(f"{label:>8}: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find image:\n{path}\n"
                "Check that the project structure and filenames are correct."
            )

        images.append((label, Image.open(path).convert("RGB")))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.5))

    for ax, (label, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"Example Evaluation Scene {SCENE_ID}: Light, Medium, and Heavy Noise",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("=" * 70)
    print("DONE")
    print(f"Saved: {OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
