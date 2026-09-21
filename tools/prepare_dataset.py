from pathlib import Path

METADATA_HEADER = (
    "image_id,split,noise_level,scene_id,gt_id,iso,roi_x,roi_y,roi_w,roi_h,"
    "noise_measure,mean_luminance,notes\n"
)


def setup() -> None:
    dirs = [
        "data/development",
        "data/evaluation/light",
        "data/evaluation/medium",
        "data/evaluation/heavy",
        "data/ground_truth",
        "outputs/tables",
        "outputs/figures",
        "outputs/intermediate",
        "outputs/configs",
        "outputs/logs",
    ]
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

    metadata_path = Path("data/metadata.csv")
    if not metadata_path.exists():
        metadata_path.write_text(METADATA_HEADER, encoding="utf-8")
        print(f"Created {metadata_path}")

    print("Project folders are ready.")


if __name__ == "__main__":
    setup()
