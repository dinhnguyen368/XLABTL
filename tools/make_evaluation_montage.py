from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

SCENES = [6, 7, 8, 9, 10]

# Có thể đổi thành "light", "medium", "heavy"
NOISE_LEVEL = "heavy"

OUTPUT = Path(
    "outputs/figures/evaluation_5scenes_4panel.png"
)

# Phóng ảnh nhỏ 137x253 lên để nhìn rõ
SCALE = 3

# Kích thước khoảng cách
CELL_PADDING = 12
TITLE_HEIGHT = 45
ROW_LABEL_WIDTH = 90

# ============================================================
# FONT
# ============================================================

def load_font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


FONT_TITLE = load_font(28)
FONT_SCENE = load_font(24)
FONT_SMALL = load_font(18)


# ============================================================
# PATH HELPERS
# ============================================================

def get_paths(scene):
    suffix = {
    "light": "l",
    "medium": "m",
    "heavy": "h",
    }[NOISE_LEVEL]

    scene_name = f"eval_{scene:02d}_{suffix}"

    original = (
        Path("outputs/intermediate/E3_comparison")
        / scene_name
        / "original.jpg"
    )

    ground_truth = (
        Path("outputs/intermediate/E3_comparison")
        / scene_name
        / "ground_truth.png"
    )

    proposed = (
        Path("outputs/intermediate/E3_comparison")
        / scene_name
        / "edges.png"
    )

    baseline = (
        Path("outputs/intermediate/E1_baseline")
        / scene_name
        / "edges.png"
    )

    return original, ground_truth, baseline, proposed


# ============================================================
# IMAGE HELPERS
# ============================================================

def open_image(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file:\n{path}"
        )

    return Image.open(path).convert("RGB")


def prepare_image(img):
    """
    Phóng to ảnh bằng nearest-neighbor để giữ nguyên
    hình dạng pixel của edge map/GT.
    """
    return img.resize(
        (
            img.width * SCALE,
            img.height * SCALE,
        ),
        Image.Resampling.NEAREST,
    )


def add_border(img):
    return ImageOps.expand(
        img,
        border=2,
        fill="black",
    )


def add_caption(img, text, width):
    result = Image.new(
        "RGB",
        (
            width,
            img.height + 35,
        ),
        "white",
    )

    result.paste(
        img,
        (
            (width - img.width) // 2,
            0,
        ),
    )

    draw = ImageDraw.Draw(result)

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=FONT_SMALL,
    )

    text_width = bbox[2] - bbox[0]

    draw.text(
        (
            (width - text_width) // 2,
            img.height + 7,
        ),
        text,
        fill="black",
        font=FONT_SMALL,
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "Original",
        "Ground Truth",
        "Baseline Canny",
        "Proposed",
    ]

    print("=" * 70)
    print("CREATING 5-SCENE EVALUATION MONTAGE")
    print(f"Noise level: {NOISE_LEVEL}")
    print("=" * 70)

    all_rows = []

    for scene in SCENES:

        paths = get_paths(scene)

        print()
        print(f"Scene {scene}")

        images = [
            open_image(path)
            for path in paths
        ]

        images = [
            prepare_image(img)
            for img in images
        ]

        images = [
            add_border(img)
            for img in images
        ]

        all_rows.append(images)

        for path in paths:
            print(f"  {path}")

    # Tất cả ảnh phải cùng kích thước
    cell_width = max(
        img.width
        for row in all_rows
        for img in row
    )

    cell_height = max(
        img.height
        for row in all_rows
        for img in row
    )

    # Thêm caption vào từng cell
    cell_images = []

    for row in all_rows:

        new_row = []

        for img, caption in zip(
            row,
            columns,
        ):

            # Căn giữa trong cell
            canvas = Image.new(
                "RGB",
                (
                    cell_width,
                    cell_height,
                ),
                "white",
            )

            x = (
                cell_width
                - img.width
            ) // 2

            y = (
                cell_height
                - img.height
            ) // 2

            canvas.paste(
                img,
                (x, y),
            )

            new_row.append(
                add_caption(
                    canvas,
                    caption,
                    cell_width,
                )
            )

        cell_images.append(
            new_row
        )

    final_cell_height = (
        cell_height + 35
    )

    # ========================================================
    # FINAL CANVAS
    # ========================================================

    total_width = (
        ROW_LABEL_WIDTH
        + 4 * (
            cell_width
            + CELL_PADDING
        )
        + CELL_PADDING
    )

    total_height = (
        TITLE_HEIGHT
        + 5 * (
            final_cell_height
            + CELL_PADDING
        )
        + CELL_PADDING
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            total_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    # ========================================================
    # MAIN TITLE
    # ========================================================

    title = (
        "Evaluation: Original vs Ground Truth "
        "vs Baseline vs Proposed"
    )

    bbox = draw.textbbox(
        (0, 0),
        title,
        font=FONT_TITLE,
    )

    title_width = (
        bbox[2] - bbox[0]
    )

    draw.text(
        (
            (total_width - title_width) // 2,
            8,
        ),
        title,
        fill="black",
        font=FONT_TITLE,
    )

    # ========================================================
    # TABLE
    # ========================================================

    start_y = TITLE_HEIGHT

    for row_idx, row in enumerate(
        cell_images
    ):

        scene = SCENES[row_idx]

        y = (
            start_y
            + CELL_PADDING
            + row_idx * (
                final_cell_height
                + CELL_PADDING
            )
        )

        # Scene label
        label = f"Scene {scene}"

        bbox = draw.textbbox(
            (0, 0),
            label,
            font=FONT_SCENE,
        )

        label_height = (
            bbox[3] - bbox[1]
        )

        draw.text(
            (
                12,
                y + (
                    final_cell_height
                    - label_height
                ) // 2,
            ),
            label,
            fill="black",
            font=FONT_SCENE,
        )

        for col_idx, img in enumerate(row):

            x = (
                ROW_LABEL_WIDTH
                + CELL_PADDING
                + col_idx * (
                    cell_width
                    + CELL_PADDING
                )
            )

            canvas.paste(
                img,
                (x, y),
            )

            # Border cell
            draw.rectangle(
                [
                    x,
                    y,
                    x + img.width - 1,
                    y + img.height - 1,
                ],
                outline="black",
                width=1,
            )

    # ========================================================
    # SAVE
    # ========================================================

    canvas.save(
        OUTPUT,
        format="PNG",
        dpi=(300, 300),
    )

    print()
    print("=" * 70)
    print("DONE")
    print(f"Saved: {OUTPUT}")
    print(f"Size : {canvas.width} x {canvas.height}")
    print("=" * 70)


if __name__ == "__main__":
    main()