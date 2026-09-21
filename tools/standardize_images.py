from pathlib import Path
import cv2


TARGET_WIDTH = 137
TARGET_HEIGHT = 253


def fit_and_pad(image, target_width, target_height):
    """
    Giữ nguyên tỷ lệ ảnh, resize để vừa trong khung mục tiêu,
    sau đó đệm phần thiếu bằng phản chiếu biên.
    """
    h, w = image.shape[:2]

    # Tính tỷ lệ resize để ảnh nằm gọn trong khung đích
    scale = min(target_width / w, target_height / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    )

    # Tính lượng padding ở 4 phía
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left

    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top

    result = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101
    )

    return result


def process_folder(folder):
    folder = Path(folder)

    if not folder.exists():
        print(f"[SKIP] Không tồn tại: {folder}")
        return

    images = sorted(folder.glob("*.jpg"))

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[ERROR] Không đọc được: {image_path}")
            continue

        old_h, old_w = image.shape[:2]

        standardized = fit_and_pad(
            image,
            TARGET_WIDTH,
            TARGET_HEIGHT
        )

        ok = cv2.imwrite(
            str(image_path),
            standardized,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        if not ok:
            print(f"[ERROR] Không thể ghi: {image_path}")
            continue

        new_h, new_w = standardized.shape[:2]

        print(
            f"[OK] {image_path.name}: "
            f"{old_w}x{old_h} -> {new_w}x{new_h}"
        )


def main():
    folders = [
        "data/development",
        "data/evaluation/light",
        "data/evaluation/medium",
        "data/evaluation/heavy",
    ]

    print("=== STANDARDIZE 30 IMAGES ===")
    print(f"Target size: {TARGET_WIDTH} x {TARGET_HEIGHT}")
    print()

    for folder in folders:
        print(f"\n--- {folder} ---")
        process_folder(folder)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()