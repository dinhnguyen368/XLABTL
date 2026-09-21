from pathlib import Path
import cv2
import json


OUTPUT_FILE = Path("outputs/configs/roi_coordinates.json")


SCENES = {
    "scene01": "data/development/dev_01_l.jpg",
    "scene02": "data/development/dev_02_l.jpg",
    "scene03": "data/development/dev_03_l.jpg",
    "scene04": "data/development/dev_04_l.jpg",
    "scene05": "data/development/dev_05_l.jpg",
    "scene06": "data/evaluation/light/eval_06_l.jpg",
    "scene07": "data/evaluation/light/eval_07_l.jpg",
    "scene08": "data/evaluation/light/eval_08_l.jpg",
    "scene09": "data/evaluation/light/eval_09_l.jpg",
    "scene10": "data/evaluation/light/eval_10_l.jpg",
}


def select_roi_for_scene(scene_id, image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

    display = image.copy()

    # Hiển thị kích thước ảnh
    h, w = image.shape[:2]

    instructions = (
        f"{scene_id} | SIZE={w}x{h}\n"
        "Kéo chuột chọn ROI trong vùng nền đồng nhất.\n"
        "ENTER: xác nhận | ESC: bỏ qua"
    )

    # OpenCV đặt cửa sổ
    window_name = f"Select ROI - {scene_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Phóng cửa sổ cho dễ thao tác
    cv2.resizeWindow(window_name, 700, 700)

    # Ghi hướng dẫn lên ảnh
    cv2.putText(
        display,
        scene_id,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    print()
    print("=" * 60)
    print(f"SCENE: {scene_id}")
    print(f"IMAGE: {image_path}")
    print(f"SIZE : {w} x {h}")
    print()
    print("Hãy kéo chuột chọn một vùng ROI tương đối đồng nhất.")
    print("Không chọn vật thể, chữ, texture hoặc cạnh.")
    print("Nhấn ENTER để xác nhận.")
    print("Nhấn ESC để bỏ qua.")
    print("=" * 60)

    roi = cv2.selectROI(
        window_name,
        display,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(window_name)

    x, y, roi_w, roi_h = map(int, roi)

    if roi_w <= 0 or roi_h <= 0:
        return None

    return {
        "x": x,
        "y": y,
        "w": roi_w,
        "h": roi_h,
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    print("\n=== ROI SELECTION ===")
    print("Bạn sẽ chọn 1 ROI cho mỗi scene.")
    print("ROI của Light sẽ được dùng chung cho Medium và Heavy.")
    print()

    for scene_id, image_path in SCENES.items():
        roi = select_roi_for_scene(scene_id, image_path)

        if roi is None:
            print(f"[ERROR] Chưa chọn ROI cho {scene_id}.")
            print("Bạn phải chọn ROI cho tất cả 10 scene.")
            return

        results[scene_id] = roi

        print(f"[OK] {scene_id}: {roi}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print()
    print("=" * 60)
    print("ĐÃ LƯU ROI:")
    print(OUTPUT_FILE)
    print("=" * 60)

    for scene_id, roi in results.items():
        print(
            f"{scene_id}: "
            f"x={roi['x']}, "
            f"y={roi['y']}, "
            f"w={roi['w']}, "
            f"h={roi['h']}"
        )


if __name__ == "__main__":
    main()