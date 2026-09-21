from pathlib import Path
import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("data/ground_truth")

SCENES = [
    ("scene01", "data/development/dev_01_l.jpg"),
    ("scene02", "data/development/dev_02_l.jpg"),
    ("scene03", "data/development/dev_03_l.jpg"),
    ("scene04", "data/development/dev_04_l.jpg"),
    ("scene05", "data/development/dev_05_l.jpg"),
    ("scene06", "data/evaluation/light/eval_06_l.jpg"),
    ("scene07", "data/evaluation/light/eval_07_l.jpg"),
    ("scene08", "data/evaluation/light/eval_08_l.jpg"),
    ("scene09", "data/evaluation/light/eval_09_l.jpg"),
    ("scene10", "data/evaluation/light/eval_10_l.jpg"),
]

# Phóng to ảnh để dễ vẽ
DISPLAY_SCALE = 4

# Bán kính nét vẽ trên ảnh gốc
BRUSH_RADIUS = 1


# ============================================================
# GLOBAL STATE
# ============================================================

current_mask = None
current_image = None
current_scene = None


def make_display():
    """
    Tạo ảnh hiển thị:
    - ảnh gốc được phóng to
    - cạnh GT được tô màu đỏ để dễ nhìn
    """
    global current_mask, current_image, current_scene

    base = current_image.copy()

    # Vẽ các pixel GT lên ảnh gốc
    edge_pixels = current_mask > 0
    base[edge_pixels] = (0, 0, 255)

    # Phóng to
    display = cv2.resize(
        base,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    # Thông tin hướng dẫn
    cv2.putText(
        display,
        f"{current_scene} | LEFT=draw | RIGHT=erase | "
        f"S=save | N=next | B=back | C=clear | ESC=exit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return display


def draw_on_mask(x, y, value):
    """
    x,y là tọa độ trên cửa sổ phóng to.
    Chuyển về tọa độ ảnh gốc rồi vẽ vào mask.
    """
    global current_mask

    real_x = int(x / DISPLAY_SCALE)
    real_y = int(y / DISPLAY_SCALE)

    h, w = current_mask.shape

    if not (0 <= real_x < w and 0 <= real_y < h):
        return

    cv2.circle(
        current_mask,
        (real_x, real_y),
        BRUSH_RADIUS,
        value,
        -1,
    )


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        draw_on_mask(x, y, 255)

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            draw_on_mask(x, y, 255)

        elif flags & cv2.EVENT_FLAG_RBUTTON:
            draw_on_mask(x, y, 0)


def load_scene(index):
    global current_image, current_mask, current_scene

    scene_id, image_path = SCENES[index]

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh: {image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Không thể đọc ảnh: {image_path}"
        )

    h, w = image.shape[:2]

    # GT luôn đúng kích thước ảnh
    mask_path = OUTPUT_DIR / f"{scene_id}_gt.png"

    if mask_path.exists():
        existing = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if existing is not None and existing.shape == (h, w):
            mask = existing.copy()
            print(f"[LOAD] Đã có GT cũ: {mask_path}")
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            print(f"[NEW] Tạo GT mới: {scene_id}")

    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        print(f"[NEW] Tạo GT mới: {scene_id}")

    current_image = image
    current_mask = mask
    current_scene = scene_id

    print()
    print("=" * 70)
    print(f"SCENE: {scene_id}")
    print(f"IMAGE: {image_path}")
    print(f"SIZE : {w} x {h}")
    print("=" * 70)
    print("Chuột trái  : vẽ cạnh")
    print("Chuột phải  : xóa cạnh")
    print("S            : lưu")
    print("N            : lưu + scene tiếp theo")
    print("B            : scene trước")
    print("C            : xóa toàn bộ GT hiện tại")
    print("ESC          : thoát")
    print()


def save_current():
    global current_mask, current_scene

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR /
        f"{current_scene}_gt.png"
    )

    ok = cv2.imwrite(
        str(output_path),
        current_mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )

    if not ok:
        print(f"[ERROR] Không thể lưu: {output_path}")
        return False

    print(f"[SAVED] {output_path}")
    return True


def main():
    global current_mask

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index = 0

    window_name = "Ground Truth Annotation"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        800,
        1000
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    load_scene(index)

    while True:
        display = make_display()

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        # ESC
        if key == 27:
            print("Thoát.")
            break

        # S = save
        elif key in (ord("s"), ord("S")):
            save_current()

        # N = save + next
        elif key in (ord("n"), ord("N")):
            save_current()

            if index < len(SCENES) - 1:
                index += 1
                load_scene(index)
            else:
                print()
                print("=" * 70)
                print("ĐÃ HOÀN THÀNH SCENE 10")
                print("=" * 70)
                break

        # B = previous
        elif key in (ord("b"), ord("B")):
            if index > 0:
                save_current()
                index -= 1
                load_scene(index)

        # C = clear
        elif key in (ord("c"), ord("C")):
            current_mask[:] = 0
            print(f"[CLEAR] {current_scene}")

    cv2.destroyAllWindows()

    print()
    print("=== GROUND TRUTH FILES ===")

    for scene_id, _ in SCENES:
        output = OUTPUT_DIR / f"{scene_id}_gt.png"

        if output.exists():
            img = cv2.imread(
                str(output),
                cv2.IMREAD_GRAYSCALE
            )

            if img is not None:
                white_pixels = int(np.sum(img > 0))

                print(
                    f"{output.name}: "
                    f"size={img.shape[1]}x{img.shape[0]}, "
                    f"edge_pixels={white_pixels}"
                )
        else:
            print(f"{output.name}: MISSING")


if __name__ == "__main__":
    main()