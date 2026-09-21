import argparse
from pathlib import Path

import cv2
import numpy as np


drawing = False
gt_canvas = None
previous_point = None


def draw_callback(event, x, y, flags, param):
    global drawing, gt_canvas, previous_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        previous_point = (x, y)
        cv2.circle(gt_canvas, (x, y), 1, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current = (x, y)
        if previous_point is not None:
            cv2.line(gt_canvas, previous_point, current, 255, thickness=1)
        previous_point = current
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current = (x, y)
        if previous_point is not None:
            cv2.line(gt_canvas, previous_point, current, 255, thickness=1)
        previous_point = None


def annotate(img_path: str, out_path: str) -> None:
    global gt_canvas

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gt_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
    window = "Manual Edge Annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, draw_callback)

    print("Left-drag: draw edge")
    print("c: clear | s: save | q: quit without saving")

    while True:
        display = cv2.addWeighted(img, 0.45, np.zeros_like(img), 0.55, 0)
        display[gt_canvas > 0] = (255, 255, 255)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("c"):
            gt_canvas.fill(0)
        elif key == ord("s"):
            output = Path(out_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), gt_canvas):
                raise IOError(f"Failed to save Ground Truth: {output}")
            print(f"Saved Ground Truth: {output}")
            break
        elif key == ord("q"):
            print("Annotation cancelled without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual edge annotation tool.")
    parser.add_argument("--img", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        annotate(args.img, args.out)
    except Exception as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
