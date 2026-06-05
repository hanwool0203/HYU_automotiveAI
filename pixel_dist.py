#!/usr/bin/env python3

import cv2
import math
import argparse
import os


points = []
fixed_y = None


def mouse_callback(event, x, y, flags, param):
    global points, fixed_y

    image = param["image"]
    display = param["display"]

    if event == cv2.EVENT_LBUTTONDOWN:
        # 첫 번째 클릭의 y좌표를 기준 y로 고정
        if fixed_y is None:
            fixed_y = y
            print(f"[INFO] Fixed y set to: {fixed_y}")

        # 이후 모든 점은 x만 사용하고 y는 fixed_y로 고정
        y = fixed_y

        points.append((x, y))

        idx = len(points)

        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            display,
            f"P{idx} ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        print(f"P{idx}: x={x}, y={y}")

        # 기준 y 라인 표시
        cv2.line(
            display,
            (0, fixed_y),
            (image.shape[1], fixed_y),
            (255, 0, 0),
            1
        )

        # 점이 2개 이상이면 직전 점과 거리 계산
        if len(points) >= 2:
            p1 = points[-2]
            p2 = points[-1]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx ** 2 + dy ** 2)

            print("----------")
            print(f"P{idx-1} -> P{idx}")
            print(f"dx = {dx} px")
            print(f"dy = {dy} px")
            print(f"distance = {dist:.2f} px")
            print(f"x-direction distance = {abs(dx)} px")
            print(f"y-direction distance = {abs(dy)} px")
            print("----------")

            cv2.line(display, p1, p2, (0, 255, 0), 2)

            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = fixed_y

            cv2.putText(
                display,
                f"{abs(dx)}px",
                (mid_x + 10, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(param["window_name"], display)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 우클릭하면 마지막 점 삭제
        if points:
            removed = points.pop()
            print(f"Removed last point: {removed}")
            redraw(param)


def redraw(param):
    display = param["image"].copy()

    for i, point in enumerate(points):
        x, y = point
        idx = i + 1

        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            display,
            f"P{idx} ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    for i in range(1, len(points)):
        p1 = points[i - 1]
        p2 = points[i]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)

        cv2.line(display, p1, p2, (0, 255, 0), 2)

        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)

        cv2.putText(
            display,
            f"{dist:.1f}px",
            (mid_x + 10, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    param["display"][:] = display[:]
    cv2.imshow(param["window_name"], param["display"])


def print_all_measurements():
    if len(points) < 2:
        print("[INFO] Need at least 2 points.")
        return

    print("\n==============================")
    print("All measurements")
    print("==============================")

    for i in range(1, len(points)):
        p1 = points[i - 1]
        p2 = points[i]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)

        print(f"P{i} -> P{i+1}")
        print(f"  P{i}   = {p1}")
        print(f"  P{i+1} = {p2}")
        print(f"  dx = {dx} px")
        print(f"  dy = {dy} px")
        print(f"  distance = {dist:.2f} px")
        print(f"  x distance = {abs(dx)} px")
        print(f"  y distance = {abs(dy)} px")

    print("==============================\n")


def main():
    DEFAULT_IMAGE_PATH = "/home/ircv7/workspace/ros2_ws/capture.png"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path",
        nargs="?",
        default=DEFAULT_IMAGE_PATH,
        help="Path to BEV debug image"
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Path to save annotated image"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image not found: {args.image_path}")
        return

    image = cv2.imread(args.image_path)

    if image is None:
        print(f"[ERROR] Failed to read image: {args.image_path}")
        return

    window_name = "measure_pixel_distance"
    display = image.copy()

    param = {
        "image": image,
        "display": display,
        "window_name": window_name,
    }

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, mouse_callback, param)

    print("[INFO] Image opened.")
    print("[INFO] Left click : add point")
    print("[INFO] Right click: remove last point")
    print("[INFO] p          : print all measurements")
    print("[INFO] s          : save annotated image")
    print("[INFO] r          : reset points")
    print("[INFO] q or ESC   : quit")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('p'):
            print_all_measurements()

        elif key == ord('r'):
            global fixed_y
            points.clear()
            fixed_y = None
            param["display"][:] = image.copy()
            cv2.imshow(window_name, param["display"])
            print("[INFO] Reset points and fixed y.")

        elif key == ord('s'):
            if args.save is not None:
                save_path = args.save
            else:
                base, ext = os.path.splitext(args.image_path)
                save_path = base + "_measured.png"

            cv2.imwrite(save_path, param["display"])
            print(f"[INFO] Saved annotated image: {save_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()