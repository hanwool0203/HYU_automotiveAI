import os
import glob
import cv2
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", default="./stereo_calib_images")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)

    # 체커보드 내부 코너 개수
    parser.add_argument("--cols", type=int, default=8, help="checkerboard inner corners cols")
    parser.add_argument("--rows", type=int, default=6, help="checkerboard inner corners rows")

    # 체커보드 한 칸 크기, meter 단위
    parser.add_argument("--square", type=float, default=0.025, help="checkerboard square size in meter")

    parser.add_argument("--out", default="stereo_calib_result_640x360.npz")

    args = parser.parse_args()

    IMAGE_SIZE = (args.width, args.height)
    CHECKERBOARD = (args.cols, args.rows)
    SQUARE_SIZE = args.square

    # ============================================================
    # 네가 이미 구한 intrinsic 값
    # ============================================================
    K1 = np.array([
        [395.39461,   0.     , 322.36372],
        [  0.     , 396.08816, 172.76244],
        [  0.0,   0.0,   1.0]
    ], dtype=np.float64)

    D1 = np.array([
        -0.313120, 0.086347, -0.001000, -0.001350, 0.000000
    ], dtype=np.float64)

    K2 = np.array([
        [396.47452,   0.     , 310.8965],
        [  0.     , 397.44982, 181.63141],
        [  0.0,   0.0,   1.0]
    ], dtype=np.float64)

    D2 = np.array([
        -0.326130, 0.092355, -0.000391, 0.001183, 0.000000
    ], dtype=np.float64)

    left_dir = os.path.join(args.image_dir, "left")
    right_dir = os.path.join(args.image_dir, "right")

    left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))

    print("Left images:", len(left_images))
    print("Right images:", len(right_images))

    if len(left_images) != len(right_images):
        print("left/right 이미지 개수가 다릅니다.")
        return

    if len(left_images) == 0:
        print("이미지가 없습니다.")
        return

    # 체커보드 3D 좌표
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5
    )

    used_count = 0

    for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        left = cv2.imread(left_path)
        right = cv2.imread(right_path)

        if left is None or right is None:
            print(f"[{idx}] 이미지 로드 실패")
            continue

        if left.shape[1] != args.width or left.shape[0] != args.height:
            left = cv2.resize(left, IMAGE_SIZE)

        if right.shape[1] != args.width or right.shape[0] != args.height:
            right = cv2.resize(right, IMAGE_SIZE)

        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(
            gray_l,
            CHECKERBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        ret_r, corners_r = cv2.findChessboardCorners(
            gray_r,
            CHECKERBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(
                gray_l,
                corners_l,
                (11, 11),
                (-1, -1),
                criteria
            )

            corners_r = cv2.cornerSubPix(
                gray_r,
                corners_r,
                (11, 11),
                (-1, -1),
                criteria
            )

            objpoints.append(objp.copy())
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

            used_count += 1

            print(f"[{idx:04d}] OK")

            vis_l = left.copy()
            vis_r = right.copy()

            cv2.drawChessboardCorners(vis_l, CHECKERBOARD, corners_l, ret_l)
            cv2.drawChessboardCorners(vis_r, CHECKERBOARD, corners_r, ret_r)

            combined = np.hstack((vis_l, vis_r))

            cv2.putText(
                combined,
                f"OK pair {idx}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("checkerboard detection", combined)
            key = cv2.waitKey(80) & 0xFF
            if key == ord("q"):
                break

        else:
            print(f"[{idx:04d}] FAIL left={ret_l}, right={ret_r}")

    cv2.destroyAllWindows()

    print()
    print("Used stereo pairs:", used_count)

    if used_count < 15:
        print("사용 가능한 이미지쌍이 너무 적습니다.")
        print("최소 15쌍, 권장 30쌍 이상을 추천합니다.")
        return

    # ============================================================
    # Stereo calibration
    # 내부 파라미터는 고정하고 R, T만 추정
    # ============================================================
    flags = cv2.CALIB_FIX_INTRINSIC

    rms, K1_new, D1_new, K2_new, D2_new, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        K1,
        D1,
        K2,
        D2,
        IMAGE_SIZE,
        criteria=criteria,
        flags=flags
    )

    print()
    print("==========================================")
    print("Stereo calibration result")
    print("==========================================")
    print("RMS error:", rms)
    print()
    print("R =")
    print(R)
    print()
    print("T =")
    print(T)
    print()
    print("Baseline norm:", np.linalg.norm(T), "m")

    np.savez(
        args.out,
        K1=K1_new,
        D1=D1_new,
        K2=K2_new,
        D2=D2_new,
        R=R,
        T=T,
        E=E,
        F=F,
        rms=rms,
        image_width=args.width,
        image_height=args.height,
        checkerboard_cols=args.cols,
        checkerboard_rows=args.rows,
        square_size=SQUARE_SIZE
    )

    print()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()