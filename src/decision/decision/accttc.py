#!/usr/bin/env python3

import json
import time
import math

import cv2

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np


class ACCTTCNode(Node):
    """
    ACC + TTC node.

    변경된 구조:
    1. slow_sign 판단은 하지 않음
       - slow_sign 판단은 통합 판단 노드에서 수행
       - 통합 판단 노드가 /mission_mode = "ttc" publish

    2. 이 노드는 /mission_mode를 구독함
       - mission_mode == "ttc"일 때만 ACC/TTC 제어 활성화
       - mission_mode != "ttc"이면 normal 주행 속도 publish

    3. TTC 모드 중 box가 보이면 depth 기반 추종
       - box와의 거리
       - closing speed
       - TTC
       - distance following speed 계산

    4. pure_pursuit에는 아래 두 토픽만 보냄
       - /drive_mode = normal 또는 acc
       - /acc/target_speed_mps
    """

    def __init__(self):
        super().__init__("acc_ttc_node")

        # ==============================
        # Topics
        # ==============================
        self.declare_parameter("yolo_detection_topic", "/yolo/left_detections")
        self.declare_parameter("depth_topic", "/stereo/depth/image_raw")

        self.declare_parameter("mission_mode_topic", "/mission_mode")

        self.declare_parameter("drive_mode_topic", "/drive_mode")
        self.declare_parameter("acc_speed_topic", "/acc/target_speed_mps")

        # ==============================
        # Mission mode
        # ==============================
        self.declare_parameter("ttc_mission_mode", "ttc")

        # ==============================
        # box filtering
        # ==============================
        self.declare_parameter("box_conf_threshold", 0.45)
        self.declare_parameter("box_min_area", 500.0)

        # 화면 중앙 근처 box만 ACC/TTC 추종 대상으로 사용
        self.declare_parameter("box_center_min", 0.15)
        self.declare_parameter("box_center_max", 0.85)

        # ==============================
        # Speed parameters
        # pure pursuit의 base_speed_mps와 맞추는 값
        # ==============================
        self.declare_parameter("base_speed_mps", 0.25)
        self.declare_parameter("max_acc_speed_mps", 0.25)
        self.declare_parameter("min_follow_speed_mps", 0.06)
        self.declare_parameter("depth_invalid_slow_speed_mps", 0.15)

        # 너무 작은 속도 명령으로 덜덜거리는 것 방지
        self.declare_parameter("speed_deadband_mps", 0.025)

        # 속도 변화율 제한
        self.declare_parameter("speed_rise_rate_mps2", 0.20)
        self.declare_parameter("speed_fall_rate_mps2", 0.50)

        # ==============================
        # Following distance / TTC
        # ==============================
        self.declare_parameter("target_follow_distance_m", 0.30)
        self.declare_parameter("emergency_distance_m", 0.18)

        # 거리 오차 기반 속도 gain
        # speed = distance_kp * (depth - target_follow_distance)
        self.declare_parameter("distance_kp", 0.55)

        # TTC 기준
        self.declare_parameter("ttc_stop_threshold_sec", 0.8)
        self.declare_parameter("ttc_slow_threshold_sec", 1.5)
        self.declare_parameter("ttc_slow_scale", 0.45)

        # ==============================
        # Depth representative options
        # ==============================
        self.declare_parameter("depth_min_m", 0.05)
        self.declare_parameter("depth_max_m", 3.0)

        # bbox 전체가 아니라 중앙 영역만 사용
        self.declare_parameter("bbox_inner_ratio", 0.60)

        # depth median 계산 최소 유효 픽셀 수
        self.declare_parameter("min_valid_depth_pixels", 10)

        # depth low-pass filter
        self.declare_parameter("depth_filter_alpha", 0.35)

        # relative speed low-pass filter
        self.declare_parameter("relative_speed_filter_alpha", 0.40)

        # ==============================
        # Load params
        # ==============================
        self.yolo_detection_topic = self.get_parameter("yolo_detection_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.mission_mode_topic = self.get_parameter("mission_mode_topic").value

        self.drive_mode_topic = self.get_parameter("drive_mode_topic").value
        self.acc_speed_topic = self.get_parameter("acc_speed_topic").value

        self.ttc_mission_mode = (
            self.get_parameter("ttc_mission_mode").value.strip().lower()
        )

        self.box_conf_threshold = float(
            self.get_parameter("box_conf_threshold").value
        )
        self.box_min_area = float(
            self.get_parameter("box_min_area").value
        )
        self.box_center_min = float(
            self.get_parameter("box_center_min").value
        )
        self.box_center_max = float(
            self.get_parameter("box_center_max").value
        )

        self.base_speed_mps = float(
            self.get_parameter("base_speed_mps").value
        )
        self.max_acc_speed_mps = float(
            self.get_parameter("max_acc_speed_mps").value
        )
        self.min_follow_speed_mps = float(
            self.get_parameter("min_follow_speed_mps").value
        )
        self.speed_deadband_mps = float(
            self.get_parameter("speed_deadband_mps").value
        )
        self.speed_rise_rate_mps2 = float(
            self.get_parameter("speed_rise_rate_mps2").value
        )
        self.speed_fall_rate_mps2 = float(
            self.get_parameter("speed_fall_rate_mps2").value
        )
        self.depth_invalid_slow_speed_mps = float(
            self.get_parameter("depth_invalid_slow_speed_mps").value
        )

        self.target_follow_distance_m = float(
            self.get_parameter("target_follow_distance_m").value
        )
        self.emergency_distance_m = float(
            self.get_parameter("emergency_distance_m").value
        )
        self.distance_kp = float(
            self.get_parameter("distance_kp").value
        )
        self.ttc_stop_threshold_sec = float(
            self.get_parameter("ttc_stop_threshold_sec").value
        )
        self.ttc_slow_threshold_sec = float(
            self.get_parameter("ttc_slow_threshold_sec").value
        )
        self.ttc_slow_scale = float(
            self.get_parameter("ttc_slow_scale").value
        )

        self.depth_min_m = float(
            self.get_parameter("depth_min_m").value
        )
        self.depth_max_m = float(
            self.get_parameter("depth_max_m").value
        )
        self.bbox_inner_ratio = float(
            self.get_parameter("bbox_inner_ratio").value
        )
        self.min_valid_depth_pixels = int(
            self.get_parameter("min_valid_depth_pixels").value
        )
        self.depth_filter_alpha = float(
            self.get_parameter("depth_filter_alpha").value
        )
        self.relative_speed_filter_alpha = float(
            self.get_parameter("relative_speed_filter_alpha").value
        )

        # ==============================
        # State
        # ==============================
        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_depth_time = None

        self.mission_mode = "normal"
        self.acc_enabled = False

        self.filtered_depth = None
        self.prev_depth = None
        self.prev_depth_time = None

        # positive = box와 가까워지는 중
        self.filtered_closing_speed = 0.0

        self.last_target_speed = self.base_speed_mps
        self.last_speed_publish_time = time.time()

        # ==============================
        # Left camera calibration
        # YOLO bbox: /left/image_raw 기준
        # Depth image: /stereo/left_rect 기준
        # 따라서 raw pixel -> rectified pixel 변환에 사용
        # ==============================

        self.image_width = 640
        self.image_height = 360
        self.image_size = (self.image_width, self.image_height)

        self.K1 = np.array([
            [395.39461,   0.     , 322.36372],
            [  0.     , 396.08816, 172.76244],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)

        self.D1 = np.array([
            -0.313120, 0.086347, -0.001000, -0.001350, 0.000000
        ], dtype=np.float64)

        self.K2 = np.array([
            [396.47452,   0.     , 310.8965],
            [  0.     , 397.44982, 181.63141],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)

        self.D2 = np.array([
            -0.326130, 0.092355, -0.000391, 0.001183, 0.000000
        ], dtype=np.float64)

        self.R = np.array([
            [0.99954904, -0.00461258, 0.02967218],
            [0.00468782, 0.99998597, -0.00246655],
            [-0.02966038, 0.00260453, 0.99955664]
        ], dtype=np.float64)

        self.T = np.array([
            [-0.03802564],
            [-0.00080493],
            [-0.00858097]
        ], dtype=np.float64)

        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(
            self.K1,
            self.D1,
            self.K2,
            self.D2,
            self.image_size,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        # ==============================
        # ROS interfaces
        # ==============================
        self.mission_mode_sub = self.create_subscription(
            String,
            self.mission_mode_topic,
            self.mission_mode_callback,
            10
        )

        self.yolo_sub = self.create_subscription(
            String,
            self.yolo_detection_topic,
            self.yolo_callback,
            10
        )

        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            depth_qos
        )

        self.drive_mode_pub = self.create_publisher(
            String,
            self.drive_mode_topic,
            10
        )

        self.acc_speed_pub = self.create_publisher(
            Float32,
            self.acc_speed_topic,
            10
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("ACC TTC Node started")
        self.get_logger().info(f"Subscribing mission mode: {self.mission_mode_topic}")
        self.get_logger().info(f"Subscribing YOLO        : {self.yolo_detection_topic}")
        self.get_logger().info(f"Subscribing depth       : {self.depth_topic}")
        self.get_logger().info(f"Publishing drive mode   : {self.drive_mode_topic}")
        self.get_logger().info(f"Publishing ACC speed    : {self.acc_speed_topic}")
        self.get_logger().info("slow_sign detection is removed from ACC-TTC node")
        self.get_logger().info("ACC-TTC is enabled only when mission_mode == ttc")

        # 초기 상태 publish
        self.publish_drive_mode("normal")
        self.publish_acc_speed(self.base_speed_mps)

    # ============================================================
    # ROS callbacks
    # ============================================================
    def mission_mode_callback(self, msg):
        new_mode = msg.data.strip().lower()

        if new_mode == self.mission_mode:
            return

        old_mode = self.mission_mode
        self.mission_mode = new_mode

        self.get_logger().warn(
            f"Mission mode changed: {old_mode} -> {self.mission_mode}"
        )

        if self.is_ttc_mode():
            self.acc_enabled = True
            self.reset_tracking_state()

            self.publish_drive_mode("acc")
            self.publish_acc_speed(self.base_speed_mps)

            self.get_logger().warn("TTC mission ON -> ACC/TTC control enabled")

        else:
            self.acc_enabled = False
            self.reset_tracking_state()

            self.publish_drive_mode("normal")
            self.publish_acc_speed(self.base_speed_mps)

            self.get_logger().warn("TTC mission OFF -> normal drive mode")

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="32FC1"
            )
        except Exception as e:
            self.get_logger().warn(f"Depth cv_bridge error: {e}")
            return

        self.latest_depth = depth
        self.latest_depth_time = time.time()

    def yolo_callback(self, msg):
        # TTC 미션이 아닐 때는 YOLO box를 추종 대상으로 사용하지 않음
        if not self.is_ttc_mode():
            self.acc_enabled = False
            self.publish_drive_mode("normal")
            self.publish_acc_speed(self.base_speed_mps)
            return

        self.acc_enabled = True

        try:
            data = json.loads(msg.data)
            image_width = int(data.get("image_width", 640))
            image_height = int(data.get("image_height", 360))
            detections = data.get("detections", [])
        except Exception as e:
            self.get_logger().warn(f"Failed to parse YOLO JSON: {e}")
            return

        best_box = None

        for det in detections:
            class_name = det.get("class_name", "")
            conf = float(det.get("confidence", 0.0))

            bbox = det.get("bbox", {})
            size = det.get("size", {})
            center = det.get("center", {})

            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            y2 = float(bbox.get("y2", 0.0))

            w = float(size.get("w", max(0.0, x2 - x1)))
            h = float(size.get("h", max(0.0, y2 - y1)))
            area = w * h

            center_x_px = float(center.get("x", (x1 + x2) / 2.0))
            center_x_norm = center_x_px / max(float(image_width), 1.0)

            # ==============================
            # box: ACC/TTC target
            # ==============================
            if (
                class_name == "box"
                and conf >= self.box_conf_threshold
                and area >= self.box_min_area
                and self.box_center_min <= center_x_norm <= self.box_center_max
            ):
                candidate = {
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "area": area,
                    "image_width": image_width,
                    "image_height": image_height,
                    "center_x_norm": center_x_norm,
                }

                # bbox 면적이 큰 box를 추종 대상으로 선택
                if best_box is None or area > best_box["area"]:
                    best_box = candidate

        self.update_acc_speed(best_box)

    # ============================================================
    # Mode / State
    # ============================================================
    def is_ttc_mode(self):
        return self.mission_mode == self.ttc_mission_mode

    def reset_tracking_state(self):
        self.filtered_depth = None
        self.prev_depth = None
        self.prev_depth_time = None
        self.filtered_closing_speed = 0.0
        self.last_target_speed = self.base_speed_mps
        self.last_speed_publish_time = time.time()

    # ============================================================
    # ACC/TTC speed logic
    # ============================================================
    def update_acc_speed(self, best_box):
        if not self.is_ttc_mode():
            self.acc_enabled = False
            self.publish_drive_mode("normal")
            self.publish_acc_speed(self.base_speed_mps)
            return

        self.acc_enabled = True
        self.publish_drive_mode("acc")

        if best_box is None:
            # TTC 모드는 유지하되, 추종 대상이 없으면 기본 속도로 주행
            self.reset_tracking_state()

            target_speed = self.clip(
                self.base_speed_mps,
                0.0,
                self.max_acc_speed_mps
            )

            self.publish_acc_speed(target_speed)

            self.get_logger().info(
                f"TTC | no box | keep ACC mode | speed={target_speed:.3f} m/s",
                throttle_duration_sec=0.5
            )
            return

        if self.latest_depth is None:
            target_speed = 0.0
            self.publish_acc_speed(target_speed)

            self.get_logger().warn(
                "TTC | no depth image | speed=0",
                throttle_duration_sec=0.5
            )
            return

        raw_depth = self.get_representative_depth(best_box)

        if raw_depth is None:
            # box는 보이지만 depth가 안 잡히는 상황
            # 완전 정지하지 않고 depth가 다시 잡힐 때까지 천천히 서행
            target_speed = self.clip(
                self.depth_invalid_slow_speed_mps,
                0.0,
                self.max_acc_speed_mps
            )

            target_speed = self.rate_limit_speed(target_speed)

            if target_speed < self.speed_deadband_mps:
                target_speed = 0.0

            self.publish_acc_speed(target_speed)

            self.get_logger().warn(
                f"TTC | box detected but depth invalid | "
                f"slow creeping speed={target_speed:.3f} m/s",
                throttle_duration_sec=0.5
            )
            return

        now = time.time()

        # ------------------------------
        # 1. Depth filtering
        # ------------------------------
        if self.filtered_depth is None:
            self.filtered_depth = raw_depth
        else:
            alpha = self.depth_filter_alpha
            self.filtered_depth = (
                alpha * raw_depth
                + (1.0 - alpha) * self.filtered_depth
            )

        depth = self.filtered_depth

        # ------------------------------
        # 2. Closing speed calculation
        # positive = 가까워지는 중
        # ------------------------------
        closing_speed_raw = 0.0

        if self.prev_depth is not None and self.prev_depth_time is not None:
            dt = now - self.prev_depth_time

            if dt > 1e-3:
                closing_speed_raw = -(depth - self.prev_depth) / dt

        self.prev_depth = depth
        self.prev_depth_time = now

        beta = self.relative_speed_filter_alpha
        self.filtered_closing_speed = (
            beta * closing_speed_raw
            + (1.0 - beta) * self.filtered_closing_speed
        )

        closing_speed = max(self.filtered_closing_speed, 0.0)

        if closing_speed > 0.01:
            ttc = depth / closing_speed
        else:
            ttc = float("inf")

        # ------------------------------
        # 3. Distance-following speed
        # ------------------------------
        distance_error = depth - self.target_follow_distance_m

        if depth <= self.emergency_distance_m:
            target_speed = 0.0
            reason = "emergency_distance"

        elif ttc < self.ttc_stop_threshold_sec:
            target_speed = 0.0
            reason = "ttc_stop"

        else:
            if distance_error <= 0.0:
                target_speed = 0.0
                reason = "too_close"

            else:
                target_speed = self.distance_kp * distance_error
                target_speed = self.clip(
                    target_speed,
                    0.0,
                    self.max_acc_speed_mps
                )
                reason = "follow"

                if ttc < self.ttc_slow_threshold_sec:
                    target_speed *= self.ttc_slow_scale
                    reason = "ttc_slow"

                if 0.0 < target_speed < self.min_follow_speed_mps:
                    target_speed = self.min_follow_speed_mps

        target_speed = self.rate_limit_speed(target_speed)

        if target_speed < self.speed_deadband_mps:
            target_speed = 0.0

        self.publish_acc_speed(target_speed)

        self.get_logger().warn(
            f"TTC | reason={reason} | "
            f"raw_depth={raw_depth:.3f} m | "
            f"depth={depth:.3f} m | "
            f"closing={closing_speed:.3f} m/s | "
            f"ttc={self.format_ttc(ttc)} | "
            f"target_speed={target_speed:.3f} m/s",
            throttle_duration_sec=0.3
        )

    def raw_to_rect_px(self, x, y, yolo_w=None, yolo_h=None):
        """
        /left/image_raw 기준 pixel 좌표를
        /stereo/left_rect 기준 pixel 좌표로 변환
        """

        # YOLO 이미지 크기가 calibration 크기와 다르면 먼저 640x360 기준으로 스케일
        if yolo_w is not None and yolo_h is not None:
            sx = self.image_width / max(float(yolo_w), 1.0)
            sy = self.image_height / max(float(yolo_h), 1.0)
            x = x * sx
            y = y * sy

        pts = np.array([[[float(x), float(y)]]], dtype=np.float32)

        rect = cv2.undistortPoints(
            pts,
            self.K1,
            self.D1,
            R=self.R1,
            P=self.P1
        )

        u = float(rect[0, 0, 0])
        v = float(rect[0, 0, 1])

        return u, v

    def raw_bbox_to_rect_bbox(self, bbox, yolo_w, yolo_h):
        """
        YOLO raw bbox를 rectified bbox로 변환
        """

        x1, y1, x2, y2 = bbox

        points = [
            self.raw_to_rect_px(x1, y1, yolo_w, yolo_h),
            self.raw_to_rect_px(x2, y1, yolo_w, yolo_h),
            self.raw_to_rect_px(x2, y2, yolo_w, yolo_h),
            self.raw_to_rect_px(x1, y2, yolo_w, yolo_h),
        ]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        rx1 = int(round(min(xs)))
        ry1 = int(round(min(ys)))
        rx2 = int(round(max(xs)))
        ry2 = int(round(max(ys)))

        return rx1, ry1, rx2, ry2

    def get_representative_depth(self, box):
        if self.latest_depth is None:
            return None

        depth = self.latest_depth
        h_img, w_img = depth.shape[:2]

        raw_bbox = box["bbox"]

        yolo_w = float(box.get("image_width", self.image_width))
        yolo_h = float(box.get("image_height", self.image_height))

        x1, y1, x2, y2 = self.raw_bbox_to_rect_bbox(
            raw_bbox,
            yolo_w,
            yolo_h
        )

        x1 = self.clip_int(x1, 0, w_img - 1)
        x2 = self.clip_int(x2, 0, w_img - 1)
        y1 = self.clip_int(y1, 0, h_img - 1)
        y2 = self.clip_int(y2, 0, h_img - 1)

        if x2 <= x1 or y2 <= y1:
            return None

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        bw = x2 - x1
        bh = y2 - y1

        inner_w = max(2, int(bw * self.bbox_inner_ratio))
        inner_h = max(2, int(bh * self.bbox_inner_ratio))

        ix1 = self.clip_int(cx - inner_w // 2, 0, w_img - 1)
        ix2 = self.clip_int(cx + inner_w // 2, 0, w_img - 1)
        iy1 = self.clip_int(cy - inner_h // 2, 0, h_img - 1)
        iy2 = self.clip_int(cy + inner_h // 2, 0, h_img - 1)

        if ix2 <= ix1 or iy2 <= iy1:
            return None

        roi = depth[iy1:iy2, ix1:ix2]

        if roi.size == 0:
            return None

        valid = roi[np.isfinite(roi)]
        valid = valid[
            (valid >= self.depth_min_m)
            & (valid <= self.depth_max_m)
        ]

        if valid.size < self.min_valid_depth_pixels:
            return None

        return float(np.median(valid))

    # ============================================================
    # Timer
    # ============================================================
    def timer_callback(self):
        # TTC 미션이 아니면 무조건 normal
        if not self.is_ttc_mode():
            self.acc_enabled = False
            self.publish_drive_mode("normal")
            self.publish_acc_speed(self.base_speed_mps)
            return

        # TTC 미션이면 acc mode 유지
        self.acc_enabled = True
        self.publish_drive_mode("acc")

        self.publish_acc_speed(self.last_target_speed)

    # ============================================================
    # Publishers
    # ============================================================
    def publish_drive_mode(self, mode):
        msg = String()
        msg.data = mode
        self.drive_mode_pub.publish(msg)

    def publish_acc_speed(self, speed_mps):
        speed_mps = self.clip(
            speed_mps,
            0.0,
            self.max_acc_speed_mps
        )

        self.last_target_speed = speed_mps

        msg = Float32()
        msg.data = float(speed_mps)
        self.acc_speed_pub.publish(msg)

    # ============================================================
    # Utils
    # ============================================================
    def rate_limit_speed(self, target_speed):
        now = time.time()
        dt = now - self.last_speed_publish_time

        if dt <= 0.0:
            return target_speed

        prev = self.last_target_speed

        if target_speed > prev:
            max_delta = self.speed_rise_rate_mps2 * dt
        else:
            max_delta = self.speed_fall_rate_mps2 * dt

        limited = self.clip(
            target_speed,
            prev - max_delta,
            prev + max_delta
        )

        self.last_speed_publish_time = now
        return limited

    @staticmethod
    def clip(value, min_value, max_value):
        return max(min(value, max_value), min_value)

    @staticmethod
    def clip_int(value, min_value, max_value):
        return int(max(min(value, max_value), min_value))

    @staticmethod
    def format_ttc(ttc):
        if math.isinf(ttc):
            return "inf"
        return f"{ttc:.2f}s"


def main(args=None):
    rclpy.init(args=args)
    node = ACCTTCNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()