#!/usr/bin/env python3

import json
import time
import math

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class RoverTTCSafetyNode(Node):
    def __init__(self):
        super().__init__("rover_ttc_safety_node")

        # ============================================================
        # Topics
        # ============================================================
        self.declare_parameter("yolo_detections_topic", "/yolo/left_detections")
        self.declare_parameter("depth_topic", "/stereo/depth/image_raw")

        # sign_decision과 충돌 방지를 위해 기본은 /safety/drive_mode
        # 필요하면 실행 시 -p drive_mode_topic:=/drive_mode 로 바꿀 수 있음
        self.declare_parameter("drive_mode_topic", "/safety/drive_mode")
        self.declare_parameter("ttc_info_topic", "/safety/ttc_info")

        # ============================================================
        # YOLO / depth parameters
        # ============================================================
        self.declare_parameter("rover_label", "other_rover")
        self.declare_parameter("conf_threshold", 0.35)

        # bbox 안쪽만 사용해서 depth 노이즈 줄이기
        self.declare_parameter("bbox_margin_ratio", 0.10)

        # bbox 안 depth 중 가까운 쪽 percentile 사용
        # 5.0이면 가까운 쪽 5% 지점
        self.declare_parameter("depth_percentile", 5.0)

        self.declare_parameter("depth_timeout_sec", 0.3)
        self.declare_parameter("detection_timeout_sec", 0.5)

        # ============================================================
        # TTC / safety parameters
        # ============================================================
        # 거리가 이 값보다 가까우면 TTC와 상관없이 즉시 정지
        self.declare_parameter("emergency_distance_m", 0.30)

        # 이 거리 이내면 안전상 정지
        self.declare_parameter("stop_distance_m", 0.45)

        # 이 거리 이내면 서행
        self.declare_parameter("slow_distance_m", 0.80)

        # TTC가 이 값보다 작으면 정지
        self.declare_parameter("ttc_stop_sec", 1.0)

        # TTC가 이 값보다 작으면 서행
        self.declare_parameter("ttc_slow_sec", 2.0)

        # closing speed가 너무 작으면 TTC 계산을 무시
        self.declare_parameter("min_closing_speed_mps", 0.05)

        # 속도 추정 smoothing
        # 0.0이면 이전값만, 1.0이면 현재 측정값만 사용
        self.declare_parameter("closing_speed_alpha", 0.4)

        # flicker 방지용 hold
        self.declare_parameter("stop_hold_sec", 0.8)
        self.declare_parameter("slow_hold_sec", 0.5)

        # publish 주기
        self.declare_parameter("publish_rate_hz", 20.0)

        # ============================================================
        # Read parameters
        # ============================================================
        self.yolo_topic = self.get_parameter("yolo_detections_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.drive_mode_topic = self.get_parameter("drive_mode_topic").value
        self.ttc_info_topic = self.get_parameter("ttc_info_topic").value

        self.rover_label = self.get_parameter("rover_label").value.lower()
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)

        self.bbox_margin_ratio = float(self.get_parameter("bbox_margin_ratio").value)
        self.depth_percentile = float(self.get_parameter("depth_percentile").value)
        self.depth_timeout_sec = float(self.get_parameter("depth_timeout_sec").value)
        self.detection_timeout_sec = float(
            self.get_parameter("detection_timeout_sec").value
        )

        self.emergency_distance_m = float(
            self.get_parameter("emergency_distance_m").value
        )
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.slow_distance_m = float(self.get_parameter("slow_distance_m").value)

        self.ttc_stop_sec = float(self.get_parameter("ttc_stop_sec").value)
        self.ttc_slow_sec = float(self.get_parameter("ttc_slow_sec").value)

        self.min_closing_speed_mps = float(
            self.get_parameter("min_closing_speed_mps").value
        )
        self.closing_speed_alpha = float(
            self.get_parameter("closing_speed_alpha").value
        )

        self.stop_hold_sec = float(self.get_parameter("stop_hold_sec").value)
        self.slow_hold_sec = float(self.get_parameter("slow_hold_sec").value)

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        timer_period = 1.0 / max(publish_rate_hz, 1.0)

        # ============================================================
        # State
        # ============================================================
        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_depth_time = 0.0

        self.prev_rover_depth = None
        self.prev_rover_time = None
        self.filtered_closing_speed = 0.0

        self.last_detection_time = 0.0
        self.last_mode = "normal"

        self.stop_until = 0.0
        self.slow_until = 0.0

        self.last_info = {
            "mode": "normal",
            "reason": "init",
            "depth_m": None,
            "closing_speed_mps": 0.0,
            "ttc_sec": None,
            "confidence": None,
        }

        # ============================================================
        # QoS
        # /stereo/depth/image_raw는 stereo_depth_node에서 BEST_EFFORT로 publish함
        # ============================================================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ============================================================
        # ROS pubs/subs
        # ============================================================
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            sensor_qos,
        )

        self.yolo_sub = self.create_subscription(
            String,
            self.yolo_topic,
            self.yolo_callback,
            10,
        )

        self.drive_mode_pub = self.create_publisher(
            String,
            self.drive_mode_topic,
            10,
        )

        self.ttc_info_pub = self.create_publisher(
            String,
            self.ttc_info_topic,
            10,
        )

        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info("Rover TTC Safety Node Started")
        self.get_logger().info(f"Sub YOLO   : {self.yolo_topic}")
        self.get_logger().info(f"Sub depth  : {self.depth_topic}")
        self.get_logger().info(f"Pub mode   : {self.drive_mode_topic}")
        self.get_logger().info(f"Pub TTC    : {self.ttc_info_topic}")
        self.get_logger().info(
            f"label={self.rover_label}, conf>={self.conf_threshold}, "
            f"stop_distance={self.stop_distance_m:.2f}m, "
            f"slow_distance={self.slow_distance_m:.2f}m, "
            f"ttc_stop={self.ttc_stop_sec:.2f}s, "
            f"ttc_slow={self.ttc_slow_sec:.2f}s"
        )

    # ============================================================
    # Callbacks
    # ============================================================
    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().warn(f"Depth cv_bridge failed: {e}")
            return

        self.latest_depth = depth
        self.latest_depth_time = time.time()

    def yolo_callback(self, msg):
        now = time.time()

        if self.latest_depth is None:
            return

        if now - self.latest_depth_time > self.depth_timeout_sec:
            self.get_logger().warn("Depth image is too old. Ignore TTC update.")
            return

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Failed to parse YOLO detections JSON")
            return

        detections = data.get("detections", [])

        best_candidate = None

        for det in detections:
            class_name = str(det.get("class_name", "")).lower()
            confidence = float(det.get("confidence", 0.0))

            if class_name != self.rover_label:
                continue

            if confidence < self.conf_threshold:
                continue

            bbox = det.get("bbox", {})
            depth_m = self.get_depth_percentile_in_bbox(bbox)

            if depth_m is None:
                continue

            if best_candidate is None or depth_m < best_candidate["depth_m"]:
                best_candidate = {
                    "confidence": confidence,
                    "bbox": bbox,
                    "depth_m": depth_m,
                }

        if best_candidate is None:
            return

        self.last_detection_time = now

        depth_m = best_candidate["depth_m"]
        confidence = best_candidate["confidence"]

        closing_speed = self.update_closing_speed(depth_m, now)
        ttc = self.compute_ttc(depth_m, closing_speed)

        mode, reason = self.decide_mode(depth_m, closing_speed, ttc, now)

        if mode == "stop":
            self.stop_until = now + self.stop_hold_sec
        elif mode == "slow":
            self.slow_until = now + self.slow_hold_sec

        self.last_info = {
            "mode": mode,
            "reason": reason,
            "depth_m": depth_m,
            "closing_speed_mps": closing_speed,
            "ttc_sec": None if math.isinf(ttc) else ttc,
            "confidence": confidence,
        }

        self.get_logger().info(
            f"other_rover | depth={depth_m:.3f}m | "
            f"closing={closing_speed:.3f}m/s | "
            f"ttc={'inf' if math.isinf(ttc) else f'{ttc:.2f}s'} | "
            f"mode={mode} | reason={reason}"
        )

    # ============================================================
    # Depth / TTC
    # ============================================================
    def get_depth_percentile_in_bbox(self, bbox):
        if self.latest_depth is None:
            return None

        depth = self.latest_depth
        h, w = depth.shape[:2]

        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 2.0 or bh <= 2.0:
            return None

        mx = bw * self.bbox_margin_ratio
        my = bh * self.bbox_margin_ratio

        x1 = int(max(0, min(w - 1, x1 + mx)))
        y1 = int(max(0, min(h - 1, y1 + my)))
        x2 = int(max(0, min(w, x2 - mx)))
        y2 = int(max(0, min(h, y2 - my)))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = depth[y1:y2, x1:x2]

        valid = roi[np.isfinite(roi)]
        valid = valid[valid > 0.0]

        if valid.size == 0:
            return None

        depth_m = float(np.percentile(valid, self.depth_percentile))

        return depth_m

    def update_closing_speed(self, depth_m, now):
        if self.prev_rover_depth is None or self.prev_rover_time is None:
            self.prev_rover_depth = depth_m
            self.prev_rover_time = now
            self.filtered_closing_speed = 0.0
            return 0.0

        dt = now - self.prev_rover_time

        if dt <= 1e-3:
            return self.filtered_closing_speed

        raw_closing_speed = (self.prev_rover_depth - depth_m) / dt

        # 멀어지는 경우는 closing speed 0으로 처리
        raw_closing_speed = max(0.0, raw_closing_speed)

        alpha = np.clip(self.closing_speed_alpha, 0.0, 1.0)

        self.filtered_closing_speed = (
            alpha * raw_closing_speed
            + (1.0 - alpha) * self.filtered_closing_speed
        )

        self.prev_rover_depth = depth_m
        self.prev_rover_time = now

        return self.filtered_closing_speed

    def compute_ttc(self, depth_m, closing_speed):
        if closing_speed < self.min_closing_speed_mps:
            return float("inf")

        return depth_m / closing_speed

    def decide_mode(self, depth_m, closing_speed, ttc, now):
        if depth_m <= self.emergency_distance_m:
            return "stop", "emergency_distance"

        if depth_m <= self.stop_distance_m:
            return "stop", "stop_distance"

        if not math.isinf(ttc) and ttc <= self.ttc_stop_sec:
            return "stop", "ttc_stop"

        if depth_m <= self.slow_distance_m:
            return "slow", "slow_distance"

        if not math.isinf(ttc) and ttc <= self.ttc_slow_sec:
            return "slow", "ttc_slow"

        return "normal", "safe"

    # ============================================================
    # Timer publish
    # ============================================================
    def timer_callback(self):
        now = time.time()

        # detection이 사라지면 normal 복귀
        if now - self.last_detection_time > self.detection_timeout_sec:
            self.prev_rover_depth = None
            self.prev_rover_time = None
            self.filtered_closing_speed = 0.0

            if now >= self.stop_until and now >= self.slow_until:
                self.publish_mode("normal", reason="detection_timeout")
            return

        # hold 우선순위
        if now < self.stop_until:
            self.publish_mode("stop", reason="stop_hold")
            return

        if now < self.slow_until:
            self.publish_mode("slow", reason="slow_hold")
            return

        mode = self.last_info.get("mode", "normal")
        reason = self.last_info.get("reason", "last_info")

        self.publish_mode(mode, reason=reason)

    def publish_mode(self, mode, reason=""):
        msg = String()
        msg.data = mode
        self.drive_mode_pub.publish(msg)

        info = dict(self.last_info)
        info["published_mode"] = mode
        info["published_reason"] = reason
        info["stamp"] = time.time()

        info_msg = String()
        info_msg.data = json.dumps(info)
        self.ttc_info_pub.publish(info_msg)

        if self.last_mode != mode:
            self.get_logger().warn(f"Safety mode changed: {self.last_mode} -> {mode}")
            self.last_mode = mode


def main(args=None):
    rclpy.init(args=args)

    node = RoverTTCSafetyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_mode("stop", reason="node_shutdown")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()