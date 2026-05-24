#!/usr/bin/env python3

import json
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class SignDecisionNode(Node):
    def __init__(self):
        super().__init__("sign_decision_node")

        # ============================================================
        # Topics
        # ============================================================
        self.declare_parameter("yolo_detections_topic", "/yolo/left_detections")
        self.declare_parameter("image_topic", "/left/image_raw")
        self.declare_parameter("drive_mode_topic", "/drive_mode")

        # ============================================================
        # Sign parameters
        # ============================================================
        self.declare_parameter("stop_duration_sec", 2.0)
        self.declare_parameter("slow_duration_sec", 5.0)
        self.declare_parameter("decision_cooldown_sec", 1.0)

        self.declare_parameter("stop_label", "stop_sign")
        self.declare_parameter("slow_label", "slow_sign")
        self.declare_parameter("traffic_light_label", "traffic_light")
        self.declare_parameter("rover_label", "other_rover")

        self.declare_parameter("stop_conf_threshold", 0.50)
        self.declare_parameter("stop_min_bbox_area", 1700.0)
        self.declare_parameter("slow_conf_threshold", 0.50)
        self.declare_parameter("slow_min_bbox_area", 1000.0)
        self.declare_parameter("traffic_conf_threshold", 0.50)
        self.declare_parameter("rover_conf_threshold", 0.50)

        # traffic_light unknown stop은 N프레임 연속 가까울 때만 적용
        self.declare_parameter("traffic_near_confirm_frames", 3)
        self.declare_parameter("traffic_min_bbox_area", 500.0)

        self.declare_parameter("rover_stop_min_bbox_area", 12000.0)
        self.declare_parameter("rover_stop_duration_sec", 1.5)

        # 최신 image가 너무 오래됐으면 무시
        self.declare_parameter("image_timeout_sec", 0.3)

        # 너무 작은 박스 무시
        self.declare_parameter("min_bbox_area", 0.0)

        # bbox 가장자리 제거 비율
        self.declare_parameter("bbox_margin_ratio", 0.10)

        # stop_sign, slow_sign은 한 번만 처리
        self.declare_parameter("one_shot_stop", True)
        self.declare_parameter("one_shot_slow", True)
        self.declare_parameter("one_shot_traffic_light", True)
        self.declare_parameter("one_shot_rover", False)

        # 신호등 색 판단 기준
        self.declare_parameter("green_ratio_threshold", 0.005)

        self.yolo_detections_topic = self.get_parameter("yolo_detections_topic").value
        self.image_topic = self.get_parameter("image_topic").value
        self.drive_mode_topic = self.get_parameter("drive_mode_topic").value

        self.stop_duration_sec = float(self.get_parameter("stop_duration_sec").value)
        self.slow_duration_sec = float(self.get_parameter("slow_duration_sec").value)
        self.decision_cooldown_sec = float(
            self.get_parameter("decision_cooldown_sec").value
        )

        self.stop_label = self.get_parameter("stop_label").value.lower()
        self.slow_label = self.get_parameter("slow_label").value.lower()
        self.traffic_light_label = self.get_parameter("traffic_light_label").value.lower()
        self.rover_label = self.get_parameter("rover_label").value.lower()

        self.stop_conf_threshold = float(
            self.get_parameter("stop_conf_threshold").value
        )
        self.stop_min_bbox_area = float(
            self.get_parameter("stop_min_bbox_area").value
        )
        self.slow_conf_threshold = float(
            self.get_parameter("slow_conf_threshold").value
        )
        self.slow_min_bbox_area = float(
            self.get_parameter("slow_min_bbox_area").value
        )
        self.traffic_conf_threshold = float(
            self.get_parameter("traffic_conf_threshold").value
        )
        self.rover_conf_threshold = float(
            self.get_parameter("rover_conf_threshold").value
        )
        self.rover_stop_min_bbox_area = float(
            self.get_parameter("rover_stop_min_bbox_area").value
        )

        self.rover_stop_duration_sec = float(
            self.get_parameter("rover_stop_duration_sec").value
        )
        self.traffic_near_confirm_frames = int(
            self.get_parameter("traffic_near_confirm_frames").value
        )
        self.traffic_min_bbox_area = float(
            self.get_parameter("traffic_min_bbox_area").value
        )
        self.image_timeout_sec = float(self.get_parameter("image_timeout_sec").value)

        self.min_bbox_area = float(self.get_parameter("min_bbox_area").value)
        self.bbox_margin_ratio = float(
            self.get_parameter("bbox_margin_ratio").value
        )

        self.one_shot_stop = bool(self.get_parameter("one_shot_stop").value)
        self.one_shot_slow = bool(self.get_parameter("one_shot_slow").value)
        self.one_shot_rover = bool(self.get_parameter("one_shot_rover").value)
        self.one_shot_traffic_light = bool(
            self.get_parameter("one_shot_traffic_light").value
        )
        self.green_ratio_threshold = float(
            self.get_parameter("green_ratio_threshold").value
        )

        # ============================================================
        # State
        # ============================================================
        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_image_time = 0.0

        self.current_drive_mode = "normal"

        self.last_stop_decision_time = 0.0
        self.last_slow_decision_time = 0.0
        self.last_traffic_decision_time = 0.0

        self.stop_until = 0.0
        self.slow_until = 0.0

        self.handled_stop_sign = False
        self.handled_slow_sign = False
        self.handled_traffic_light = False
        self.handled_rover = False
        self.last_rover_decision_time = 0.0
        self.rover_stop_until = 0.0

        # traffic_light는 one-shot 아님
        # 빨간불이면 계속 정지 상태 유지, 초록불이면 해제
        self.traffic_light_stop_active = False

        # unknown traffic_light가 몇 프레임 연속 잡혔는지
        self.traffic_near_confirm_count = 0

        # ============================================================
        # QoS
        # stereo_depth_node의 image/depth publisher가 BEST_EFFORT이므로 맞춤
        # ============================================================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ============================================================
        # ROS pubs/subs
        # ============================================================
        self.det_sub = self.create_subscription(
            String,
            self.yolo_detections_topic,
            self.detections_callback,
            10,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            sensor_qos,
        )

        self.drive_mode_pub = self.create_publisher(
            String,
            self.drive_mode_topic,
            10,
        )

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.publish_drive_mode("normal")

        self.get_logger().info("Sign Decision Node Started")
        self.get_logger().info(f"YOLO detections topic: {self.yolo_detections_topic}")
        self.get_logger().info(f"Image topic          : {self.image_topic}")
        self.get_logger().info(f"Drive mode topic     : {self.drive_mode_topic}")
        self.get_logger().info(f"Traffic light rule: GREEN confirmed -> release, otherwise near traffic_light -> stop")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image cv_bridge failed: {e}")
            return

        self.latest_image = image
        self.latest_image_time = time.time()

    def crop_bbox(self, arr, bbox):
        h, w = arr.shape[:2]

        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))

        bw = x2 - x1
        bh = y2 - y1

        mx = bw * self.bbox_margin_ratio
        my = bh * self.bbox_margin_ratio

        x1 = int(max(0, min(w - 1, x1 + mx)))
        y1 = int(max(0, min(h - 1, y1 + my)))
        x2 = int(max(0, min(w, x2 - mx)))
        y2 = int(max(0, min(h, y2 - my)))

        if x2 <= x1 or y2 <= y1:
            return None

        return arr[y1:y2, x1:x2]

    def classify_traffic_light_color(self, bbox):
        if self.latest_image is None:
            return "unknown", 0.0, 0.0

        now = time.time()
        if now - self.latest_image_time > self.image_timeout_sec:
            return "unknown", 0.0, 0.0

        roi = self.crop_bbox(self.latest_image, bbox)

        if roi is None or roi.size == 0:
            return "unknown", 0.0, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # green만 판단
        lower_green = np.array([35, 60, 60], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)

        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_count = cv2.countNonZero(green_mask)

        area = max(1, roi.shape[0] * roi.shape[1])
        green_ratio = green_count / area

        # red_ratio는 로그 호환용으로 0.0 반환
        red_ratio = 0.0

        if green_ratio >= self.green_ratio_threshold:
            return "green", red_ratio, green_ratio

        return "unknown", red_ratio, green_ratio

    def detections_callback(self, msg):
        now = time.time()

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Failed to parse YOLO detections JSON")
            return

        detections = data.get("detections", [])

        if not detections:
            return

        best_stop = None
        best_slow = None
        best_traffic = None
        best_rover = None

        for det in detections:
            class_name = str(det.get("class_name", "")).lower()
            confidence = float(det.get("confidence", 0.0))
            bbox = det.get("bbox", {})

            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            y2 = float(bbox.get("y2", 0.0))
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            if area < self.min_bbox_area:
                continue

            if class_name not in [
                self.stop_label,
                self.slow_label,
                self.traffic_light_label,
                self.rover_label,
            ]:
                continue
            # ============================================================
            # traffic_light는 stop/slow와 다르게 먼저 색상 판단
            # red는 사용하지 않고, green만 출발 신호로 사용
            # green이 아니면 unknown으로 보고, 가까울 때만 stop 후보로 사용
            # ============================================================
            if class_name == self.traffic_light_label:
                if self.one_shot_traffic_light and self.handled_traffic_light:
                    continue
                if confidence < self.traffic_conf_threshold:
                    continue
                if area < self.traffic_min_bbox_area:
                    self.get_logger().info(
                        f"traffic_light ignored: conf={confidence:.3f} ,area={area:.1f} < {self.traffic_min_bbox_area:.1f}"
                    )
                    continue

                color, red_ratio, green_ratio = self.classify_traffic_light_color(bbox)

                # green이 아니면 unknown으로 보고 stop 후보로 사용
                # 실제 출발은 아래 traffic 처리부에서 green_confirm_frames로 결정
                if color == "green":
                    if best_traffic is None or confidence > best_traffic["confidence"]:
                        best_traffic = {
                            "confidence": confidence,
                            "area": area,
                            "color": "green",
                            "red_ratio": red_ratio,
                            "green_ratio": green_ratio,
                        }
                    continue

                if best_traffic is None or confidence > best_traffic["confidence"]:
                    best_traffic = {
                        "confidence": confidence,
                        "area": area,
                        "color": "unknown",
                        "red_ratio": red_ratio,
                        "green_ratio": green_ratio,
                    }

                continue

            # ============================================================
            # other_rover:
            # depth 없이 bbox area 기준으로 가까움 판단
            # ============================================================
            if class_name == self.rover_label:
                if confidence < self.rover_conf_threshold:
                    continue

                if area < self.rover_stop_min_bbox_area:
                    self.get_logger().info(
                        f"other_rover ignored: "
                        f"conf={confidence:.3f}, "
                        f"area={area:.1f} < {self.rover_stop_min_bbox_area:.1f}"
                    )
                    continue

                if best_rover is None or confidence > best_rover["confidence"]:
                    best_rover = {
                        "confidence": confidence,
                        "area": area,
                    }

                continue

            if class_name == self.stop_label:
                if confidence < self.stop_conf_threshold:
                    continue

                if area < self.stop_min_bbox_area:
                    self.get_logger().info(
                        f"stop_sign ignored: "
                        f"conf={confidence:.3f}, "
                        f"area={area:.1f} < {self.stop_min_bbox_area:.1f}"
                    )
                    continue

                if best_stop is None or confidence > best_stop["confidence"]:
                    best_stop = {
                        "confidence": confidence,
                        "area": area,
                    }

                continue

            if class_name == self.slow_label:
                if confidence < self.slow_conf_threshold:
                    continue

                if area < self.slow_min_bbox_area:
                    self.get_logger().info(
                        f"slow_sign ignored: "
                        f"conf={confidence:.3f}, "
                        f"area={area:.1f} < {self.slow_min_bbox_area:.1f}"
                    )
                    continue

                if best_slow is None or confidence > best_slow["confidence"]:
                    best_slow = {
                        "confidence": confidence,
                        "area": area,
                    }

                continue

        # ============================================================
        # 1. Traffic light 처리
        #    traffic_light는 one-shot 아님
        #    red   -> 계속 stop 유지
        #    green -> traffic stop 해제
        # ============================================================

        if best_traffic is not None:
            color = best_traffic["color"]

            self.get_logger().info(
                f"TRAFFIC_LIGHT | color={color}, "
                f"conf={best_traffic['confidence']:.3f}, "
                f"area={best_traffic['area']:.1f}, "
                f"green_ratio={best_traffic['green_ratio']:.3f}, "
            )

            # ============================================================
            # GREEN:
            # traffic light stop latch만 해제
            # 여기서 return하지 않음.
            # 그래야 같은 프레임의 stop_sign / slow_sign도 아래에서 처리됨.
            # ============================================================
            if color == "green":
                self.traffic_near_confirm_count = 0
                self.traffic_light_stop_active = False
                if self.one_shot_traffic_light:
                    self.handled_traffic_light = True
                self.get_logger().info("GREEN traffic light -> traffic stop released")

            else: #unknown
                self.traffic_near_confirm_count += 1

                if self.traffic_near_confirm_count < self.traffic_near_confirm_frames:
                    self.get_logger().info(
                        f"UNKNOWN traffic_light near buffering "
                        f"{self.traffic_near_confirm_count}/"
                        f"{self.traffic_near_confirm_frames} | "
                    )
                    return

                if now - self.last_traffic_decision_time >= self.decision_cooldown_sec:
                    self.last_traffic_decision_time = now
                    self.traffic_near_confirm_count = 0
                    self.traffic_light_stop_active = True
                    self.publish_drive_mode("stop")
                    self.get_logger().warn(
                        "Traffic light detected near but GREEN not confirmed "
                        f"for {self.traffic_near_confirm_frames} frames -> STOP"
                    )
                    return

        # ============================================================
        # 2. Other rover 처리
        #    bbox area가 충분히 크면 정지
        # ============================================================

        if best_rover is not None:
            if self.one_shot_rover and self.handled_rover:
                return

            if now - self.last_rover_decision_time >= self.decision_cooldown_sec:
                self.handled_rover = True
                self.last_rover_decision_time = now
                self.rover_stop_until = now + self.rover_stop_duration_sec

                self.publish_drive_mode("stop")

                self.get_logger().warn(
                    f"OTHER ROVER STOP triggered | "
                    f"conf={best_rover['confidence']:.3f}, "
                    f"area={best_rover['area']:.1f} >= "
                    f"{self.rover_stop_min_bbox_area:.1f} | "
                    f"stop for {self.rover_stop_duration_sec:.1f}s"
                )

            return

        # ============================================================
        # 3. Stop sign 처리
        #    stop_sign은 one-shot
        #    단, traffic_light_stop_active와 독립적이어야 함
        # ============================================================

        if best_stop is not None:
            if self.one_shot_stop and self.handled_stop_sign:
                # 이미 stop_sign은 처리했으므로 무시
                # 여기서 traffic_light는 이미 위에서 처리했기 때문에 return해도 됨
                return

            if now - self.last_stop_decision_time >= self.decision_cooldown_sec:
                self.handled_stop_sign = True
                self.last_stop_decision_time = now
                self.stop_until = now + self.stop_duration_sec

                self.publish_drive_mode("stop")

                self.get_logger().warn(
                    f"STOP SIGN triggered | "
                    f"conf={best_stop['confidence']:.3f}, "
                    f"area={best_stop['area']:.1f} | "
                    f"stop for {self.stop_duration_sec:.1f}s"
                )
            return

        # ============================================================
        # 4. Slow sign 처리
        # ============================================================

        if best_slow is not None:
            if self.one_shot_slow and self.handled_slow_sign:
                return

            if now - self.last_slow_decision_time >= self.decision_cooldown_sec:
                self.handled_slow_sign = True
                self.last_slow_decision_time = now
                self.slow_until = now + self.slow_duration_sec

                if not self.traffic_light_stop_active and now >= self.stop_until:
                    self.publish_drive_mode("slow")

                self.get_logger().info(
                    f"SLOW SIGN triggered | "
                    f"conf={best_slow['confidence']:.3f}, "
                    f"area={best_slow['area']:.1f} | "
                    f"slow for {self.slow_duration_sec:.1f}s"
                )

    def timer_callback(self):
        now = time.time()

        # traffic light stop latch이 최우선
        if self.traffic_light_stop_active:
            if self.current_drive_mode != "stop":
                self.publish_drive_mode("stop")
            return
        
        # other_rover bbox 기반 정지
        if now < self.rover_stop_until:
            if self.current_drive_mode != "stop":
                self.publish_drive_mode("stop")
            return

        # stop sign 2초 정지
        if now < self.stop_until:
            if self.current_drive_mode != "stop":
                self.publish_drive_mode("stop")
            return

        # slow sign 서행
        if now < self.slow_until:
            if self.current_drive_mode != "slow":
                self.publish_drive_mode("slow")
            return

        # 기본 주행
        if self.current_drive_mode != "normal":
            self.publish_drive_mode("normal")

    def publish_drive_mode(self, mode):
        self.current_drive_mode = mode

        msg = String()
        msg.data = mode
        self.drive_mode_pub.publish(msg)

        self.get_logger().info(f"Drive mode: {mode}")


def main(args=None):
    rclpy.init(args=args)

    node = SignDecisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_drive_mode("stop")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()