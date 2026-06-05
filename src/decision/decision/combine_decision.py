#!/usr/bin/env python3

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CombinedRouteDecisionNode(Node):
    """
    통합 판단 노드.

    자동 주행용 구조:
    1. 외부 /mission_mode_cmd 사용 안 함.
    2. 시작 상태는 normal.
    3. normal 상태에서는 box 감지 시 현재 route_select의 반대 차선으로 회피.
    4. left_sign 감지 시 left_lane.
    5. right_sign 감지 시 right_lane.
    6. slow_sign 감지 시 mission_mode를 ttc로 전환.
    7. ttc 상태에서는 box 회피를 하지 않음.
       - box는 ACC-TTC 노드가 depth 기반 추종 대상으로 사용.
    """

    def __init__(self):
        super().__init__("combined_route_decision_node")

        # ==============================
        # Topics
        # ==============================
        self.declare_parameter("yolo_detection_topic", "/yolo/left_detections")
        self.declare_parameter("route_select_topic", "/route_select")
        self.declare_parameter("mission_mode_topic", "/mission_mode")

        # ==============================
        # Route
        # ==============================
        self.declare_parameter("default_route", "center")

        # ==============================
        # sign 판단 조건
        # left_sign -> left_lane
        # right_sign -> right_lane
        # ==============================
        self.declare_parameter("sign_conf_threshold", 0.70)
        self.declare_parameter("sign_min_bbox_area", 800.0)
        self.declare_parameter("sign_confirm_count", 3)
        self.declare_parameter("sign_hold_time_sec", 2.0)

        # ==============================
        # slow_sign -> TTC trigger
        # ==============================
        self.declare_parameter("slow_sign_conf_threshold", 0.70)
        self.declare_parameter("slow_sign_min_bbox_area", 800.0)
        self.declare_parameter("slow_sign_confirm_count", 3)
        self.declare_parameter("slow_sign_hold_time_sec", 2.0)

        # slow_sign을 한번 보면 TTC 상태로 계속 유지할지
        # 자동 주행에서는 보통 True 추천
        self.declare_parameter("ttc_latch", True)

        # ==============================
        # box 회피 판단 조건
        # normal 상태에서만 사용
        # ==============================
        self.declare_parameter("box_conf_threshold", 0.45)
        self.declare_parameter("box_center_min", 0.20)
        self.declare_parameter("box_center_max", 0.80)
        self.declare_parameter("box_height_threshold", 0.25)
        self.declare_parameter("box_required_frames", 3)

        # box가 사라진 뒤 몇 초 후 원래 route로 복귀할지
        self.declare_parameter("avoidance_return_delay", 5.0)

        # 회피 들어가면 최소 몇 초 유지할지
        self.declare_parameter("avoidance_min_hold_time", 2.0)

        # ==============================
        # Load params
        # ==============================
        self.yolo_detection_topic = self.get_parameter("yolo_detection_topic").value
        self.route_select_topic = self.get_parameter("route_select_topic").value
        self.mission_mode_topic = self.get_parameter("mission_mode_topic").value

        self.default_route = self.get_parameter("default_route").value.strip().lower()

        if self.default_route not in ["left_lane", "right_lane", "center"]:
            self.get_logger().warn(
                f"Unknown default_route={self.default_route}. Fallback to right_lane."
            )
            self.default_route = "right_lane"

        self.sign_conf_threshold = float(
            self.get_parameter("sign_conf_threshold").value
        )
        self.sign_min_bbox_area = float(
            self.get_parameter("sign_min_bbox_area").value
        )
        self.sign_confirm_count = int(
            self.get_parameter("sign_confirm_count").value
        )
        self.sign_hold_time_sec = float(
            self.get_parameter("sign_hold_time_sec").value
        )

        self.slow_sign_conf_threshold = float(
            self.get_parameter("slow_sign_conf_threshold").value
        )
        self.slow_sign_min_bbox_area = float(
            self.get_parameter("slow_sign_min_bbox_area").value
        )
        self.slow_sign_confirm_count = int(
            self.get_parameter("slow_sign_confirm_count").value
        )
        self.slow_sign_hold_time_sec = float(
            self.get_parameter("slow_sign_hold_time_sec").value
        )
        self.ttc_latch = bool(
            self.get_parameter("ttc_latch").value
        )

        self.box_conf_threshold = float(
            self.get_parameter("box_conf_threshold").value
        )
        self.box_center_min = float(
            self.get_parameter("box_center_min").value
        )
        self.box_center_max = float(
            self.get_parameter("box_center_max").value
        )
        self.box_height_threshold = float(
            self.get_parameter("box_height_threshold").value
        )
        self.box_required_frames = int(
            self.get_parameter("box_required_frames").value
        )

        self.avoidance_return_delay = float(
            self.get_parameter("avoidance_return_delay").value
        )
        self.avoidance_min_hold_time = float(
            self.get_parameter("avoidance_min_hold_time").value
        )

        # ==============================
        # State
        # ==============================
        self.mission_mode = "normal"

        # sign이 정한 기본 목표 route
        self.sign_route = self.default_route

        # 실제 /route_select로 나가는 최종 route
        self.current_route = self.default_route

        self.left_sign_count = 0
        self.right_sign_count = 0
        self.slow_sign_count = 0
        self.box_count = 0

        self.last_sign_change_time = 0.0
        self.last_slow_sign_trigger_time = 0.0
        self.last_box_time = 0.0

        self.in_avoidance = False
        self.avoidance_start_time = 0.0
        self.avoid_route = None

        # ==============================
        # ROS interface
        # ==============================
        self.yolo_sub = self.create_subscription(
            String,
            self.yolo_detection_topic,
            self.detection_callback,
            10
        )

        self.route_pub = self.create_publisher(
            String,
            self.route_select_topic,
            10
        )

        self.mission_mode_pub = self.create_publisher(
            String,
            self.mission_mode_topic,
            10
        )

        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info("Combined route decision node started")
        self.get_logger().info(f"Subscribing YOLO detections : {self.yolo_detection_topic}")
        self.get_logger().info(f"Publishing route select     : {self.route_select_topic}")
        self.get_logger().info(f"Publishing mission mode     : {self.mission_mode_topic}")
        self.get_logger().info(f"Default route               : {self.default_route}")
        self.get_logger().info("Auto mode: normal -> box avoidance, slow_sign -> ttc")

        self.publish_route(self.current_route, force=True)
        self.publish_mission_mode(force=True)

    # ============================================================
    # YOLO callback
    # ============================================================
    def detection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            image_width = float(data.get("image_width", 640))
            image_height = float(data.get("image_height", 360))
            detections = data.get("detections", [])
        except Exception as e:
            self.get_logger().warn(f"Failed to parse YOLO JSON: {e}")
            return

        best_left_sign = None
        best_right_sign = None
        box_detected = False
        slow_sign_found = False

        for det in detections:
            class_name = det.get("class_name", "")
            confidence = float(det.get("confidence", 0.0))

            bbox = det.get("bbox", {})
            center = det.get("center", {})
            size = det.get("size", {})

            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            y2 = float(bbox.get("y2", 0.0))

            center_x_px = float(center.get("x", (x1 + x2) / 2.0))
            box_w_px = float(size.get("w", max(0.0, x2 - x1)))
            box_h_px = float(size.get("h", max(0.0, y2 - y1)))

            area = box_w_px * box_h_px

            # ==============================
            # 1. left/right sign detection
            # ==============================
            if (
                class_name in ["left_sign", "right_sign"]
                and confidence >= self.sign_conf_threshold
                and area >= self.sign_min_bbox_area
            ):
                item = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "area": area,
                }

                if class_name == "left_sign":
                    if best_left_sign is None or confidence > best_left_sign["confidence"]:
                        best_left_sign = item

                elif class_name == "right_sign":
                    if best_right_sign is None or confidence > best_right_sign["confidence"]:
                        best_right_sign = item

            # ==============================
            # 2. slow_sign detection
            # slow_sign -> mission_mode = ttc
            # ==============================
            if (
                class_name == "slow_sign"
                and confidence >= self.slow_sign_conf_threshold
                and area >= self.slow_sign_min_bbox_area
            ):
                slow_sign_found = True

            # ==============================
            # 3. box detection
            # normal 상태에서만 회피 판단에 사용
            # ttc 상태에서는 ACC-TTC 노드가 box를 추종 대상으로 사용
            # ==============================
            if class_name == "box":
                center_x = center_x_px / max(image_width, 1.0)
                height_ratio = box_h_px / max(image_height, 1.0)

                is_front_box = (
                    confidence >= self.box_conf_threshold
                    and self.box_center_min <= center_x <= self.box_center_max
                    and height_ratio >= self.box_height_threshold
                )

                if is_front_box:
                    box_detected = True

                    if self.mission_mode == "normal":
                        self.get_logger().warn(
                            f"box detected for avoidance | conf={confidence:.2f}, "
                            f"center_x={center_x:.2f}, "
                            f"height_ratio={height_ratio:.2f}",
                            throttle_duration_sec=0.5
                        )
                    elif self.mission_mode == "ttc":
                        self.get_logger().info(
                            "box detected in TTC mode -> route avoidance disabled",
                            throttle_duration_sec=1.0
                        )

        # 순서 중요:
        # slow_sign이 보이면 먼저 TTC로 전환
        self.update_slow_sign_mission(slow_sign_found)

        # 표지판 route 변경은 유지
        # 단, 회피 중이면 바로 route에 반영하지 않고 sign_route만 갱신
        self.update_sign_route(best_left_sign, best_right_sign)

        # normal일 때만 box 회피
        self.update_box_avoidance(box_detected)

    # ============================================================
    # Mission logic
    # ============================================================
    def update_slow_sign_mission(self, slow_sign_found):
        now = time.time()

        if self.mission_mode == "ttc" and self.ttc_latch:
            return

        if slow_sign_found:
            self.slow_sign_count += 1
        else:
            self.slow_sign_count = 0
            return

        if self.slow_sign_count < self.slow_sign_confirm_count:
            return

        if now - self.last_slow_sign_trigger_time < self.slow_sign_hold_time_sec:
            return

        self.last_slow_sign_trigger_time = now

        if self.mission_mode != "ttc":
            old_mode = self.mission_mode
            self.mission_mode = "ttc"

            # TTC 진입 시 box 회피 상태 해제
            self.reset_avoidance_state()

            self.get_logger().warn(
                f"slow_sign detected -> mission_mode changed: {old_mode} -> ttc"
            )

            self.publish_mission_mode(force=True)

    def publish_mission_mode(self, force=False):
        msg = String()
        msg.data = self.mission_mode
        self.mission_mode_pub.publish(msg)

    # ============================================================
    # Sign route logic
    # ============================================================
    def update_sign_route(self, best_left_sign, best_right_sign):
        detected_sign_route = None

        if best_left_sign is not None and best_right_sign is not None:
            if best_left_sign["confidence"] >= best_right_sign["confidence"]:
                detected_sign_route = "left_lane"
            else:
                detected_sign_route = "right_lane"

        elif best_left_sign is not None:
            detected_sign_route = "left_lane"

        elif best_right_sign is not None:
            detected_sign_route = "right_lane"

        else:
            self.left_sign_count = 0
            self.right_sign_count = 0
            return

        if detected_sign_route == "left_lane":
            self.left_sign_count += 1
            self.right_sign_count = 0

        elif detected_sign_route == "right_lane":
            self.right_sign_count += 1
            self.left_sign_count = 0

        now = time.time()

        if now - self.last_sign_change_time < self.sign_hold_time_sec:
            return

        if self.left_sign_count >= self.sign_confirm_count:
            self.set_sign_route("left_lane")

        elif self.right_sign_count >= self.sign_confirm_count:
            self.set_sign_route("right_lane")

    def set_sign_route(self, route):
        if route not in ["left_lane", "right_lane", "center"]:
            return

        if route != self.sign_route:
            self.get_logger().info(f"Sign route changed: {self.sign_route} -> {route}")

        self.sign_route = route
        self.last_sign_change_time = time.time()

        # 회피 중이 아닐 때만 바로 최종 route에 반영
        if not self.in_avoidance:
            self.publish_route(self.sign_route)

    # ============================================================
    # Box avoidance logic
    # ============================================================
    def update_box_avoidance(self, box_detected):
        now = time.time()

        # 핵심:
        # TTC 모드에서는 box 회피를 하지 않음.
        # box는 ACC-TTC 노드가 depth 기반 추종 대상으로 사용.
        if self.mission_mode != "normal":
            return

        if box_detected:
            self.box_count += 1
            self.last_box_time = now
        else:
            self.box_count = 0

        if self.box_count >= self.box_required_frames:
            if not self.in_avoidance:
                self.in_avoidance = True
                self.avoidance_start_time = now

                # 현재 route_select 기준 반대 방향으로 회피
                self.avoid_route = self.get_opposite_route(self.current_route)

                self.get_logger().warn(
                    f"Avoidance mode ON: "
                    f"current_route={self.current_route} -> avoid_route={self.avoid_route}"
                )

            self.publish_route(self.avoid_route)

    def get_opposite_route(self, route):
        if route == "left_lane":
            return "right_lane"

        if route == "right_lane":
            return "left_lane"

        # center 상태에서 box를 만나면 default_route의 반대 방향으로 회피
        if self.default_route == "left_lane":
            return "right_lane"

        if self.default_route == "right_lane":
            return "left_lane"

        return "right_lane"

    def reset_avoidance_state(self):
        self.box_count = 0
        self.last_box_time = 0.0
        self.avoidance_start_time = 0.0
        self.in_avoidance = False
        self.avoid_route = None

    # ============================================================
    # Timer
    # ============================================================
    def timer_callback(self):
        now = time.time()

        self.publish_mission_mode(force=True)

        if self.in_avoidance:
            # TTC로 바뀌면 회피 즉시 종료
            if self.mission_mode != "normal":
                self.reset_avoidance_state()
                self.publish_route(self.sign_route, force=True)
                return

            hold_elapsed = now - self.avoidance_start_time
            no_box_elapsed = now - self.last_box_time

            if self.avoid_route is None:
                self.avoid_route = self.get_opposite_route(self.current_route)

            # 회피 진입 후 최소 유지 시간 동안은 회피 route 유지
            if hold_elapsed < self.avoidance_min_hold_time:
                self.publish_route(self.avoid_route, force=True)
                return

            # box가 return_delay 동안 안 보이면 sign_route로 복귀
            if no_box_elapsed >= self.avoidance_return_delay:
                self.in_avoidance = False

                self.get_logger().warn(
                    f"Avoidance mode OFF: return to sign_route={self.sign_route}"
                )

                self.avoid_route = None
                self.publish_route(self.sign_route, force=True)
                return

            # 아직 box가 최근에 보였으면 계속 회피 route 유지
            self.publish_route(self.avoid_route, force=True)
            return

        # 평상시에는 sign_route 유지
        self.publish_route(self.sign_route, force=True)

    # ============================================================
    # Publishers
    # ============================================================
    def publish_route(self, route, force=False):
        if route not in ["left_lane", "right_lane", "center"]:
            self.get_logger().warn(f"Invalid route command: {route}")
            return

        if not force and route == self.current_route:
            return

        msg = String()
        msg.data = route
        self.route_pub.publish(msg)

        if route != self.current_route:
            self.get_logger().warn(
                f"Route command changed: {self.current_route} -> {route}"
            )

        self.current_route = route


def main(args=None):
    rclpy.init(args=args)
    node = CombinedRouteDecisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()