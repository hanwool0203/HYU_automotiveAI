#!/usr/bin/env python3

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class YoloRouteDecisionNode(Node):
    def __init__(self):
        super().__init__("yolo_route_decision_node")

        # ==============================
        # Parameters
        # ==============================
        self.declare_parameter("yolo_detection_topic", "/yolo/left_detections")

        # LaneMulticlassONNXNode가 구독하는 토픽
        # 이 토픽에 left / center / right를 보내면 차선 노드가 target을 바꿈
        self.declare_parameter("route_select_topic", "/route_select")

        self.declare_parameter("min_confidence", 0.50)
        self.declare_parameter("min_bbox_area", 800.0)

        # 몇 프레임 연속 감지되어야 route 변경할지
        self.declare_parameter("confirm_count", 2)

        # route 변경 후 최소 유지 시간
        self.declare_parameter("route_hold_time_sec", 2.0)

        # 기본 주행 차선
        self.declare_parameter("default_route", "right_lane")

        self.yolo_detection_topic = self.get_parameter("yolo_detection_topic").value
        self.route_select_topic = self.get_parameter("route_select_topic").value

        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.min_bbox_area = float(self.get_parameter("min_bbox_area").value)
        self.confirm_count = int(self.get_parameter("confirm_count").value)
        self.route_hold_time_sec = float(
            self.get_parameter("route_hold_time_sec").value
        )

        self.current_route = (
            self.get_parameter("default_route").value.strip().lower()
        )

        if self.current_route not in ["left_lane", "center", "right_lane"]:
            self.get_logger().warn(
                f"Unknown default_route={self.current_route}. Fallback to center."
            )
            self.current_route = "center"

        # ==============================
        # State
        # ==============================
        self.left_count = 0
        self.right_count = 0
        self.last_route_change_time = 0.0

        # ==============================
        # ROS Interface
        # ==============================
        self.det_sub = self.create_subscription(
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

        # 현재 route를 계속 publish
        self.timer = self.create_timer(0.2, self.publish_route)

        self.get_logger().info("YOLO Route Decision Node started")
        self.get_logger().info(f"Subscribing YOLO detections : {self.yolo_detection_topic}")
        self.get_logger().info(f"Publishing route select     : {self.route_select_topic}")
        self.get_logger().info(f"Default route               : {self.current_route}")
        self.get_logger().info(
            f"min_confidence={self.min_confidence}, "
            f"min_bbox_area={self.min_bbox_area}, "
            f"confirm_count={self.confirm_count}, "
            f"route_hold_time_sec={self.route_hold_time_sec}"
        )

    def detection_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid JSON from YOLO detection topic")
            return

        detections = data.get("detections", [])

        best_left = None
        best_right = None

        for det in detections:
            class_name = det.get("class_name", "")
            confidence = float(det.get("confidence", 0.0))

            size = det.get("size", {})
            w = float(size.get("w", 0.0))
            h = float(size.get("h", 0.0))
            area = w * h

            if confidence < self.min_confidence:
                continue

            if area < self.min_bbox_area:
                continue

            item = {
                "class_name": class_name,
                "confidence": confidence,
                "area": area,
            }

            if class_name == "left_sign":
                if best_left is None or confidence > best_left["confidence"]:
                    best_left = item

            elif class_name == "right_sign":
                if best_right is None or confidence > best_right["confidence"]:
                    best_right = item

        self.update_route_decision(best_left, best_right)

    def update_route_decision(self, best_left, best_right):
        detected_route = None

        # left_sign, right_sign이 동시에 잡히면 confidence 높은 쪽 선택
        if best_left is not None and best_right is not None:
            if best_left["confidence"] >= best_right["confidence"]:
                detected_route = "left"
            else:
                detected_route = "right"

        elif best_left is not None:
            detected_route = "left"

        elif best_right is not None:
            detected_route = "right"

        else:
            self.left_count = 0
            self.right_count = 0
            return

        if detected_route == "left":
            self.left_count += 1
            self.right_count = 0

        elif detected_route == "right":
            self.right_count += 1
            self.left_count = 0

        now = time.time()

        # route가 너무 자주 바뀌는 것 방지
        if now - self.last_route_change_time < self.route_hold_time_sec:
            return

        if self.left_count >= self.confirm_count:
            self.set_route("left_lane")

        elif self.right_count >= self.confirm_count:
            self.set_route("right_lane")

    def set_route(self, route):
        if route not in ["left_lane", "center", "right_lane"]:
            return

        if route != self.current_route:
            self.get_logger().info(
                f"Route changed: {self.current_route} -> {route}"
            )

        self.current_route = route
        self.last_route_change_time = time.time()
        self.publish_route()

    def publish_route(self):
        msg = String()
        msg.data = self.current_route
        self.route_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloRouteDecisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()