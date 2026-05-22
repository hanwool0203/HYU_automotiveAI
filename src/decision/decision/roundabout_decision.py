#!/usr/bin/env python3

import json
from statistics import mode
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Bool

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class RoundaboutDecisionNode(Node):
    def __init__(self):
        super().__init__("roundabout_decision_node")

        # ============================================================
        # Topics
        # ============================================================
        self.declare_parameter("yolo_detections_topic", "/yolo/left_detections")
        self.declare_parameter("target_left_topic", "/centerline/target_left")
        self.declare_parameter("target_right_topic", "/centerline/target_right")
        self.declare_parameter("route_select_topic", "/route_select")
        self.declare_parameter("ttc_enable_topic", "/ttc/enable")
        self.declare_parameter("ttc_enable_entry_count", 2)
        self.declare_parameter("drive_mode_topic", "/drive_mode")

        # ============================================================
        # YOLO labels
        # ============================================================
        self.declare_parameter("left_arrow_label", "left_arrow")
        self.declare_parameter("right_arrow_label", "right_arrow")
        self.declare_parameter("arrow_conf_threshold", 0.45)

        # ============================================================
        # Route rule
        # ============================================================
        self.declare_parameter("entry_route", "right")       # 진입 시 무조건 right
        self.declare_parameter("roundabout_route", "left")   # 반시계방향 회전
        self.declare_parameter("exit_route", "right")        # 출구 탈출
        self.declare_parameter("default_route", "right")
        self.declare_parameter("final_route", "center")

        # 좌회전이면 3번째 출구, 우회전이면 1번째 출구
        self.declare_parameter("left_arrow_first_exit_index", 3)
        self.declare_parameter("left_arrow_second_exit_index", 1)

        self.declare_parameter("right_arrow_first_exit_index", 1)
        self.declare_parameter("right_arrow_second_exit_index", 3)

        # 진입 직후 right를 유지할 시간
        self.declare_parameter("entry_hold_sec", 3.0)

        # 출구 탈출 시 right 유지 시간
        self.declare_parameter("exit_hold_sec", 5.0)

        # 회전교차로 탈출 후 유지할 route
        self.declare_parameter("after_exit_left_route", "left")
        self.declare_parameter("after_exit_right_route", "right")

        # ============================================================
        # Exit detection by target x gap
        # ============================================================
        self.declare_parameter("exit_gap_px", 180.0)
        self.declare_parameter("exit_gap_reset_px", 120.0)
        self.declare_parameter("exit_confirm_frames", 3)
        self.declare_parameter("exit_min_interval_sec", 2.0)
        self.declare_parameter("target_timeout_sec", 0.3)

        # gap 노이즈 제한
        self.declare_parameter("gap_max_px", 350.0)
        self.declare_parameter("gap_jump_max_px", 300.0)

        self.declare_parameter("enable_intersection_fix_after_exit", True)
        self.declare_parameter("intersection_trigger_delay_sec", 1.0)
        self.declare_parameter("intersection_trigger_gap_px", 80.0)
        self.declare_parameter("intersection_trigger_confirm_frames", 3)

        self.declare_parameter("intersection_fix_left_route", "left")
        self.declare_parameter("intersection_fix_right_route", "right")
        self.declare_parameter("intersection_fix_hold_sec", 5.0)
        self.declare_parameter("after_intersection_route", "right")

        # 회전교차로 진입은 1차/2차 모두 left-right target gap으로 판단
        self.declare_parameter("roundabout_entry_gap_px", 180.0)
        self.declare_parameter("roundabout_entry_confirm_frames", 3)
        self.declare_parameter("roundabout_entry_delay_sec", 0.5)

        self.yolo_topic = self.get_parameter("yolo_detections_topic").value
        self.left_topic = self.get_parameter("target_left_topic").value
        self.right_topic = self.get_parameter("target_right_topic").value
        self.route_topic = self.get_parameter("route_select_topic").value
        self.drive_mode_topic = self.get_parameter("drive_mode_topic").value

        self.ttc_enable_topic = self.get_parameter("ttc_enable_topic").value
        self.ttc_enable_entry_count = int(
            self.get_parameter("ttc_enable_entry_count").value
        )

        self.left_arrow_label = self.get_parameter("left_arrow_label").value.lower()
        self.right_arrow_label = self.get_parameter("right_arrow_label").value.lower()
        self.arrow_conf_threshold = float(
            self.get_parameter("arrow_conf_threshold").value
        )

        self.entry_route = self.get_parameter("entry_route").value.lower()
        self.roundabout_route = self.get_parameter("roundabout_route").value.lower()
        self.exit_route = self.get_parameter("exit_route").value.lower()
        self.default_route = self.get_parameter("default_route").value.lower()
        self.final_route = self.get_parameter("final_route").value.lower()
        self.left_arrow_first_exit_index = int(
            self.get_parameter("left_arrow_first_exit_index").value
        )
        self.left_arrow_second_exit_index = int(
            self.get_parameter("left_arrow_second_exit_index").value
        )

        self.right_arrow_first_exit_index = int(
            self.get_parameter("right_arrow_first_exit_index").value
        )
        self.right_arrow_second_exit_index = int(
            self.get_parameter("right_arrow_second_exit_index").value
        )

        self.entry_hold_sec = float(self.get_parameter("entry_hold_sec").value)
        self.exit_hold_sec = float(self.get_parameter("exit_hold_sec").value)

        self.exit_gap_px = float(self.get_parameter("exit_gap_px").value)
        self.exit_gap_reset_px = float(self.get_parameter("exit_gap_reset_px").value)
        self.exit_confirm_frames = int(self.get_parameter("exit_confirm_frames").value)
        self.exit_min_interval_sec = float(
            self.get_parameter("exit_min_interval_sec").value
        )
        self.target_timeout_sec = float(
            self.get_parameter("target_timeout_sec").value
        )
        self.after_exit_left_route = self.get_parameter("after_exit_left_route").value.lower()
        self.after_exit_right_route = self.get_parameter("after_exit_right_route").value.lower()

        self.enable_intersection_fix_after_exit = bool(
            self.get_parameter("enable_intersection_fix_after_exit").value
        )
        self.intersection_trigger_delay_sec = float(
            self.get_parameter("intersection_trigger_delay_sec").value
        )
        self.intersection_trigger_gap_px = float(
            self.get_parameter("intersection_trigger_gap_px").value
        )
        self.intersection_trigger_confirm_frames = int(
            self.get_parameter("intersection_trigger_confirm_frames").value
        )
        self.intersection_fix_left_route = self.get_parameter("intersection_fix_left_route").value.lower()
        self.intersection_fix_right_route = self.get_parameter("intersection_fix_right_route").value.lower()
        self.intersection_fix_hold_sec = float(
            self.get_parameter("intersection_fix_hold_sec").value
        )
        self.after_intersection_route = self.get_parameter("after_intersection_route").value.lower()

        self.roundabout_entry_gap_px = float(
            self.get_parameter("roundabout_entry_gap_px").value
        )
        self.roundabout_entry_confirm_frames = int(
            self.get_parameter("roundabout_entry_confirm_frames").value
        )
        self.roundabout_entry_delay_sec = float(
            self.get_parameter("roundabout_entry_delay_sec").value
        )

        self.gap_max_px = float(self.get_parameter("gap_max_px").value)
        self.gap_jump_max_px = float(self.get_parameter("gap_jump_max_px").value)

        # ============================================================
        # State
        # ============================================================
        self.state = "IDLE"
        # IDLE
        # ENTERING
        # IN_ROUNDABOUT
        # EXITING
        # AFTER_EXIT_LEFT
        # AFTER_EXIT_RIGHT
        # INTERSECTION_LEFT_FIX
        # AFTER_INTERSECTION_RIGHT
        # FINAL_CENTER

        self.desired_exit_index = None
        self.exit_count = 0

        self.left_target = None
        self.right_target = None
        self.left_time = 0.0
        self.right_time = 0.0

        self.gap_confirm_count = 0
        self.split_active = False
        self.last_exit_count_time = 0.0

        self.entry_until = 0.0
        self.exit_until = 0.0

        self.current_route = None
        self.last_log_time = 0.0

        self.after_exit_start_time = 0.0

        self.intersection_trigger_confirm_count = 0
        self.intersection_fix_until = 0.0
        self.intersection_fix_done = False

        self.after_intersection_start_time = 0.0

        self.pending_exit_index = None
        self.pending_arrow_label = None

        self.roundabout_entry_confirm_count = 0
        self.roundabout_entry_enable_time = 0.0
        self.roundabout_entry_count = 0

        self.active_entry_hold_sec = self.entry_hold_sec

        self.last_valid_gap = None

        self.first_exit_index = None
        self.second_exit_index = None
        self.arrow_plan = None

        # ============================================================
        # ROS pubs/subs
        # ============================================================
        self.yolo_sub = self.create_subscription(
            String,
            self.yolo_topic,
            self.yolo_callback,
            10,
        )

        self.left_sub = self.create_subscription(
            PointStamped,
            self.left_topic,
            self.left_callback,
            10,
        )

        self.right_sub = self.create_subscription(
            PointStamped,
            self.right_topic,
            self.right_callback,
            10,
        )

        self.route_pub = self.create_publisher(
            String,
            self.route_topic,
            10,
        )

        self.ttc_enable_pub = self.create_publisher(
            Bool,
            self.ttc_enable_topic,
            10,
        )
        self.drive_mode_pub = self.create_publisher(
            String,
            self.drive_mode_topic,
            10,
        )

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.publish_route(self.default_route)
        self.publish_ttc_enable(False)

        self.get_logger().info("Roundabout Decision Node Started")
        self.get_logger().info(f"Sub YOLO        : {self.yolo_topic}")
        self.get_logger().info(f"Sub target left : {self.left_topic}")
        self.get_logger().info(f"Sub target right: {self.right_topic}")
        self.get_logger().info(f"Pub route       : {self.route_topic}")
        self.get_logger().info(
            f"entry_route={self.entry_route}, "
            f"roundabout_route={self.roundabout_route}, "
            f"exit_route={self.exit_route}"
        )
        self.get_logger().info(
            f"left_arrow plan: "
            f"first={self.left_arrow_first_exit_index}, "
            f"second={self.left_arrow_second_exit_index}"
        )
        self.get_logger().info(
            f"right_arrow plan: "
            f"first={self.right_arrow_first_exit_index}, "
            f"second={self.right_arrow_second_exit_index}"
        )

    # ============================================================
    # Callbacks
    # ============================================================
    def left_callback(self, msg):
        self.left_target = msg
        self.left_time = time.time()

    def right_callback(self, msg):
        self.right_target = msg
        self.right_time = time.time()

    def yolo_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Failed to parse YOLO detections JSON")
            return

        detections = data.get("detections", [])
        if not detections:
            return

        best_left = None
        best_right = None

        for det in detections:
            class_name = str(det.get("class_name", "")).lower()
            confidence = float(det.get("confidence", 0.0))

            if confidence < self.arrow_conf_threshold:
                continue

            if class_name == self.left_arrow_label:
                if best_left is None or confidence > best_left:
                    best_left = confidence

            elif class_name == self.right_arrow_label:
                if best_right is None or confidence > best_right:
                    best_right = confidence

        # 회전교차로 처리 중에는 새 화살표 무시
        if self.state in [
            "ENTERING",
            "IN_ROUNDABOUT",
            "EXITING",
            "AFTER_EXIT_LEFT",
            "AFTER_EXIT_RIGHT",
            "INTERSECTION_LEFT_FIX",
            "INTERSECTION_RIGHT_FIX",
            "AFTER_INTERSECTION_RIGHT",
            "FINAL_CENTER",
        ]:
            return
        
        # 이미 화살표를 한 번 보고 출구 계획이 저장되어 있으면
        # 계속 감지되는 같은 표지판으로 enable_time을 갱신하지 않음
        if self.pending_exit_index is not None:
            return

        if best_right is not None:
            self.arrow_plan = "right_arrow"
            self.first_exit_index = self.right_arrow_first_exit_index
            self.second_exit_index = self.right_arrow_second_exit_index

            self.pending_exit_index = self.first_exit_index
            self.pending_arrow_label = "right_arrow"
            self.roundabout_entry_enable_time = time.time() + self.roundabout_entry_delay_sec
            self.roundabout_entry_confirm_count = 0

            self.get_logger().warn(
                f"RIGHT_ARROW detected | conf={best_right:.3f} | "
                f"first_exit={self.first_exit_index}, "
                f"second_exit={self.second_exit_index} | "
                "waiting for roundabout entry gap"
            )

        elif best_left is not None:
            self.arrow_plan = "left_arrow"
            self.first_exit_index = self.left_arrow_first_exit_index
            self.second_exit_index = self.left_arrow_second_exit_index

            self.pending_exit_index = self.first_exit_index
            self.pending_arrow_label = "left_arrow"
            self.roundabout_entry_enable_time = time.time() + self.roundabout_entry_delay_sec
            self.roundabout_entry_confirm_count = 0

            self.get_logger().warn(
                f"LEFT_ARROW detected | conf={best_left:.3f} | "
                f"first_exit={self.first_exit_index}, "
                f"second_exit={self.second_exit_index} | "
                "waiting for roundabout entry gap"
            )

    # ============================================================
    # Core logic
    # ============================================================
    def start_roundabout(self, desired_exit_index, reason):
        now = time.time()

        self.state = "ENTERING"
        self.desired_exit_index = int(desired_exit_index)
        self.publish_drive_mode("roundabout")

        self.exit_count = 0
        self.gap_confirm_count = 0
        self.split_active = False
        self.last_exit_count_time = 0.0

        self.active_entry_hold_sec = self.entry_hold_sec
        self.entry_until = now + self.active_entry_hold_sec

        # entry gap trigger는 이미 사용했으므로 초기화
        self.roundabout_entry_confirm_count = 0

        self.publish_route(self.entry_route)

        self.get_logger().warn(
            f"ROUNDABOUT START | {reason} | "
            f"desired_exit={self.desired_exit_index} | "
            f"ENTERING route={self.entry_route} for {self.active_entry_hold_sec:.1f}s"
        )

    def targets_ready(self):
        now = time.time()

        if self.left_target is None or self.right_target is None:
            return False

        if now - self.left_time > self.target_timeout_sec:
            return False

        if now - self.right_time > self.target_timeout_sec:
            return False

        # trt_inf.py에서 point.z = 1.0이면 valid, 0.0이면 invalid
        if self.left_target.point.z < 0.5:
            return False

        if self.right_target.point.z < 0.5:
            return False

        return True

    def get_target_gap(self):
        if not self.targets_ready():
            return None

        lx = float(self.left_target.point.x)
        rx = float(self.right_target.point.x)

        return abs(rx - lx)

    def update_exit_count_by_gap(self):
        now = time.time()
        gap = self.get_target_gap()

        if gap is None:
            self.gap_confirm_count = 0
            self.last_valid_gap = None
            return

        # 너무 큰 gap은 잘못 잡힌 cluster로 보고 무시
        if gap > self.gap_max_px:
            self.gap_confirm_count = 0
            self.last_valid_gap = None
            self.get_logger().warn(
                f"Exit gap ignored: too large gap={gap:.1f}px > {self.gap_max_px:.1f}px"
            )
            return

        # 이전 gap과 너무 크게 튀면 무시
        if self.last_valid_gap is not None:
            jump = abs(gap - self.last_valid_gap)

            if jump > self.gap_jump_max_px:
                self.gap_confirm_count = 0
                self.last_valid_gap = None
                self.get_logger().warn(
                    f"Exit gap ignored: jump too large. "
                    f"gap={gap:.1f}px, last={self.last_valid_gap}, jump={jump:.1f}px"
                )
                return

        if gap is None:
            self.gap_confirm_count = 0
            return

        if  self.exit_gap_px <= gap <= self.gap_max_px:
            self.gap_confirm_count += 1
        else:
            self.gap_confirm_count = 0

        # gap이 reset 기준 아래로 내려가야 다음 출구 카운트 가능
        if gap <= self.exit_gap_reset_px:
            if self.split_active:
                self.get_logger().info(f"Exit split reset. gap={gap:.1f}px")
            self.split_active = False
            self.last_valid_gap = None

        if self.split_active:
            return

        if self.gap_confirm_count < self.exit_confirm_frames:
            return

        if now - self.last_exit_count_time < self.exit_min_interval_sec:
            return

        self.exit_count += 1
        self.split_active = True
        self.last_exit_count_time = now

        self.get_logger().warn(
            f"EXIT CANDIDATE COUNTED | "
            f"exit_count={self.exit_count}/{self.desired_exit_index}, "
            f"gap={gap:.1f}px"
        )

    def check_intersection_gap_trigger(self):
        if not self.enable_intersection_fix_after_exit:
            return False

        gap = self.get_target_gap()

        if gap is None:
            self.intersection_trigger_confirm_count = 0
            return False

        if gap >= self.intersection_trigger_gap_px:
            self.intersection_trigger_confirm_count += 1
        else:
            self.intersection_trigger_confirm_count = 0

        if self.intersection_trigger_confirm_count >= self.intersection_trigger_confirm_frames:
            self.get_logger().warn(
                f"INTERSECTION GAP TRIGGER | "
                f"gap={gap:.1f}px >= {self.intersection_trigger_gap_px:.1f}px | "
                f"confirm={self.intersection_trigger_confirm_count}/"
                f"{self.intersection_trigger_confirm_frames}"
            )
            return True

        return False
    
    def check_roundabout_entry_gap_trigger(self):
        if self.pending_exit_index is None:
            return False

        now = time.time()

        if now < self.roundabout_entry_enable_time:
            return False

        gap = self.get_target_gap()

        if gap is None:
            self.roundabout_entry_confirm_count = 0
            return False

        # 추가 1: 너무 큰 gap은 잘못 잡힌 target split으로 보고 무시
        if gap > self.gap_max_px:
            self.roundabout_entry_confirm_count = 0
            self.get_logger().warn(
                f"Roundabout entry gap ignored: too large "
                f"gap={gap:.1f}px > {self.gap_max_px:.1f}px"
            )
            return False

        # 기존 조건 유지
        if self.roundabout_entry_gap_px <= gap <= self.gap_max_px:
            self.roundabout_entry_confirm_count += 1
        else:
            self.roundabout_entry_confirm_count = 0

        if self.roundabout_entry_confirm_count >= self.roundabout_entry_confirm_frames:
            self.get_logger().warn(
                f"ROUNDABOUT ENTRY GAP TRIGGER | "
                f"gap={gap:.1f}px >= {self.roundabout_entry_gap_px:.1f}px | "
                f"confirm={self.roundabout_entry_confirm_count}/"
                f"{self.roundabout_entry_confirm_frames} | "
                f"pending_exit_index={self.pending_exit_index}"
            )
            return True

        return False

    def timer_callback(self):
        now = time.time()

        if self.state == "FINAL_CENTER":
            self.publish_route(self.final_route)
            return

        if self.state == "IDLE":
            self.publish_route(self.default_route)

            if self.check_roundabout_entry_gap_trigger():
                self.roundabout_entry_count += 1

                if self.roundabout_entry_count >= self.ttc_enable_entry_count:
                    self.publish_ttc_enable(True)

                self.start_roundabout(
                    desired_exit_index=self.pending_exit_index,
                    reason=(
                        f"{self.pending_arrow_label}_entry_gap_trigger "
                        f"entry_count={self.roundabout_entry_count}"
                    ),
                )

            return

        # ============================================================
        # 1. ENTERING
        #    진입 직후에는 좌/우 표지판과 상관없이 무조건 right
        # ============================================================
        if self.state == "ENTERING":
            # 진입 중에는 무조건 right 유지
            # 중요: 여기서는 절대 update_exit_count_by_gap() 호출하지 않음
            self.publish_route(self.entry_route)

            if now >= self.entry_until:
                # 진입 구간에서 보였던 갈림길/진입점 split은 버림
                self.exit_count = 0
                self.gap_confirm_count = 0
                self.split_active = False
                self.last_exit_count_time = 0.0

                self.state = "IN_ROUNDABOUT"

                if self.desired_exit_index <= 1:
                    # 우회전은 첫 번째 출구이므로 right 유지
                    self.publish_route(self.exit_route)
                    self.get_logger().warn(
                        f"ENTERING done -> IN_ROUNDABOUT | "
                        f"desired_exit=1, keep route={self.exit_route}"
                    )
                else:
                    # 좌회전은 진입 이후에 left로 회전
                    self.publish_route(self.roundabout_route)
                    self.get_logger().warn(
                        f"ENTERING done -> IN_ROUNDABOUT | "
                        f"desired_exit={self.desired_exit_index}, "
                        f"route={self.roundabout_route} | "
                        f"exit_count reset to 0"
                    )

            return

        # ============================================================
        # 2. IN_ROUNDABOUT
        # ============================================================
        if self.state == "IN_ROUNDABOUT":
            if self.desired_exit_index <= 1:
                # 1번째 출구는 별도 exit gap count 없이 right 고정
                self.publish_route(self.exit_route)

                self.state = "EXITING"
                self.exit_until = now + self.exit_hold_sec

                self.get_logger().warn(
                    f"FIRST EXIT route | desired_exit=1 | "
                    f"route={self.exit_route} | no exit gap counting"
                )

                return

            # 좌회전: 회전교차로 내부에서는 left로 반시계방향 회전
            self.publish_route(self.roundabout_route)
            self.update_exit_count_by_gap()

            if self.exit_count >= self.desired_exit_index:
                self.state = "EXITING"
                self.exit_until = now + self.exit_hold_sec
                self.publish_route(self.exit_route)

                self.get_logger().warn(
                    f"TARGET EXIT reached | "
                    f"exit_count={self.exit_count}, "
                    f"route={self.exit_route}, "
                    f"hold={self.exit_hold_sec:.1f}s"
                )

            return

        # ============================================================
        # 3. EXITING
        # ============================================================
        if self.state == "EXITING":
            self.publish_route(self.exit_route)

            if now >= self.exit_until:
                self.publish_drive_mode("normal")

                # 2번째 회전교차로까지 끝났으면 최종 종료
                if self.roundabout_entry_count >= 2:
                    self.state = "FINAL_CENTER"

                    # 마지막 좌/중앙/우 교차로에서는 center target을 따라감
                    self.publish_route(self.final_route)

                    self.get_logger().warn(
                        f"Final roundabout completed. "
                        f"State=FINAL_CENTER, route={self.final_route}"
                    )
                    return

                # 1번째 회전교차로 탈출 후 공통 초기화
                self.after_exit_start_time = now
                self.intersection_trigger_confirm_count = 0
                self.intersection_fix_done = False

                # right_arrow 시나리오:
                # 1차 회전교차로 1번 출구 탈출 후 right 유지
                if self.arrow_plan == "right_arrow":
                    self.state = "AFTER_EXIT_RIGHT"
                    self.roundabout_entry_confirm_count = 0
                    self.roundabout_entry_enable_time = now + self.roundabout_entry_delay_sec

                    # 두 번째 회전교차로는 right_arrow 계획상 3번째 출구
                    if self.second_exit_index is not None:
                        self.pending_exit_index = self.second_exit_index
                        self.pending_arrow_label = self.arrow_plan

                    self.publish_route(self.after_exit_right_route)

                    self.get_logger().warn(
                        f"Roundabout exit completed. "
                        f"State=AFTER_EXIT_RIGHT, route={self.after_exit_right_route}. "
                        f"Next roundabout entry gap trigger enabled after "
                        f"{self.roundabout_entry_delay_sec:.1f}s"
                    )

                    return

                # left_arrow 시나리오:
                # 1차 회전교차로 3번 출구 탈출 후 left 유지 + 일반 교차로 보정
                self.state = "AFTER_EXIT_LEFT"
                self.publish_route(self.after_exit_left_route)

                self.get_logger().warn(
                    f"Roundabout exit completed. "
                    f"State=AFTER_EXIT_LEFT, route={self.after_exit_left_route}. "
                    f"Intersection gap trigger enabled after "
                    f"{self.intersection_trigger_delay_sec:.1f}s"
                )

                return

        # ============================================================
        # AFTER_EXIT_LEFT
        # 1차 회전교차로 탈출 후 left 유지
        # 여기서 첫 번째 gap trigger는 일반 교차로 통과용 right 보정
        # ============================================================
        if self.state == "AFTER_EXIT_LEFT":
            self.publish_route(self.after_exit_left_route)

            elapsed = now - self.after_exit_start_time

            if now - self.last_log_time > 1.0:
                self.get_logger().info(
                    f"AFTER_EXIT_LEFT | route={self.after_exit_left_route} | "
                    f"elapsed={elapsed:.1f}s"
                )
                self.last_log_time = now

            # 2번째 회전교차로까지 끝난 뒤에는 더 이상 일반 교차로 보정 안 함
            if self.roundabout_entry_count >= 2:
                if now - self.last_log_time > 1.0:
                    self.get_logger().info(
                        f"FINAL AFTER_EXIT_LEFT | route={self.after_exit_left_route}"
                    )
                    self.last_log_time = now
                return

            if elapsed < self.intersection_trigger_delay_sec:
                return

            if not self.intersection_fix_done and self.check_intersection_gap_trigger():
                self.state = "INTERSECTION_LEFT_FIX"
                self.intersection_fix_until = now + self.intersection_fix_hold_sec
                self.intersection_fix_done = True

                # left_arrow 시나리오: 일반 교차로에서는 left를 유지
                self.publish_route(self.intersection_fix_left_route)

                self.get_logger().warn(
                    f"INTERSECTION_LEFT_FIX START | "
                    f"route={self.intersection_fix_left_route} for "
                    f"{self.intersection_fix_hold_sec:.1f}s"
                )

            return
        
        # ============================================================
        # AFTER_EXIT_RIGHT
        # right_arrow 시나리오:
        # 1차 회전교차로 1번 출구 탈출 후 right 유지
        # 여기서 gap trigger는 일반 교차로 right 보정 시작점
        # ============================================================
        if self.state == "AFTER_EXIT_RIGHT":
            self.publish_route(self.after_exit_right_route)

            elapsed = now - self.after_exit_start_time

            if now - self.last_log_time > 1.0:
                self.get_logger().info(
                    f"AFTER_EXIT_RIGHT | route={self.after_exit_right_route} | "
                    f"elapsed={elapsed:.1f}s"
                )
                self.last_log_time = now

            if elapsed < self.intersection_trigger_delay_sec:
                return

            if not self.intersection_fix_done and self.check_intersection_gap_trigger():
                self.state = "INTERSECTION_RIGHT_FIX"
                self.intersection_fix_until = now + self.intersection_fix_hold_sec
                self.intersection_fix_done = True

                # right_arrow 시나리오: 일반 교차로에서는 right를 유지
                self.publish_route(self.intersection_fix_right_route)

                self.get_logger().warn(
                    f"INTERSECTION_RIGHT_FIX START | "
                    f"route={self.intersection_fix_right_route} for "
                    f"{self.intersection_fix_hold_sec:.1f}s"
                )

            return


        # ============================================================
        # INTERSECTION_LEFT_FIX
        # 일반 교차로에서 left를 5초 유지한 뒤 right로 전환
        # ============================================================
        if self.state == "INTERSECTION_LEFT_FIX":
            # left_arrow 시나리오: 일반 교차로 통과 중에는 left 유지
            self.publish_route(self.intersection_fix_left_route)

            if now >= self.intersection_fix_until:
                self.state = "AFTER_INTERSECTION_RIGHT"
                self.after_intersection_start_time = now

                self.roundabout_entry_confirm_count = 0
                self.roundabout_entry_enable_time = now + self.roundabout_entry_delay_sec

                if self.second_exit_index is not None:
                    self.pending_exit_index = self.second_exit_index
                    self.pending_arrow_label = self.arrow_plan

                self.publish_route(self.after_intersection_route)

                self.get_logger().warn(
                    f"INTERSECTION_LEFT_FIX done. "
                    f"State=AFTER_INTERSECTION_RIGHT, "
                    f"route={self.after_intersection_route}. "
                    f"Next roundabout entry gap trigger enabled after "
                    f"{self.roundabout_entry_delay_sec:.1f}s"
                )

            return
        
        # ============================================================
        # INTERSECTION_RIGHT_FIX
        # right_arrow 시나리오:
        # 일반 교차로에서 right를 일정 시간 유지한 뒤
        # 기존 AFTER_INTERSECTION_RIGHT 상태로 이동
        # ============================================================
        if self.state == "INTERSECTION_RIGHT_FIX":
            self.publish_route(self.intersection_fix_right_route)

            if now >= self.intersection_fix_until:
                self.state = "AFTER_INTERSECTION_RIGHT"
                self.after_intersection_start_time = now

                self.roundabout_entry_confirm_count = 0
                self.roundabout_entry_enable_time = now + self.roundabout_entry_delay_sec

                if self.second_exit_index is not None:
                    self.pending_exit_index = self.second_exit_index
                    self.pending_arrow_label = self.arrow_plan

                # left/right 시나리오 모두 보정 후에는 right 유지
                self.publish_route(self.after_intersection_route)

                self.get_logger().warn(
                    f"INTERSECTION_RIGHT_FIX done. "
                    f"State=AFTER_INTERSECTION_RIGHT, "
                    f"route={self.after_intersection_route}. "
                    f"Next roundabout entry gap trigger enabled after "
                    f"{self.roundabout_entry_delay_sec:.1f}s"
                )

            return

        # ============================================================
        # AFTER_INTERSECTION_RIGHT
        # 일반 교차로 통과 후 right 유지
        # 여기서 두 번째 gap trigger는 2차 회전교차로 진입
        # ============================================================
        if self.state == "AFTER_INTERSECTION_RIGHT":
            self.publish_route(self.after_intersection_route)

            elapsed = now - self.after_intersection_start_time

            if now - self.last_log_time > 1.0:
                self.get_logger().info(
                    f"AFTER_INTERSECTION_RIGHT | "
                    f"route={self.after_intersection_route} | "
                    f"elapsed={elapsed:.1f}s"
                )
                self.last_log_time = now

            if elapsed < self.roundabout_entry_delay_sec:
                return

            if self.check_roundabout_entry_gap_trigger():
                self.roundabout_entry_count += 1

                if self.roundabout_entry_count >= self.ttc_enable_entry_count:
                    self.publish_ttc_enable(True)

                if self.roundabout_entry_count >= 2:
                    desired_exit = self.second_exit_index
                else:
                    desired_exit = self.first_exit_index

                self.start_roundabout(
                    desired_exit_index=desired_exit,
                    reason=(
                        f"roundabout_entry_gap_trigger "
                        f"entry_count={self.roundabout_entry_count}, "
                        f"plan={self.arrow_plan}, "
                        f"desired_exit={desired_exit}"
                    ),
                )

            return

    def publish_route(self, route):
        route = route.lower()

        if route not in ["left", "right", "center"]:
            self.get_logger().warn(f"Invalid route: {route}")
            return

        msg = String()
        msg.data = route
        self.route_pub.publish(msg)

        if self.current_route != route:
            self.current_route = route
            self.get_logger().info(f"Route select: {route}")
    
    def publish_ttc_enable(self, enable):
        msg = Bool()
        msg.data = bool(enable)
        self.ttc_enable_pub.publish(msg)

        if enable:
            self.get_logger().warn("TTC safety ENABLED")
        else:
            self.get_logger().info("TTC safety disabled")

    def publish_drive_mode(self, mode):
        msg = String()
        msg.data = mode
        self.drive_mode_pub.publish(msg)
        self.get_logger().info(f"Drive mode publish: {mode}")


def main(args=None):
    rclpy.init(args=args)

    node = RoundaboutDecisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_ttc_enable(False)
        node.publish_route("right")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()