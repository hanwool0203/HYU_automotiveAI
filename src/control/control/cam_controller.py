#!/usr/bin/env python3

import time
import threading
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

from .base_ctrl import BaseController


class CameraTargetDiffController(Node):
    def __init__(self):
        super().__init__("camera_target_diff_controller")

        # =========================
        # ROS Parameters
        # =========================
        self.declare_parameter("target_point_topic", "/centerline/target_point")

        # TensorRT 입력 이미지 크기
        self.declare_parameter("image_width", 320.0)
        self.declare_parameter("image_height", 180.0)

        # 차량 기준 중심 x
        # 보통 image_width / 2 = 160
        # 카메라가 삐뚤어져 있으면 155, 165 이런 식으로 보정 가능
        self.declare_parameter("car_center_x", 160.0)

        # 기본 전진 명령값
        # BaseController JSON에 들어가는 값 기준
        self.declare_parameter("base_cmd", 0.15)

        # 커브에서 최소 속도
        self.declare_parameter("min_cmd", 0.12)

        # 최종 모터 명령 제한
        self.declare_parameter("max_cmd", 0.40)

        # 조향 gain
        # target이 오른쪽으로 벗어났을 때 좌우 바퀴 속도 차이를 얼마나 줄지
        self.declare_parameter("turn_gain", 0.28)

        # D 제어. target이 튀는 경우 조금 완화
        self.declare_parameter("turn_d_gain", 0.01)

        # 너무 작은 오차는 무시
        self.declare_parameter("deadband", 0.00)

        # 커브에서 속도 줄이기
        self.declare_parameter("slow_down_on_curve", False)
        self.declare_parameter("slowdown_ratio", 0.5)

        # target 끊겼을 때 정지까지 시간
        self.declare_parameter("target_timeout", 0.3)

        # 제어 주기
        self.declare_parameter("control_hz", 20.0)

        # 모터 방향 매핑
        self.declare_parameter("invert_motor_direction", False)
        self.declare_parameter("swap_left_right", False)

        self.declare_parameter("cmd_alpha", 0.45)

        self.prev_left_cmd = 0.0
        self.prev_right_cmd = 0.0

        # USB 포트
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)

        target_topic = self.get_parameter("target_point_topic").value
        serial_port = self.get_parameter("serial_port").value
        baudrate = int(self.get_parameter("baudrate").value)

        self.base = BaseController(serial_port, baudrate)

        self.latest_target = None
        self.latest_stamp_time = None

        self.prev_error = 0.0
        self.prev_time = None

        self.target_sub = self.create_subscription(
            PointStamped,
            target_topic,
            self.target_callback,
            10,
        )

        control_hz = float(self.get_parameter("control_hz").value)
        self.timer = self.create_timer(1.0 / control_hz, self.control_loop)

        self.get_logger().info("Camera Target Differential Controller Started")
        self.get_logger().info(f"Subscribing target point: {target_topic}")
        self.get_logger().info(f"Using serial: {serial_port}, baudrate: {baudrate}")

    def target_callback(self, msg):
        self.latest_target = msg
        self.latest_stamp_time = time.time()

    def control_loop(self):
        if self.latest_target is None:
            self.send_stop()
            return

        target_timeout = float(self.get_parameter("target_timeout").value)

        if self.latest_stamp_time is None:
            self.send_stop()
            return

        if time.time() - self.latest_stamp_time > target_timeout:
            self.get_logger().warn(
                "Target timeout. Stop vehicle.",
                throttle_duration_sec=0.5,
            )
            self.send_stop()
            return

        valid = self.latest_target.point.z > 0.5

        if not valid:
            self.get_logger().warn(
                "Invalid target. Stop vehicle.",
                throttle_duration_sec=0.5,
            )
            self.send_stop()
            return

        tx = float(self.latest_target.point.x)
        ty = float(self.latest_target.point.y)

        left_cmd, right_cmd = self.compute_diff_command(tx, ty)
        self.send_motor_command(left_cmd, right_cmd)

    def compute_diff_command(self, tx, ty):
        image_width = float(self.get_parameter("image_width").value)
        image_height = float(self.get_parameter("image_height").value)
        car_center_x = float(self.get_parameter("car_center_x").value)

        base_cmd = float(self.get_parameter("base_cmd").value)
        min_cmd = float(self.get_parameter("min_cmd").value)
        max_cmd = float(self.get_parameter("max_cmd").value)

        turn_gain = float(self.get_parameter("turn_gain").value)
        turn_d_gain = float(self.get_parameter("turn_d_gain").value)
        deadband = float(self.get_parameter("deadband").value)

        slow_down_on_curve = bool(self.get_parameter("slow_down_on_curve").value)
        slowdown_ratio = float(self.get_parameter("slowdown_ratio").value)

        # =========================
        # Camera coordinate error
        # =========================
        # error > 0: target이 이미지 오른쪽
        # error < 0: target이 이미지 왼쪽
        error_px = tx - car_center_x

        # -1 ~ 1 근처로 정규화
        norm_error = error_px / (image_width / 2.0)

        # deadband 적용
        if abs(norm_error) < deadband:
            norm_error = 0.0

        # =========================
        # D term
        # =========================
        now = time.time()

        if self.prev_time is None:
            d_error = 0.0
        else:
            dt = now - self.prev_time
            if dt > 1e-6:
                d_error = (norm_error - self.prev_error) / dt
            else:
                d_error = 0.0

        self.prev_time = now
        self.prev_error = norm_error

        # =========================
        # Turn command
        # =========================
        # abs(norm_error)가 클수록 더 강하게 돌도록 비선형 증폭
        curve = min(abs(norm_error), 1.0)

        # target이 많이 벗어나면 turn을 더 크게
        turn_cmd = turn_gain * norm_error * (1.0 + 1.5 * curve)

        # D항은 처음에는 너무 튈 수 있으니 약하게만 적용
        turn_cmd += turn_d_gain * d_error

        # 회전 명령 제한
        turn_cmd = self.clip(turn_cmd, -max_cmd, max_cmd)

        # =========================
        # Speed command
        # =========================
        # target이 많이 벗어나면 전진 속도를 확 줄여서 회전이 잘 되게 함
        if slow_down_on_curve:
            speed_cmd = base_cmd * (1.0 - slowdown_ratio * curve)
            speed_cmd = self.clip(speed_cmd, min_cmd, base_cmd)
        else:
            speed_cmd = base_cmd

        # # =========================
        # # Differential command
        # # =========================
        # left_cmd = speed_cmd + turn_cmd
        # right_cmd = speed_cmd - turn_cmd

        # # target이 크게 왼쪽/오른쪽이면 pivot turn 허용
        # pivot_threshold = 0.45
        # pivot_inner_cmd = 0.16   # 안쪽 바퀴 후진 세기
        # pivot_outer_cmd = 0.30   # 바깥쪽 바퀴 전진 세기

        # if norm_error < -pivot_threshold:
        #     # target이 왼쪽 → 왼쪽으로 강하게 회전
        #     left_cmd = -pivot_inner_cmd
        #     right_cmd = pivot_outer_cmd

        # elif norm_error > pivot_threshold:
        #     # target이 오른쪽 → 오른쪽으로 강하게 회전
        #     left_cmd = pivot_outer_cmd
        #     right_cmd = -pivot_inner_cmd

        # =========================
        # Smooth turn command
        # =========================
        curve = min(abs(norm_error), 1.0)

        # 작은 오차에서는 약하게, 큰 오차에서는 점진적으로 강하게
        # norm_error^3 항을 섞어서 부드럽게 증가시킴
        smooth_error = 0.6 * norm_error + 0.4 * (norm_error ** 3)

        turn_cmd = turn_gain * smooth_error + turn_d_gain * d_error

        # 너무 갑자기 큰 회전 방지
        max_turn_cmd = max_cmd * 0.85
        turn_cmd = self.clip(turn_cmd, -max_turn_cmd, max_turn_cmd)

        # =========================
        # Smooth speed command
        # =========================
        if slow_down_on_curve:
            # 오차가 커질수록 부드럽게 감속
            speed_cmd = base_cmd * (1.0 - slowdown_ratio * curve)
            speed_cmd = self.clip(speed_cmd, min_cmd, base_cmd)
        else:
            speed_cmd = base_cmd

        # =========================
        # Differential command
        # =========================
        left_cmd = speed_cmd + turn_cmd
        right_cmd = speed_cmd - turn_cmd

        left_cmd = self.clip(left_cmd, -max_cmd, max_cmd)
        right_cmd = self.clip(right_cmd, -max_cmd, max_cmd)

        self.get_logger().info(
            f"target=({tx:.1f},{ty:.1f}) "
            f"err_px={error_px:.1f} norm={norm_error:.3f} "
            f"speed={speed_cmd:.3f} turn={turn_cmd:.3f} "
            f"cmd L={left_cmd:.3f}, R={right_cmd:.3f}",
            throttle_duration_sec=0.3,
        )

        cmd_alpha = float(self.get_parameter("cmd_alpha").value)

        left_cmd = cmd_alpha * left_cmd + (1.0 - cmd_alpha) * self.prev_left_cmd
        right_cmd = cmd_alpha * right_cmd + (1.0 - cmd_alpha) * self.prev_right_cmd

        self.prev_left_cmd = left_cmd
        self.prev_right_cmd = right_cmd

        return left_cmd, right_cmd

    def send_motor_command(self, left, right):
        left_mapped, right_mapped = self.apply_motor_mapping(left, right)

        cmd = {
            "T": 1,
            "L": float(left_mapped),
            "R": float(right_mapped),
        }

        threading.Thread(
            target=self.base.base_json_ctrl,
            args=(cmd,),
            daemon=True,
        ).start()

        self.get_logger().info(
            f"raw L={left:.3f}, R={right:.3f} | "
            f"mapped L={left_mapped:.3f}, R={right_mapped:.3f}",
            throttle_duration_sec=0.5,
        )

    def apply_motor_mapping(self, left, right):
        invert_motor_direction = bool(
            self.get_parameter("invert_motor_direction").value
        )
        swap_left_right = bool(
            self.get_parameter("swap_left_right").value
        )

        if invert_motor_direction:
            left = -left
            right = -right

        if swap_left_right:
            left, right = right, left

        return left, right

    def send_stop(self):
        cmd = {
            "T": 1,
            "L": 0.0,
            "R": 0.0,
        }

        threading.Thread(
            target=self.base.base_json_ctrl,
            args=(cmd,),
            daemon=True,
        ).start()

    @staticmethod
    def clip(value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def destroy_node(self):
        self.send_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = CameraTargetDiffController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.send_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()