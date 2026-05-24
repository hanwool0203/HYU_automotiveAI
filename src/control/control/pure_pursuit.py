#!/usr/bin/env python3

import math
import time
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

from .base_ctrl import BaseController


class LanePurePursuitController(Node):
    def __init__(self):
        super().__init__('lane_pure_pursuit_controller')

        # =========================
        # ROS Parameters
        # =========================
        self.declare_parameter('target_point_topic', '/centerline/target_point')

        # lane_detector_right.py의 BEV 크기와 반드시 맞춰야 함
        self.declare_parameter('bev_width', 1200.0)
        self.declare_parameter('bev_height', 1000.0)

        # 차량 기준점.
        # 처음에는 BEV 이미지 아래 중앙으로 둔다.
        # 차량을 트랙 중앙에 세웠을 때 target_x가 계속 570 근처면 car_center_x를 570으로 바꿔도 됨.
        self.declare_parameter('car_center_x', 500.0)
        self.declare_parameter('car_y', 1000.0)

        # =========================
        # Meter scale parameters
        # =========================
        # 실제 측정해서 넣어야 하는 값
        # 예: 실제 차선 간격 0.435m, BEV에서 520px이면 0.435 / 605 = 0.00125
        self.declare_parameter('meter_per_pixel_x', 0.000719)

        # 예: 전방 0.50m 마커가 BEV에서 250px이면 0.305 / 340 = 0.002
        self.declare_parameter('meter_per_pixel_y', 0.00089)

        # 좌우 바퀴 중심 간 거리 [m]
        self.declare_parameter('track_width_m', 0.13)

        # =========================
        # Control parameters
        # =========================

        # 실제 m/s 기준 목표 속도.
        # 초반에는 낮게 시작.
        self.declare_parameter('base_speed_mps', 0.20)
        self.declare_parameter('roundabout_speed_mps', 0.15)

        # m/s를 JSON 명령값으로 바꾸는 비율.
        # 예: 0.12 m/s일 때 JSON 0.10 정도가 적당하면 cmd_per_mps = 0.10 / 0.12 = 0.83
        # 일단 1.0으로 시작해도 됨.
        self.declare_parameter('cmd_per_mps', 1.0)

        # 최종 JSON 명령 제한
        self.declare_parameter('max_cmd', 0.35)

        # 최소 추종 거리 [m]
        self.declare_parameter('min_lookahead_m', 0.20)

        # 곡률 보정 gain.
        # 정석 pure pursuit는 1.0.
        # 너무 흔들리면 0.5~0.8, 커브를 못 돌면 1.2~1.5
        self.declare_parameter('curvature_gain', 6.5)

        # target point가 끊겼을 때 정지까지 걸리는 시간
        self.declare_parameter('target_timeout', 0.3)

        # 제어 주기
        self.declare_parameter('control_hz', 20.0)

        # 모터 방향 매핑
        # 네 키보드 코드 기준: 최종 JSON 전송 전에 부호 반전 필요, 좌우 swap은 하지 않음
        self.declare_parameter('invert_motor_direction', False)
        self.declare_parameter('swap_left_right', False)

        # =========================
        # Decision / drive mode parameters
        # =========================
        self.declare_parameter('drive_mode_topic', '/drive_mode')
        self.declare_parameter('slow_speed_scale', 0.5)

        # USB 포트
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)

        serial_port = self.get_parameter('serial_port').value
        baudrate = int(self.get_parameter('baudrate').value)

        self.base = BaseController(serial_port, baudrate)

        self.latest_target = None
        self.latest_stamp_time = None
        self.drive_mode = 'normal'

        target_topic = self.get_parameter('target_point_topic').value
        self.target_sub = self.create_subscription(
            PointStamped,
            target_topic,
            self.target_callback,
            10
        )

        drive_mode_topic = self.get_parameter('drive_mode_topic').value
        self.drive_mode_sub = self.create_subscription(
            String,
            drive_mode_topic,
            self.drive_mode_callback,
            10
        )
        self.get_logger().info(f'Subscribing drive mode: {drive_mode_topic}')

        control_hz = float(self.get_parameter('control_hz').value)
        self.timer = self.create_timer(1.0 / control_hz, self.control_loop)

        self.get_logger().info('Meter-based Lane Pure Pursuit Controller Started')
        self.get_logger().info(f'Subscribing target point: {target_topic}')
        self.get_logger().info(f'Using serial: {serial_port}, baudrate: {baudrate}')

    def target_callback(self, msg):
        self.latest_target = msg
        self.latest_stamp_time = time.time()

    def drive_mode_callback(self, msg):
        mode = msg.data.strip().lower()

        if mode not in ['normal', 'slow', 'stop', 'roundabout']:
            self.get_logger().warn(
                f'Unknown drive mode: {mode}',
                throttle_duration_sec=0.5
            )
            return

        if mode != self.drive_mode:
            self.get_logger().info(f'Drive mode changed: {self.drive_mode} -> {mode}')

        self.drive_mode = mode
    
    def control_loop(self):
        if self.latest_target is None:
            self.send_stop()
            return

        target_timeout = float(self.get_parameter('target_timeout').value)

        if self.latest_stamp_time is None:
            self.send_stop()
            return

        if time.time() - self.latest_stamp_time > target_timeout:
            self.get_logger().warn('Target timeout. Stop vehicle.')
            self.send_stop()
            return

        tx_px = self.latest_target.point.x
        ty_px = self.latest_target.point.y
        valid = self.latest_target.point.z > 0.5

        if not valid:
            self.get_logger().warn(
                'Invalid centerline target. Stop vehicle.',
                throttle_duration_sec=0.5
            )
            self.send_stop()
            return
        
        if self.drive_mode == 'stop':
            self.get_logger().warn(
                'Drive mode STOP. Stop vehicle.',
                throttle_duration_sec=0.5
            )
            self.send_stop()
            return

        left_cmd, right_cmd = self.compute_pure_pursuit_meter(tx_px, ty_px)

        self.send_motor_command(left_cmd, right_cmd)

    def compute_pure_pursuit_meter(self, tx_px, ty_px):
        # =========================
        # Get parameters
        # =========================
        car_x_px = float(self.get_parameter('car_center_x').value)
        car_y_px = float(self.get_parameter('car_y').value)

        meter_per_pixel_x = float(self.get_parameter('meter_per_pixel_x').value)
        meter_per_pixel_y = float(self.get_parameter('meter_per_pixel_y').value)

        track_width_m = float(self.get_parameter('track_width_m').value)
        base_speed_mps = float(self.get_parameter('base_speed_mps').value)

        if self.drive_mode == 'roundabout':
            base_speed_mps = float(self.get_parameter('roundabout_speed_mps').value)

        elif self.drive_mode == 'slow':
            slow_speed_scale = float(self.get_parameter('slow_speed_scale').value)
            base_speed_mps *= slow_speed_scale        

        cmd_per_mps = float(self.get_parameter('cmd_per_mps').value)
        max_cmd = float(self.get_parameter('max_cmd').value)
        min_lookahead_m = float(self.get_parameter('min_lookahead_m').value)
        curvature_gain = float(self.get_parameter('curvature_gain').value)

        # =========================
        # Pixel coordinate → Vehicle coordinate [m]
        # =========================
        # BEV image coordinate:
        # x: right positive
        # y: downward positive
        #
        # Vehicle coordinate:
        # forward_m: forward positive
        # lateral_m: left positive
        #
        # target이 이미지 오른쪽에 있으면 tx_px - car_x_px > 0
        # 차량 기준 오른쪽은 negative lateral로 둘 것이므로 - 부호를 붙인다.
        forward_m = (car_y_px - ty_px) * meter_per_pixel_y
        lateral_m = -(tx_px - car_x_px) * meter_per_pixel_x

        lookahead_m = math.sqrt(forward_m ** 2 + lateral_m ** 2)

        if lookahead_m < min_lookahead_m:
            self.get_logger().warn(
                f'Lookahead too small: {lookahead_m:.3f} m. Stop.',
                throttle_duration_sec=0.5
            )
            return 0.0, 0.0

        # target이 차량 뒤쪽이면 정지
        if forward_m <= 0.0:
            self.get_logger().warn(
                f'Target is behind or too close. forward_m={forward_m:.3f}. Stop.',
                throttle_duration_sec=0.5
            )
            return 0.0, 0.0

        # =========================
        # Pure pursuit curvature
        # =========================
        # curvature [1/m]
        curvature = 2.0 * lateral_m / (lookahead_m ** 2)
        curvature *= curvature_gain

        # angular velocity [rad/s]
        omega = base_speed_mps * curvature

        # =========================
        # Differential drive kinematics [m/s]
        # =========================
        left_mps = base_speed_mps - omega * track_width_m / 2.0
        right_mps = base_speed_mps + omega * track_width_m / 2.0

        # =========================
        # m/s → JSON command
        # =========================
        left_cmd = left_mps * cmd_per_mps
        right_cmd = right_mps * cmd_per_mps

        # 너무 큰 값 제한
        left_cmd = self.clip(left_cmd, -max_cmd, max_cmd)
        right_cmd = self.clip(right_cmd, -max_cmd, max_cmd)

        self.get_logger().info(
            f'target_px=({tx_px:.1f},{ty_px:.1f}) | '
            f'forward={forward_m:.3f}m lateral={lateral_m:.3f}m '
            f'Ld={lookahead_m:.3f}m curv={curvature:.3f} | '
            f'mps L={left_mps:.3f}, R={right_mps:.3f} | '
            f'cmd L={left_cmd:.3f}, R={right_cmd:.3f}',
            throttle_duration_sec=0.5
        )

        return left_cmd, right_cmd

    def send_motor_command(self, left, right):
        # 일반 제어기 기준 명령 → 실제 차량 모터 기준 명령으로 변환
        left_mapped, right_mapped = self.apply_motor_mapping(left, right)

        cmd = {
            "T": 1,
            "L": float(left_mapped),
            "R": float(right_mapped)
        }

        threading.Thread(
            target=self.base.base_json_ctrl,
            args=(cmd,),
            daemon=True
        ).start()

        self.get_logger().info(
            f'raw L={left:.3f}, R={right:.3f} | '
            f'mapped L={left_mapped:.3f}, R={right_mapped:.3f}',
            throttle_duration_sec=0.5
        )

    def apply_motor_mapping(self, left, right):
        """
        내부 계산 기준:
        +값 = 전진

        실제 차량 기준:
        키보드 코드에서 최종적으로 send_control_async(-L, -R)를 사용했으므로
        JSON 전송 직전에 부호 반전이 필요하다.
        좌우 swap은 기본 False.
        """

        invert_motor_direction = bool(
            self.get_parameter('invert_motor_direction').value
        )
        swap_left_right = bool(
            self.get_parameter('swap_left_right').value
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
            "R": 0.0
        }

        threading.Thread(
            target=self.base.base_json_ctrl,
            args=(cmd,),
            daemon=True
        ).start()

    @staticmethod
    def clip(value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def destroy_node(self):
        self.send_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = LanePurePursuitController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.send_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()