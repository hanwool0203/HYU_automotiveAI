import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class CenterlineTRTNode(Node):
    def __init__(self):
        super().__init__("centerline_trt_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter(
            "engine_path",
            "/home/ircv7/workspace/ros2_ws/models/centerline_unet_mobilenetv2_fp16.engine",
        )
        self.declare_parameter("image_topic", "/left/image_raw")
        self.declare_parameter("input_width", 320)
        self.declare_parameter("input_height", 180)
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("publish_mask", True)

        # target point 추출용
        self.declare_parameter("roi_y_ratio", 0.45)
        self.declare_parameter("lookahead_y_ratio", 0.50)
        self.declare_parameter("target_band_px", 25)
        self.declare_parameter("min_pixels", 20)

        # 갈림길 target 선택용
        self.declare_parameter("default_route", "right")  # "left", "right", "center"
        self.declare_parameter("route_select_topic", "/route_select")
        self.declare_parameter("branch_x_gap_px", 80)
        self.declare_parameter("branch_min_pixels", 30)

        self.route_mode = self.get_parameter("default_route").value

        self.declare_parameter("bev_width", 1200)
        self.declare_parameter("bev_height", 1000)

        self.bev_width = int(self.get_parameter("bev_width").value)
        self.bev_height = int(self.get_parameter("bev_height").value)

        self.engine_path = self.get_parameter("engine_path").value
        self.image_topic = self.get_parameter("image_topic").value
        self.input_w = int(self.get_parameter("input_width").value)
        self.input_h = int(self.get_parameter("input_height").value)
        self.threshold = float(self.get_parameter("threshold").value)
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.publish_mask = bool(self.get_parameter("publish_mask").value)

        self.roi_y_ratio = float(self.get_parameter("roi_y_ratio").value)
        self.lookahead_y_ratio = float(self.get_parameter("lookahead_y_ratio").value)
        self.target_band_px = int(self.get_parameter("target_band_px").value)
        self.min_pixels = int(self.get_parameter("min_pixels").value)

        self.bridge = CvBridge()

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # =========================
        # TensorRT init
        # =========================
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine(self.engine_path)
        self.trt_context = self.engine.create_execution_context()

        self.input_name, self.output_name = self.get_io_names()

        self.get_logger().info(f"TensorRT engine loaded: {self.engine_path}")
        self.get_logger().info(f"input tensor : {self.input_name}")
        self.get_logger().info(f"output tensor: {self.output_name}")

        engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.get_logger().info(f"engine input shape : {engine_input_shape}")

        # Dynamic shape 대비
        self.input_shape = (1, 3, self.input_h, self.input_w)
        if -1 in engine_input_shape:
            self.trt_context.set_input_shape(self.input_name, self.input_shape)

        self.output_shape = tuple(self.trt_context.get_tensor_shape(self.output_name))
        self.get_logger().info(f"output shape: {self.output_shape}")

        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        # Host buffers
        self.host_input = np.empty(self.input_shape, dtype=np.float32)
        self.host_output = np.empty(self.output_shape, dtype=self.output_dtype)

        # Device buffers
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        self.stream = cuda.Stream()

        self.trt_context.set_tensor_address(self.input_name, int(self.device_input))
        self.trt_context.set_tensor_address(self.output_name, int(self.device_output))

        self.H = np.array([
            [-7.63367966e-01, -3.03807669e+00,  7.08730102e+02],
            [-4.18081681e-02, -7.21555308e+00,  1.42319841e+03],
            [5.88367703e-06, -6.47483950e-03,  1.00000000e+00]
        ], dtype=np.float32)

        # =========================
        # ROS pubs/subs
        # =========================
        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.mask_pub = self.create_publisher(Image, "/centerline/mask", 10)
        self.overlay_pub = self.create_publisher(Image, "/centerline/overlay", 10)
        self.target_pub = self.create_publisher(PointStamped, "/centerline/target_point", 10)
        
        self.target_left_pub = self.create_publisher(
            PointStamped,
            "/centerline/target_left",
            10
        )

        self.target_right_pub = self.create_publisher(
            PointStamped,
            "/centerline/target_right",
            10
        )
        self.camera_overlay_pub = self.create_publisher(
            Image,
            "/centerline/camera_overlay",
            10
        )

        route_select_topic = self.get_parameter("route_select_topic").value

        self.route_sub = self.create_subscription(
            String,
            route_select_topic,
            self.route_callback,
            10
        )

        self.get_logger().info(f"Default route: {self.route_mode}")
        self.get_logger().info(f"Subscribing route select topic: {route_select_topic}")

        self.frame_count = 0
        self.last_log_time = time.time()

        self.get_logger().info(f"Subscribed image topic: {self.image_topic}")
        self.get_logger().info("Centerline TensorRT node started.")

    def load_engine(self, engine_path):
        engine_path = Path(engine_path)

        if not engine_path.exists():
            raise FileNotFoundError(f"engine file not found: {engine_path}")

        with open(str(engine_path), "rb") as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

        return engine

    def get_io_names(self):
        input_names = []
        output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                output_names.append(name)

        if len(input_names) != 1:
            raise RuntimeError(f"Expected 1 input, got {input_names}")
        if len(output_names) != 1:
            raise RuntimeError(f"Expected 1 output, got {output_names}")

        return input_names[0], output_names[0]

    def preprocess(self, image_bgr):
        resized = cv2.resize(image_bgr, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)

        return np.ascontiguousarray(x.astype(np.float32)), resized

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def infer(self, input_tensor):
        np.copyto(self.host_input, input_tensor)

        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)

        ok = self.trt_context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")

        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()

        return self.host_output

    def extract_target_point(self, mask):
        """
        mask: uint8, shape [H, W], 0 or 255

        return:
            target_x, target_y, valid
        """
        h, w = mask.shape

        roi_start_y = int(h * self.roi_y_ratio)
        target_y = int(h * self.lookahead_y_ratio)

        y1 = max(roi_start_y, target_y - self.target_band_px)
        y2 = min(h, target_y + self.target_band_px + 1)

        band = mask[y1:y2, :]
        ys, xs = np.where(band > 0)

        if len(xs) >= self.min_pixels:
            target_x = int(np.mean(xs))
            target_y_abs = int(y1 + np.mean(ys))
            return target_x, target_y_abs, True

        # fallback: ROI 전체에서 가장 아래쪽 가까운 centerline 평균
        roi = mask[roi_start_y:h, :]
        ys, xs = np.where(roi > 0)

        if len(xs) >= self.min_pixels:
            # 아래쪽에 가까운 픽셀을 우선 사용
            ys_abs = ys + roi_start_y
            max_y = np.max(ys_abs)
            near = np.abs(ys_abs - max_y) < 15

            target_x = int(np.mean(xs[near]))
            target_y_abs = int(np.mean(ys_abs[near]))
            return target_x, target_y_abs, True

        return w // 2, target_y, False

    def route_callback(self, msg):
        route = msg.data.strip().lower()

        if route not in ["left", "right", "center"]:
            self.get_logger().warn(f"Unknown route command: {route}")
            return

        self.route_mode = route
        self.get_logger().info(f"Route mode changed to: {self.route_mode}")

    def extract_left_right_targets_bev(self, bev_mask):
        """
        BEV mask에서 lookahead band 주변의 centerline 후보를
        x 방향 cluster로 나누고 left/right/center target을 만든다.

        return:
            left_target   = (x, y, valid)
            right_target  = (x, y, valid)
            center_target = (x, y, valid)
        """
        h, w = bev_mask.shape

        branch_x_gap_px = int(self.get_parameter("branch_x_gap_px").value)
        branch_min_pixels = int(self.get_parameter("branch_min_pixels").value)

        roi_start_y = int(h * self.roi_y_ratio)
        target_y = int(h * self.lookahead_y_ratio)

        y1 = max(roi_start_y, target_y - self.target_band_px)
        y2 = min(h, target_y + self.target_band_px + 1)

        band = bev_mask[y1:y2, :]
        ys, xs = np.where(band > 0)

        # lookahead band에 픽셀이 부족하면 기존 BEV target 추출로 fallback
        if len(xs) < branch_min_pixels:
            tx, ty, valid = self.extract_target_point_bev(bev_mask)
            fallback = (tx, ty, valid)
            return fallback, fallback, fallback

        # x 좌표 기준 cluster 나누기
        unique_x = np.sort(np.unique(xs))

        clusters = []
        current = [unique_x[0]]

        for x in unique_x[1:]:
            if x - current[-1] > branch_x_gap_px:
                clusters.append(current)
                current = [x]
            else:
                current.append(x)

        clusters.append(current)

        candidates = []

        for cluster in clusters:
            x_min = int(cluster[0])
            x_max = int(cluster[-1])

            in_cluster = (xs >= x_min) & (xs <= x_max)

            if np.sum(in_cluster) < branch_min_pixels:
                continue

            cx = int(np.mean(xs[in_cluster]))
            cy = int(y1 + np.mean(ys[in_cluster]))
            count = int(np.sum(in_cluster))

            candidates.append((cx, cy, count))

        if len(candidates) == 0:
            tx, ty, valid = self.extract_target_point_bev(bev_mask)
            fallback = (tx, ty, valid)
            return fallback, fallback, fallback

        candidates = sorted(candidates, key=lambda p: p[0])

        left = candidates[0]
        right = candidates[-1]

        # center는 BEV 중앙에 가장 가까운 후보
        bev_cx = w / 2.0
        center = min(candidates, key=lambda p: abs(p[0] - bev_cx))

        left_target = (left[0], left[1], True)
        right_target = (right[0], right[1], True)
        center_target = (center[0], center[1], True)

        return left_target, right_target, center_target

    def select_target_by_route(self, left_target, right_target, center_target):
        if self.route_mode == "left":
            return left_target
        elif self.route_mode == "right":
            return right_target
        elif self.route_mode == "center":
            return center_target
        else:
            return right_target

    def publish_target_msg(self, pub, msg_header, target):
        x, y, valid = target

        point_msg = PointStamped()
        point_msg.header = msg_header
        point_msg.point.x = float(x)
        point_msg.point.y = float(y)
        point_msg.point.z = 1.0 if valid else 0.0

        pub.publish(point_msg)

    def extract_target_point_bev(self, bev_mask):
        """
        BEV mask에서 Pure Pursuit용 target point를 뽑는다.

        bev_mask: uint8, shape [bev_height, bev_width], 0 or 255

        return:
            target_x, target_y, valid
        """
        h, w = bev_mask.shape

        roi_start_y = int(h * self.roi_y_ratio)
        target_y = int(h * self.lookahead_y_ratio)

        y1 = max(roi_start_y, target_y - self.target_band_px)
        y2 = min(h, target_y + self.target_band_px + 1)

        band = bev_mask[y1:y2, :]
        ys, xs = np.where(band > 0)

        if len(xs) >= self.min_pixels:
            target_x = int(np.mean(xs))
            target_y_abs = int(y1 + np.mean(ys))
            return target_x, target_y_abs, True

        # fallback: ROI 전체에서 target_y에 가까운 픽셀 사용
        roi = bev_mask[roi_start_y:h, :]
        ys, xs = np.where(roi > 0)

        if len(xs) >= self.min_pixels:
            ys_abs = ys + roi_start_y

            near = np.abs(ys_abs - target_y) < 60

            if np.sum(near) >= self.min_pixels:
                target_x = int(np.mean(xs[near]))
                target_y_abs = int(np.mean(ys_abs[near]))
                return target_x, target_y_abs, True

            # 그래도 없으면 전체 ROI 평균
            target_x = int(np.mean(xs))
            target_y_abs = int(np.mean(ys_abs))
            return target_x, target_y_abs, True

        return w // 2, target_y, False

    def make_bev_overlay(self, bev_mask, left_target, right_target, center_target, selected_target):
        """
        BEV mask 위에 left/right/center/selected target을 표시하는 디버그 overlay.
        """
        overlay = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)

        # mask 빨간색 표시
        overlay[bev_mask > 0] = (0, 0, 255)

        # BEV 중앙 참고선
        car_center_x = int(self.bev_width // 2)
        car_y = int(self.bev_height - 1)

        cv2.line(
            overlay,
            (car_center_x, 0),
            (car_center_x, self.bev_height - 1),
            (255, 255, 255),
            1
        )

        # 차량 위치
        cv2.circle(overlay, (car_center_x, car_y), 8, (255, 0, 0), -1)

        lx, ly, lv = left_target
        rx, ry, rv = right_target
        cx, cy, cv = center_target
        sx, sy, sv = selected_target

        # left 후보: 파란색
        if lv:
            cv2.circle(overlay, (int(lx), int(ly)), 8, (255, 0, 0), -1)
            cv2.putText(
                overlay,
                "L",
                (int(lx) + 8, int(ly)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # right 후보: 노란색
        if rv:
            cv2.circle(overlay, (int(rx), int(ry)), 8, (0, 255, 255), -1)
            cv2.putText(
                overlay,
                "R",
                (int(rx) + 8, int(ry)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # center 후보: 보라색
        if cv:
            cv2.circle(overlay, (int(cx), int(cy)), 7, (255, 0, 255), -1)

        # selected target: 초록색
        if sv:
            cv2.circle(overlay, (int(sx), int(sy)), 12, (0, 255, 0), 2)

        cv2.putText(
            overlay,
            f"route={self.route_mode} selected=({int(sx)},{int(sy)})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay

    def image_callback(self, msg):
        try:
            t0 = time.time()

            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # =========================
            # 1. TensorRT inference
            # =========================
            input_tensor, resized_bgr = self.preprocess(image_bgr)
            output = self.infer(input_tensor)

            logits = output[0, 0]
            prob = self.sigmoid(logits)

            # 320x180 model output mask
            mask_small = (prob > self.threshold).astype(np.uint8) * 255

            # =========================
            # 2. 320x180 mask → original camera size
            # =========================
            orig_h, orig_w = image_bgr.shape[:2]

            mask_orig = cv2.resize(
                mask_small,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

            # =========================
            # Camera overlay publish
            # =========================
            camera_overlay = image_bgr.copy()

            # mask 영역을 빨간색으로 표시
            red_layer = np.zeros_like(camera_overlay)
            red_layer[:, :] = (0, 0, 255)

            mask_bool = mask_orig > 0

            # 원본 이미지와 빨간 mask를 반투명 합성
            camera_overlay[mask_bool] = cv2.addWeighted(
                camera_overlay[mask_bool],
                0.4,
                red_layer[mask_bool],
                0.6,
                0
            )

            # 원본 이미지 기준 target point도 표시하고 싶으면 기존 extract_target_point 사용
            cam_tx, cam_ty, cam_valid = self.extract_target_point(mask_orig)

            if cam_valid:
                cv2.circle(camera_overlay, (int(cam_tx), int(cam_ty)), 8, (0, 255, 0), -1)
                cv2.putText(
                    camera_overlay,
                    f"camera target=({cam_tx},{cam_ty})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            camera_overlay_msg = self.bridge.cv2_to_imgmsg(camera_overlay, encoding="bgr8")
            camera_overlay_msg.header = msg.header
            self.camera_overlay_pub.publish(camera_overlay_msg)

            # =========================
            # 3. original mask → BEV mask
            # =========================
            bev_mask = cv2.warpPerspective(
                mask_orig,
                self.H,
                (self.bev_width, self.bev_height),
                flags=cv2.INTER_NEAREST
            )

            # =========================
            # 4. BEV에서 left/right/center target 후보 추출
            # =========================
            left_target, right_target, center_target = self.extract_left_right_targets_bev(bev_mask)

            # =========================
            # 5. route_mode에 따라 최종 target 선택
            #    기본값은 right
            # =========================
            selected_target = self.select_target_by_route(
                left_target,
                right_target,
                center_target
            )

            target_x, target_y, valid = selected_target

            # =========================
            # 6. target 후보 publish
            # =========================
            self.publish_target_msg(
                self.target_left_pub,
                msg.header,
                left_target
            )

            self.publish_target_msg(
                self.target_right_pub,
                msg.header,
                right_target
            )

            # =========================
            # 7. 선택된 target publish
            #    기존 제어 노드는 /centerline/target_point만 보면 됨
            # =========================
            self.publish_target_msg(
                self.target_pub,
                msg.header,
                selected_target
            )

            # =========================
            # 8. BEV mask publish
            # =========================
            if self.publish_mask:
                mask_msg = self.bridge.cv2_to_imgmsg(bev_mask, encoding="mono8")
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)

            # =========================
            # 9. BEV overlay publish
            # =========================
            if self.publish_overlay:
                overlay = self.make_bev_overlay(
                    bev_mask,
                    left_target,
                    right_target,
                    center_target,
                    selected_target
                )
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                overlay_msg.header = msg.header
                self.overlay_pub.publish(overlay_msg)

            self.frame_count += 1
            now = time.time()
            if now - self.last_log_time >= 1.0:
                fps = self.frame_count / (now - self.last_log_time)
                dt_ms = (now - t0) * 1000.0
                self.get_logger().info(
                    f"FPS={fps:.1f}, last={dt_ms:.2f} ms, "
                    f"BEV target=({target_x},{target_y}), valid={valid}"
                )
                self.frame_count = 0
                self.last_log_time = now

        except Exception as e:
            self.get_logger().error(f"inference callback error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CenterlineTRTNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()