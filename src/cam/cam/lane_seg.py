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

import onnxruntime as ort


class LaneMulticlassONNXNode(Node):
    """
    Multiclass lane segmentation ONNX node.

    Model output assumption:
        [1, 4, H, W]

    Class id:
        0 = background
        1 = left
        2 = center
        3 = right

    Main output topic is kept compatible with the previous controller:
        /centerline/target_point
    """

    def __init__(self):
        super().__init__("lane_multiclass_onnx_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter(
            "model_path",
            "/home/ircv7/workspace/final_ws/models/lane_multiclass_unet_mobilenetv2.onnx",
        )
        self.declare_parameter("image_topic", "/left/image_raw")
        self.declare_parameter("input_width", 320)
        self.declare_parameter("input_height", 180)

        # publish options
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("publish_mask", True)
        self.declare_parameter("publish_camera_overlay", True)

        # ONNX Runtime 설정
        # Jetson에서 onnxruntime-gpu가 설치되어 있으면 CUDAExecutionProvider 사용
        self.declare_parameter("use_cuda", True)

        # class settings
        self.declare_parameter("num_classes", 4)

        # target point 추출용
        self.declare_parameter("roi_y_ratio", 0.45)
        self.declare_parameter("lookahead_y_ratio", 0.50)
        self.declare_parameter("target_band_px", 25)
        self.declare_parameter("min_pixels", 20)

        # route 선택용
        # /route_select 에 "left", "center", "right" String을 publish하면 변경됨
        self.declare_parameter("default_route", "center")
        self.declare_parameter("route_select_topic", "/route_select")

        # BEV output size
        self.declare_parameter("bev_width", 1200)
        self.declare_parameter("bev_height", 1000)

        self.model_path = self.get_parameter("model_path").value
        self.image_topic = self.get_parameter("image_topic").value
        self.input_w = int(self.get_parameter("input_width").value)
        self.input_h = int(self.get_parameter("input_height").value)

        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.publish_mask = bool(self.get_parameter("publish_mask").value)
        self.publish_camera_overlay = bool(self.get_parameter("publish_camera_overlay").value)
        self.use_cuda = bool(self.get_parameter("use_cuda").value)
        self.num_classes = int(self.get_parameter("num_classes").value)

        self.roi_y_ratio = float(self.get_parameter("roi_y_ratio").value)
        self.lookahead_y_ratio = float(self.get_parameter("lookahead_y_ratio").value)
        self.target_band_px = int(self.get_parameter("target_band_px").value)
        self.min_pixels = int(self.get_parameter("min_pixels").value)

        self.route_mode = self.get_parameter("default_route").value.strip().lower()

        self.bev_width = int(self.get_parameter("bev_width").value)
        self.bev_height = int(self.get_parameter("bev_height").value)

        self.class_id = {
            "left": 1,
            "center": 2,
            "right": 3,
        }

        if self.route_mode not in self.class_id:
            self.get_logger().warn(
                f"Unknown default_route={self.route_mode}. Fallback to center."
            )
            self.route_mode = "center"

        self.bridge = CvBridge()

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # =========================
        # ONNX Runtime init
        # =========================
        self.session = self.load_onnx_session(self.model_path)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]

        self.get_logger().info(f"ONNX model loaded: {self.model_path}")
        self.get_logger().info(f"input tensor : {self.input_name}")
        self.get_logger().info(f"output tensor: {self.output_name}")
        self.get_logger().info(f"providers    : {self.session.get_providers()}")
        self.get_logger().info(f"ONNX input shape : {input_meta.shape}")
        self.get_logger().info(f"ONNX output shape: {output_meta.shape}")

        # 기존 centerline_onnx_node.py에서 사용하던 homography 유지
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

        # 기존 제어 노드 호환을 위해 /centerline 이름 유지
        self.mask_pub = self.create_publisher(Image, "/centerline/mask", 10)
        self.overlay_pub = self.create_publisher(Image, "/centerline/overlay", 10)
        self.camera_overlay_pub = self.create_publisher(
            Image,
            "/centerline/camera_overlay",
            10
        )

        self.target_pub = self.create_publisher(
            PointStamped,
            "/centerline/target_point",
            10
        )
        self.target_left_pub = self.create_publisher(
            PointStamped,
            "/centerline/target_left",
            10
        )
        self.target_center_pub = self.create_publisher(
            PointStamped,
            "/centerline/target_center",
            10
        )
        self.target_right_pub = self.create_publisher(
            PointStamped,
            "/centerline/target_right",
            10
        )

        route_select_topic = self.get_parameter("route_select_topic").value
        self.route_sub = self.create_subscription(
            String,
            route_select_topic,
            self.route_callback,
            10
        )

        self.frame_count = 0
        self.last_log_time = time.time()

        self.get_logger().info(f"Subscribed image topic: {self.image_topic}")
        self.get_logger().info(f"Default route: {self.route_mode}")
        self.get_logger().info(f"Subscribing route select topic: {route_select_topic}")
        self.get_logger().info("Lane multiclass ONNX node started.")

    def load_onnx_session(self, model_path):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {model_path}")

        available = ort.get_available_providers()
        providers = []

        if self.use_cuda and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )

    def preprocess(self, image_bgr):
        resized = cv2.resize(image_bgr, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, axis=0)   # CHW -> BCHW

        return np.ascontiguousarray(x.astype(np.float32)), resized

    def infer(self, input_tensor):
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0]

    def output_to_class_mask(self, output):
        """
        ONNX output을 multiclass mask로 변환한다.

        expected:
            [1, 4, H, W] 또는 [4, H, W]

        return:
            class_mask: uint8 [H, W]
                0 = background
                1 = left
                2 = center
                3 = right
        """
        out = np.asarray(output)

        if out.ndim == 4:
            out = out[0]  # [1, C, H, W] -> [C, H, W]

        if out.ndim != 3:
            raise RuntimeError(
                f"Unexpected multiclass ONNX output shape: {out.shape}. "
                "Expected [1, C, H, W] or [C, H, W]."
            )

        c, h, w = out.shape

        if c != self.num_classes:
            self.get_logger().warn(
                f"Output channel count={c}, but num_classes={self.num_classes}. "
                "Argmax will still be applied."
            )

        class_mask = np.argmax(out, axis=0).astype(np.uint8)
        return class_mask

    def route_callback(self, msg):
        route = msg.data.strip().lower()

        if route not in self.class_id:
            self.get_logger().warn(f"Unknown route command: {route}")
            return

        self.route_mode = route
        self.get_logger().info(f"Route mode changed to: {self.route_mode}")

    def extract_target_point_bev(self, bev_binary_mask):
        """
        BEV binary mask에서 Pure Pursuit용 target point를 뽑는다.

        bev_binary_mask:
            uint8 [bev_height, bev_width], 0 or 255

        return:
            (target_x, target_y, valid)
        """
        h, w = bev_binary_mask.shape

        roi_start_y = int(h * self.roi_y_ratio)
        target_y = int(h * self.lookahead_y_ratio)

        y1 = max(roi_start_y, target_y - self.target_band_px)
        y2 = min(h, target_y + self.target_band_px + 1)

        band = bev_binary_mask[y1:y2, :]
        ys, xs = np.where(band > 0)

        if len(xs) >= self.min_pixels:
            target_x = int(np.mean(xs))
            target_y_abs = int(y1 + np.mean(ys))
            return target_x, target_y_abs, True

        # fallback: ROI 전체에서 target_y에 가까운 픽셀 사용
        roi = bev_binary_mask[roi_start_y:h, :]
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

    def select_target_by_route(self, left_target, center_target, right_target):
        if self.route_mode == "left":
            return left_target
        if self.route_mode == "right":
            return right_target
        return center_target

    def publish_target_msg(self, pub, msg_header, target):
        x, y, valid = target

        point_msg = PointStamped()
        point_msg.header = msg_header
        point_msg.point.x = float(x)
        point_msg.point.y = float(y)
        point_msg.point.z = 1.0 if valid else 0.0

        pub.publish(point_msg)

    def make_camera_overlay(self, image_bgr, class_mask_orig, selected_camera_target=None):
        """
        원본 camera image 위에 multiclass lane mask를 표시한다.
        BGR color:
            left   = red
            center = green
            right  = blue
            selected target = yellow
        """
        overlay = image_bgr.copy()

        color_layer = np.zeros_like(overlay)

        color_layer[class_mask_orig == 1] = (0, 0, 255)    # left red
        color_layer[class_mask_orig == 2] = (0, 255, 0)    # center green
        color_layer[class_mask_orig == 3] = (255, 0, 0)    # right blue

        mask_bool = class_mask_orig > 0
        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool],
            0.4,
            color_layer[mask_bool],
            0.6,
            0
        )

        if selected_camera_target is not None:
            tx, ty, valid = selected_camera_target
            if valid:
                cv2.circle(overlay, (int(tx), int(ty)), 8, (0, 255, 255), -1)
                cv2.putText(
                    overlay,
                    f"camera route={self.route_mode} target=({tx},{ty})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return overlay

    def make_bev_overlay(self, bev_class_mask, left_target, center_target, right_target, selected_target):
        """
        BEV class mask 위에 left/center/right/selected target을 표시한다.
        """
        overlay = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)

        overlay[bev_class_mask == 1] = (0, 0, 255)     # left red
        overlay[bev_class_mask == 2] = (0, 255, 0)     # center green
        overlay[bev_class_mask == 3] = (255, 0, 0)     # right blue

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
        cv2.circle(overlay, (car_center_x, car_y), 8, (255, 255, 255), -1)

        lx, ly, lv = left_target
        cx, cy, cv = center_target
        rx, ry, rv = right_target
        sx, sy, sv = selected_target

        if lv:
            cv2.circle(overlay, (int(lx), int(ly)), 8, (0, 0, 255), -1)
            cv2.putText(
                overlay, "L", (int(lx) + 8, int(ly)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
            )

        if cv:
            cv2.circle(overlay, (int(cx), int(cy)), 8, (0, 255, 0), -1)
            cv2.putText(
                overlay, "C", (int(cx) + 8, int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )

        if rv:
            cv2.circle(overlay, (int(rx), int(ry)), 8, (255, 0, 0), -1)
            cv2.putText(
                overlay, "R", (int(rx) + 8, int(ry)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA
            )

        if sv:
            cv2.circle(overlay, (int(sx), int(sy)), 12, (0, 255, 255), 2)

        cv2.putText(
            overlay,
            f"route={self.route_mode} selected=({int(sx)},{int(sy)}) valid={sv}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay

    def make_gray_publish_mask(self, bev_class_mask):
        """
        /centerline/mask는 mono8로 유지한다.
        값은 class id 확인이 쉽도록 0/80/160/240으로 publish한다.
            0   = background
            80  = left
            160 = center
            240 = right
        """
        gray = np.zeros_like(bev_class_mask, dtype=np.uint8)
        gray[bev_class_mask == 1] = 80
        gray[bev_class_mask == 2] = 160
        gray[bev_class_mask == 3] = 240
        return gray

    def image_callback(self, msg):
        try:
            t0 = time.time()

            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # =========================
            # 1. ONNX Runtime inference
            # =========================
            input_tensor, _ = self.preprocess(image_bgr)
            output = self.infer(input_tensor)

            # model output: [1, 4, 180, 320]
            # class_mask_small: 0=background, 1=left, 2=center, 3=right
            class_mask_small = self.output_to_class_mask(output)

            # =========================
            # 2. 320x180 class mask → original camera size
            # =========================
            orig_h, orig_w = image_bgr.shape[:2]
            class_mask_orig = cv2.resize(
                class_mask_small,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

            # =========================
            # 3. original class mask → BEV class mask
            # =========================
            bev_class_mask = cv2.warpPerspective(
                class_mask_orig,
                self.H,
                (self.bev_width, self.bev_height),
                flags=cv2.INTER_NEAREST
            )

            # =========================
            # 4. class별 BEV binary mask 생성
            # =========================
            bev_left_mask = (bev_class_mask == 1).astype(np.uint8) * 255
            bev_center_mask = (bev_class_mask == 2).astype(np.uint8) * 255
            bev_right_mask = (bev_class_mask == 3).astype(np.uint8) * 255

            # =========================
            # 5. class별 target 추출
            # =========================
            left_target = self.extract_target_point_bev(bev_left_mask)
            center_target = self.extract_target_point_bev(bev_center_mask)
            right_target = self.extract_target_point_bev(bev_right_mask)

            # =========================
            # 6. route_mode에 따라 최종 target 선택
            # =========================
            selected_target = self.select_target_by_route(
                left_target,
                center_target,
                right_target
            )
            target_x, target_y, valid = selected_target

            # =========================
            # 7. target 후보 publish
            # =========================
            self.publish_target_msg(self.target_left_pub, msg.header, left_target)
            self.publish_target_msg(self.target_center_pub, msg.header, center_target)
            self.publish_target_msg(self.target_right_pub, msg.header, right_target)

            # 기존 제어 노드는 이 토픽만 보면 됨
            self.publish_target_msg(self.target_pub, msg.header, selected_target)

            # =========================
            # 8. mask / overlay publish
            # =========================
            if self.publish_mask:
                mask_msg = self.bridge.cv2_to_imgmsg(
                    self.make_gray_publish_mask(bev_class_mask),
                    encoding="mono8"
                )
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)

            if self.publish_overlay:
                overlay = self.make_bev_overlay(
                    bev_class_mask,
                    left_target,
                    center_target,
                    right_target,
                    selected_target
                )
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                overlay_msg.header = msg.header
                self.overlay_pub.publish(overlay_msg)

            if self.publish_camera_overlay:
                # camera overlay에서는 selected class의 camera 좌표 target도 표시
                selected_class_id = self.class_id.get(self.route_mode, 2)
                selected_cam_mask = (class_mask_orig == selected_class_id).astype(np.uint8) * 255

                # camera image 기준 target은 BEV 제어에는 쓰지 않고 디버그 표시용
                selected_camera_target = self.extract_target_point_camera(selected_cam_mask)

                camera_overlay = self.make_camera_overlay(
                    image_bgr,
                    class_mask_orig,
                    selected_camera_target
                )
                camera_overlay_msg = self.bridge.cv2_to_imgmsg(camera_overlay, encoding="bgr8")
                camera_overlay_msg.header = msg.header
                self.camera_overlay_pub.publish(camera_overlay_msg)

            self.frame_count += 1
            now = time.time()
            if now - self.last_log_time >= 1.0:
                fps = self.frame_count / (now - self.last_log_time)
                dt_ms = (now - t0) * 1000.0

                left_count = int(np.sum(bev_class_mask == 1))
                center_count = int(np.sum(bev_class_mask == 2))
                right_count = int(np.sum(bev_class_mask == 3))

                self.get_logger().info(
                    f"FPS={fps:.1f}, last={dt_ms:.2f} ms, "
                    f"route={self.route_mode}, "
                    f"BEV target=({target_x},{target_y}), valid={valid}, "
                    f"pixels L/C/R={left_count}/{center_count}/{right_count}"
                )
                self.frame_count = 0
                self.last_log_time = now

        except Exception as e:
            self.get_logger().error(f"inference callback error: {e}")

    def extract_target_point_camera(self, binary_mask):
        """
        원본 camera image 기준 디버그 target.
        제어에는 쓰지 않음.
        """
        h, w = binary_mask.shape

        roi_start_y = int(h * self.roi_y_ratio)
        target_y = int(h * self.lookahead_y_ratio)

        y1 = max(roi_start_y, target_y - self.target_band_px)
        y2 = min(h, target_y + self.target_band_px + 1)

        band = binary_mask[y1:y2, :]
        ys, xs = np.where(band > 0)

        if len(xs) >= self.min_pixels:
            target_x = int(np.mean(xs))
            target_y_abs = int(y1 + np.mean(ys))
            return target_x, target_y_abs, True

        roi = binary_mask[roi_start_y:h, :]
        ys, xs = np.where(roi > 0)

        if len(xs) >= self.min_pixels:
            ys_abs = ys + roi_start_y
            max_y = np.max(ys_abs)
            near = np.abs(ys_abs - max_y) < 15

            target_x = int(np.mean(xs[near]))
            target_y_abs = int(np.mean(ys_abs[near]))
            return target_x, target_y_abs, True

        return w // 2, target_y, False


def main(args=None):
    rclpy.init(args=args)
    node = LaneMulticlassONNXNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
