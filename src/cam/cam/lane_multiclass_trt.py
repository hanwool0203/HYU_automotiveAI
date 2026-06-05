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


class TRTInferenceEngine:
    """
    TensorRT .engine inference wrapper.

    입력/출력은 1개씩인 segmentation engine을 기준으로 한다.
    - input : [1, 3, H, W], float32
    - output: [1, C, H, W], float32 또는 fp16

    TensorRT 8.x의 binding API와 TensorRT 10.x의 tensor API를 최대한 같이 지원한다.
    """

    def __init__(self, engine_path, input_shape, logger=None):
        self.engine_path = Path(engine_path)
        self.input_shape = tuple(input_shape)
        self.node_logger = logger

        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {self.engine_path}")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        cuda.init()
        self.cuda_context = cuda.Device(0).make_context()

        try:
            with open(self.engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")

            self.stream = cuda.Stream()
            self.use_tensor_api = hasattr(self.engine, "num_io_tensors")

            self.input_name = None
            self.output_name = None
            self.input_dtype = np.float32
            self.output_dtype = np.float32

            self._discover_io_tensors()
            self._set_dynamic_input_shape_if_needed()
            self._allocate_buffers()

        finally:
            self.cuda_context.pop()

    def _log(self, msg):
        if self.node_logger is not None:
            self.node_logger.info(msg)

    def _discover_io_tensors(self):
        if self.use_tensor_api:
            # TensorRT 8.5+ / 10.x style API
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))

                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                    self.input_dtype = dtype
                else:
                    self.output_name = name
                    self.output_dtype = dtype
        else:
            # TensorRT 8.x legacy binding API
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))

                if self.engine.binding_is_input(i):
                    self.input_name = name
                    self.input_binding_idx = i
                    self.input_dtype = dtype
                else:
                    self.output_name = name
                    self.output_binding_idx = i
                    self.output_dtype = dtype

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Could not find exactly one input and one output tensor in engine")

    def _get_tensor_shape(self, name, is_input=False):
        if self.use_tensor_api:
            if is_input:
                return tuple(self.context.get_tensor_shape(name))
            return tuple(self.context.get_tensor_shape(name))
        else:
            idx = self.input_binding_idx if is_input else self.output_binding_idx
            return tuple(self.context.get_binding_shape(idx))

    def _set_dynamic_input_shape_if_needed(self):
        if self.use_tensor_api:
            engine_shape = tuple(self.engine.get_tensor_shape(self.input_name))
            if any(dim < 0 for dim in engine_shape):
                ok = self.context.set_input_shape(self.input_name, self.input_shape)
                if not ok:
                    raise RuntimeError(
                        f"Failed to set dynamic TensorRT input shape: {self.input_shape}"
                    )
        else:
            engine_shape = tuple(self.engine.get_binding_shape(self.input_binding_idx))
            if any(dim < 0 for dim in engine_shape):
                self.context.set_binding_shape(self.input_binding_idx, self.input_shape)
                if not self.context.all_binding_shapes_specified:
                    raise RuntimeError(
                        f"Failed to set dynamic TensorRT binding shape: {self.input_shape}"
                    )

    def _allocate_buffers(self):
        input_shape = self._get_tensor_shape(self.input_name, is_input=True)
        output_shape = self._get_tensor_shape(self.output_name, is_input=False)

        # 일부 TRT 버전에서 static shape 조회가 engine 기준으로만 정상인 경우 대비
        if any(dim < 0 for dim in input_shape):
            input_shape = self.input_shape
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"Invalid TensorRT output shape after setting input: {output_shape}")

        self.input_shape_runtime = tuple(int(x) for x in input_shape)
        self.output_shape_runtime = tuple(int(x) for x in output_shape)

        self.host_input = cuda.pagelocked_empty(
            int(np.prod(self.input_shape_runtime)), self.input_dtype
        )
        self.host_output = cuda.pagelocked_empty(
            int(np.prod(self.output_shape_runtime)), self.output_dtype
        )

        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        if not self.use_tensor_api:
            self.bindings = [0] * self.engine.num_bindings
            self.bindings[self.input_binding_idx] = int(self.device_input)
            self.bindings[self.output_binding_idx] = int(self.device_output)

        self._log(f"TensorRT engine loaded: {self.engine_path}")
        self._log(f"TRT input : {self.input_name}, shape={self.input_shape_runtime}, dtype={self.input_dtype}")
        self._log(f"TRT output: {self.output_name}, shape={self.output_shape_runtime}, dtype={self.output_dtype}")

    def infer(self, input_tensor):
        input_tensor = np.ascontiguousarray(input_tensor.astype(self.input_dtype, copy=False))

        if input_tensor.shape != self.input_shape_runtime:
            raise RuntimeError(
                f"Input shape mismatch. Got {input_tensor.shape}, "
                f"but engine expects {self.input_shape_runtime}"
            )

        self.cuda_context.push()
        try:
            np.copyto(self.host_input, input_tensor.ravel())
            cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)

            if self.use_tensor_api and hasattr(self.context, "execute_async_v3"):
                self.context.set_tensor_address(self.input_name, int(self.device_input))
                self.context.set_tensor_address(self.output_name, int(self.device_output))
                ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                ok = self.context.execute_async_v2(
                    bindings=self.bindings,
                    stream_handle=self.stream.handle,
                )

            if not ok:
                raise RuntimeError("TensorRT execution failed")

            cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
            self.stream.synchronize()

            output = np.array(self.host_output, copy=True).reshape(self.output_shape_runtime)
            return output

        finally:
            self.cuda_context.pop()

    def destroy(self):
        # ROS 종료 시 CUDA context 정리
        try:
            self.cuda_context.push()
            self.cuda_context.pop()
            self.cuda_context.detach()
        except Exception:
            pass


class LaneMulticlassTRTNode(Node):
    """
    Multiclass lane segmentation TensorRT node.

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
        super().__init__("lane_multiclass_trt_node")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter(
            "engine_path",
            "/home/ircv7/workspace/final_ws/models/lane_multiclass_unet_mobilenetv2.engine",
        )
        self.declare_parameter("image_topic", "/left/image_raw")
        self.declare_parameter("input_width", 320)
        self.declare_parameter("input_height", 180)

        # publish options
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("publish_mask", True)
        self.declare_parameter("publish_camera_overlay", True)

        # class settings
        self.declare_parameter("num_classes", 4)

        # target point 추출용
        self.declare_parameter("roi_y_ratio", 0.45)
        self.declare_parameter("lookahead_y_ratio", 0.60)
        self.declare_parameter("target_band_px", 25)
        self.declare_parameter("min_pixels", 20)

        # route 선택용
        # /route_select 에 "left", "center", "right" String을 publish하면 변경됨
        self.declare_parameter("default_route", "center")
        self.declare_parameter("route_select_topic", "/route_select")

        # BEV output size
        self.declare_parameter("bev_width", 1200)
        self.declare_parameter("bev_height", 1000)

        self.engine_path = self.get_parameter("engine_path").value
        self.image_topic = self.get_parameter("image_topic").value
        self.input_w = int(self.get_parameter("input_width").value)
        self.input_h = int(self.get_parameter("input_height").value)

        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.publish_mask = bool(self.get_parameter("publish_mask").value)
        self.publish_camera_overlay = bool(self.get_parameter("publish_camera_overlay").value)
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

        # driving route mode
        # left_lane  : left-center 사이를 주행 target으로 사용
        # right_lane : center-right 사이를 주행 target으로 사용
        # center     : center 선 자체를 target으로 사용
        self.valid_routes = ["left_lane", "right_lane", "center"]

        if self.route_mode not in self.valid_routes:
            self.get_logger().warn(
                f"Unknown default_route={self.route_mode}. Fallback to right_lane."
            )
            self.route_mode = "right_lane"

        self.bridge = CvBridge()

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # =========================
        # TensorRT init
        # =========================
        self.trt_engine = TRTInferenceEngine(
            self.engine_path,
            input_shape=(1, 3, self.input_h, self.input_w),
            logger=self.get_logger(),
        )

        self.get_logger().info(f"TensorRT model loaded: {self.engine_path}")

        # 기존 centerline_onnx_node.py에서 사용하던 homography 유지
        self.H = np.array([
            [-9.05848770e-01, -4.62939149e+00,  9.20145719e+02],
            [-7.01254023e-02, -9.56641980e+00,  1.64696870e+03],
            [-7.90287230e-05, -7.28716972e-03,  1.00000000e+00]
        ], dtype=np.float32)

        # =========================
        # ROS pubs/subs
        # =========================
        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            1,
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
        self.get_logger().info("Lane multiclass TensorRT node started.")

    def preprocess(self, image_bgr):
        resized = cv2.resize(image_bgr, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, axis=0)   # CHW -> BCHW

        return np.ascontiguousarray(x.astype(np.float32)), resized

    def infer(self, input_tensor):
        return self.trt_engine.infer(input_tensor)

    def output_to_class_mask(self, output):
        """
        TensorRT output을 multiclass mask로 변환한다.

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
                f"Unexpected multiclass TensorRT output shape: {out.shape}. "
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

        # 별칭 허용: yolo/decision 노드가 left/right/center로 보내도 처리
        alias = {
            "left": "left_lane",
            "right": "right_lane",
            "center": "center",
            "left_lane": "left_lane",
            "right_lane": "right_lane",
        }

        if route in alias:
            route = alias[route]

        if route not in self.valid_routes:
            self.get_logger().warn(
                f"Unknown route command: {route}. "
                f"Use one of {self.valid_routes}"
            )
            return

        # 핵심: 기존 route와 같으면 아무 로그도 찍지 않음
        if route == self.route_mode:
            return

        old_route = self.route_mode
        self.route_mode = route

        self.get_logger().info(
            f"Route mode changed: {old_route} -> {self.route_mode}"
        )

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

    def midpoint_target(self, target_a, target_b):
        """
        두 target point 사이의 중간 target을 만든다.

        target format:
            (x, y, valid)

        return:
            (mid_x, mid_y, valid)
        """
        ax, ay, av = target_a
        bx, by, bv = target_b

        if av and bv:
            mid_x = int((ax + bx) / 2)
            mid_y = int((ay + by) / 2)
            return mid_x, mid_y, True

        return 0, 0, False


    def virtual_left_lane_from_center_right(self, center_target, right_target):
        """
        left_lane 모드에서 left 차선이 안 보일 때 사용하는 fallback.

        center-right 사이의 폭을 이용해서,
        center의 왼쪽 방향에 가상의 left-center 중앙 target을 만든다.

        BEV 좌표 기준:
            x 오른쪽 증가
            left_lane target = center_x - (right_x - center_x) / 2
        """
        cx, cy, cv = center_target
        rx, ry, rv = right_target

        if cv and rv:
            half_lane_width_px = (rx - cx) / 2.0

            virtual_x = int(cx - half_lane_width_px)
            virtual_y = int((cy + ry) / 2.0)

            return virtual_x, virtual_y, True

        return 0, 0, False


    def virtual_right_lane_from_left_center(self, left_target, center_target):
        """
        right_lane 모드에서 right 차선이 안 보일 때 사용하는 fallback.

        left-center 사이의 폭을 이용해서,
        center의 오른쪽 방향에 가상의 center-right 중앙 target을 만든다.

        BEV 좌표 기준:
            x 오른쪽 증가
            right_lane target = center_x + (center_x - left_x) / 2
        """
        lx, ly, lv = left_target
        cx, cy, cv = center_target

        if lv and cv:
            half_lane_width_px = (cx - lx) / 2.0

            virtual_x = int(cx + half_lane_width_px)
            virtual_y = int((ly + cy) / 2.0)

            return virtual_x, virtual_y, True

        return 0, 0, False


    def select_target_by_route(self, left_target, center_target, right_target):
        """
        route_mode에 따라 Pure Pursuit가 따라갈 최종 target을 선택한다.

        left_lane:
            기본: left-center 사이 중앙
            fallback: left가 안 보이면 center-right 폭으로 가상 left_lane target 생성

        right_lane:
            기본: center-right 사이 중앙
            fallback: right가 안 보이면 left-center 폭으로 가상 right_lane target 생성

        center:
            기존 center class target 그대로 사용
        """
        if self.route_mode == "left_lane":
            # 1순위: 실제 left-center 사이점
            selected = self.midpoint_target(left_target, center_target)
            if selected[2]:
                return selected

            # 2순위: left가 안 보이면 center-right로 가상 left_lane 생성
            fallback = self.virtual_left_lane_from_center_right(center_target, right_target)
            if fallback[2]:
                self.get_logger().warn(
                    "left_lane fallback: left target missing, "
                    "using virtual target from center-right.",
                    throttle_duration_sec=0.5
                )
                return fallback

            return 0, 0, False

        if self.route_mode == "right_lane":
            # 1순위: 실제 center-right 사이점
            selected = self.midpoint_target(center_target, right_target)
            if selected[2]:
                return selected

            # 2순위: right가 안 보이면 left-center로 가상 right_lane 생성
            fallback = self.virtual_right_lane_from_left_center(left_target, center_target)
            if fallback[2]:
                self.get_logger().warn(
                    "right_lane fallback: right target missing, "
                    "using virtual target from left-center.",
                    throttle_duration_sec=0.5
                )
                return fallback

            return 0, 0, False

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

        색상 구분:
            차선 영역:
                left   = 어두운 빨강
                center = 어두운 초록
                right  = 어두운 파랑

            target 점:
                L target = 밝은 자홍색
                C target = 흰색
                R target = 하늘색

            selected target:
                노란색 큰 원
        """
        overlay = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)

        # =========================
        # 1. 차선 mask 색상 - 어둡게 표시
        # =========================
        overlay[bev_class_mask == 1] = (0, 0, 120)       # left lane: dark red
        overlay[bev_class_mask == 2] = (0, 120, 0)       # center lane: dark green
        overlay[bev_class_mask == 3] = (120, 0, 0)       # right lane: dark blue

        # =========================
        # 2. BEV 중앙 참고선
        # =========================
        car_center_x = int(self.bev_width // 2)
        car_y = int(self.bev_height - 1)

        cv2.line(
            overlay,
            (car_center_x, 0),
            (car_center_x, self.bev_height - 1),
            (180, 180, 180),
            1
        )
        cv2.circle(overlay, (car_center_x, car_y), 8, (180, 180, 180), -1)

        lx, ly, lv = left_target
        cx, cy, cv = center_target
        rx, ry, rv = right_target
        sx, sy, sv = selected_target

        # =========================
        # 3. L / C / R target 점 색상 - 차선 색과 다르게 표시
        # =========================
        if lv:
            # left target: magenta
            cv2.circle(overlay, (int(lx), int(ly)), 14, (255, 0, 255), -1)
            cv2.circle(overlay, (int(lx), int(ly)), 18, (255, 255, 255), 2)
            cv2.putText(
                overlay, "L", (int(lx) + 18, int(ly) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 3, cv2.LINE_AA
            )

        if cv:
            # center target: white
            cv2.circle(overlay, (int(cx), int(cy)), 14, (255, 255, 255), -1)
            cv2.circle(overlay, (int(cx), int(cy)), 18, (0, 0, 0), 2)
            cv2.putText(
                overlay, "C", (int(cx) + 18, int(cy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA
            )

        if rv:
            # right target: cyan
            cv2.circle(overlay, (int(rx), int(ry)), 14, (255, 255, 0), -1)
            cv2.circle(overlay, (int(rx), int(ry)), 18, (255, 255, 255), 2)
            cv2.putText(
                overlay, "R", (int(rx) + 18, int(ry) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3, cv2.LINE_AA
            )

        # =========================
        # 4. 최종 selected target
        #    Pure Pursuit가 실제로 따라가는 점
        # =========================
        if sv:
            cv2.circle(overlay, (int(sx), int(sy)), 26, (0, 255, 255), 4)
            cv2.circle(overlay, (int(sx), int(sy)), 5, (0, 255, 255), -1)
            cv2.putText(
                overlay, "SELECTED", (int(sx) + 28, int(sy) + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3, cv2.LINE_AA
            )

        # =========================
        # 5. 상태 텍스트
        # =========================
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

        cv2.putText(
            overlay,
            "Targets: L=magenta, C=white, R=cyan, selected=yellow",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
            # 1. TensorRT inference
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
                left_cam_mask = (class_mask_orig == 1).astype(np.uint8) * 255
                center_cam_mask = (class_mask_orig == 2).astype(np.uint8) * 255
                right_cam_mask = (class_mask_orig == 3).astype(np.uint8) * 255

                left_cam_target = self.extract_target_point_camera(left_cam_mask)
                center_cam_target = self.extract_target_point_camera(center_cam_mask)
                right_cam_target = self.extract_target_point_camera(right_cam_mask)

                camera_overlay = self.make_camera_overlay_all_targets(
                    image_bgr,
                    class_mask_orig,
                    left_cam_target,
                    center_cam_target,
                    right_cam_target,
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

    def make_camera_overlay_all_targets(
        self,
        image_bgr,
        class_mask_orig,
        left_target,
        center_target,
        right_target,
    ):
        overlay = image_bgr.copy()
        color_layer = np.zeros_like(overlay)

        color_layer[class_mask_orig == 1] = (0, 0, 255)
        color_layer[class_mask_orig == 2] = (0, 255, 0)
        color_layer[class_mask_orig == 3] = (255, 0, 0)

        mask_bool = class_mask_orig > 0
        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool],
            0.4,
            color_layer[mask_bool],
            0.6,
            0
        )

        lx, ly, lv = left_target
        cx, cy, cv = center_target
        rx, ry, rv = right_target

        if lv:
            cv2.circle(overlay, (int(lx), int(ly)), 10, (0, 0, 255), -1)
            cv2.putText(overlay, "L", (int(lx) + 10, int(ly)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if cv:
            cv2.circle(overlay, (int(cx), int(cy)), 10, (0, 255, 0), -1)
            cv2.putText(overlay, "C", (int(cx) + 10, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if rv:
            cv2.circle(overlay, (int(rx), int(ry)), 10, (255, 0, 0), -1)
            cv2.putText(overlay, "R", (int(rx) + 10, int(ry)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(
            overlay,
            f"camera targets L/C/R | route={self.route_mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay

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

    def destroy_node(self):
        if hasattr(self, "trt_engine"):
            self.trt_engine.destroy()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneMulticlassTRTNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
