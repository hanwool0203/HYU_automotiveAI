# ~/workspace/ros2_ws/src/yolo_detector/yolo_detector/single_yolo_engine_node.py

import json
import time

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# ==============================
# 카메라 선택
# True  -> /left/image_raw 사용
# False -> /right/image_raw 사용
# ==============================

USE_LEFT_CAMERA = True
# USE_LEFT_CAMERA = False


CLASS_NAMES = {
    0: "left_arrow",
    1: "other_rover",
    2: "right_arrow",
    3: "slow_sign",
    4: "stop_sign",
    5: "traffic_light",
}


class SingleYoloEngineNode(Node):
    def __init__(self):
        super().__init__("single_yolo_engine_node")

        self.declare_parameter(
            "model_path",
            "/home/ircv7/workspace/ros2_ws/models/best_fp16.engine"
        )
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.70)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("process_every_n", 1)
        self.declare_parameter("publish_debug_image", True)

        self.model_path = self.get_parameter("model_path").value
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.process_every_n = int(self.get_parameter("process_every_n").value)
        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )

        if USE_LEFT_CAMERA:
            self.camera_name = "left"
            # self.image_topic = "/stereo/left_rect"
            self.image_topic = "/left/image_raw"
            self.det_topic = "/yolo/left_detections"
            self.debug_topic = "/yolo/left_debug_image"
        else:
            self.camera_name = "right"
            self.image_topic = "/right/image_raw"
            self.det_topic = "/yolo/right_detections"
            self.debug_topic = "/yolo/right_debug_image"

        self.bridge = CvBridge()

        self.get_logger().info(f"Loading TensorRT engine: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")

        self.frame_count = 0
        self.last_log_time = time.time()

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos
        )

        self.det_pub = self.create_publisher(
            String,
            self.det_topic,
            10
        )

        self.debug_pub = self.create_publisher(
            Image,
            self.debug_topic,
            10
        )

        self.get_logger().info("Single YOLO TensorRT node started")
        self.get_logger().info(f"Camera       : {self.camera_name}")
        self.get_logger().info(f"Subscribing  : {self.image_topic}")
        self.get_logger().info(f"Publishing   : {self.det_topic}")
        self.get_logger().info(f"Debug image  : {self.debug_topic}")
        self.get_logger().info(f"conf={self.conf}, iou={self.iou}, imgsz={self.imgsz}")

    def image_callback(self, msg):
        self.frame_count += 1

        if self.frame_count % self.process_every_n != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        image_h, image_w = frame.shape[:2]

        t0 = time.time()

        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )

        infer_ms = (time.time() - t0) * 1000.0

        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "camera": self.camera_name,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": score,
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    },
                    "center": {
                        "x": float((x1 + x2) / 2.0),
                        "y": float((y1 + y2) / 2.0)
                    },
                    "size": {
                        "w": float(x2 - x1),
                        "h": float(y2 - y1)
                    }
                })

        out_msg = String()
        out_msg.data = json.dumps({
            "camera": self.camera_name,
            "stamp_sec": msg.header.stamp.sec,
            "stamp_nanosec": msg.header.stamp.nanosec,
            "image_width": image_w,
            "image_height": image_h,
            "infer_ms": infer_ms,
            "detections": detections
        })

        self.det_pub.publish(out_msg)

        if self.publish_debug_image:
            debug = self.draw_debug(frame, detections, infer_ms)
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

        now = time.time()
        if now - self.last_log_time > 1.0:
            self.get_logger().info(
                f"{self.camera_name} | {len(detections)} detections | {infer_ms:.1f} ms"
            )
            self.last_log_time = now

    def draw_debug(self, frame, detections, infer_ms):
        debug = frame.copy()

        for det in detections:
            bbox = det["bbox"]
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])

            label = f'{det["class_name"]} {det["confidence"]:.2f}'

            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                debug,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        cv2.putText(
            debug,
            f"{self.camera_name} {infer_ms:.1f} ms",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        return debug


def main(args=None):
    rclpy.init(args=args)
    node = SingleYoloEngineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()