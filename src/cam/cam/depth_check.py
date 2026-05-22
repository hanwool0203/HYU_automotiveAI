import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class DepthClickChecker(Node):
    def __init__(self):
        super().__init__('depth_click_checker')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_overlay = None
        self.latest_disp = None

        self.window_name = "Click sign point - depth checker"

        self.depth_sub = self.create_subscription(
            Image,
            '/stereo/depth/image_raw',
            self.depth_callback,
            qos
        )

        self.overlay_sub = self.create_subscription(
            Image,
            '/stereo/depth/overlay',
            self.overlay_callback,
            qos
        )

        self.disp_sub = self.create_subscription(
            Image,
            '/stereo/disparity/raw',
            self.disp_callback,
            qos
        )

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.timer = self.create_timer(0.03, self.show_image)

        self.get_logger().info("Depth Click Checker Started")
        self.get_logger().info("Click on /stereo/depth/overlay image")
        self.get_logger().info("Press q in OpenCV window to quit")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='32FC1'
            )
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def disp_callback(self, msg):
        try:
            self.latest_disp = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='32FC1'
            )
        except Exception as e:
            self.get_logger().error(f"Disparity conversion failed: {e}")

    def overlay_callback(self, msg):
        try:
            self.latest_overlay = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f"Overlay conversion failed: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.latest_depth is None:
            self.get_logger().warn("Depth image is not received yet")
            return

        h, w = self.latest_depth.shape

        if not (0 <= x < w and 0 <= y < h):
            self.get_logger().warn(f"Clicked point out of range: ({x}, {y})")
            return

        # 클릭한 한 픽셀의 depth
        z = self.latest_depth[y, x]

        if self.latest_disp is not None:
            d = self.latest_disp[y, x]
        else:
            d = np.nan

        # 주변 영역 median depth도 같이 계산
        # 한 픽셀은 노이즈가 심할 수 있어서 ROI median이 더 안정적임
        roi_size = 15
        half = roi_size // 2

        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)

        roi = self.latest_depth[y1:y2, x1:x2]
        roi_median = np.nanmedian(roi)

        if self.latest_disp is not None:
            disp_roi = self.latest_disp[y1:y2, x1:x2]
            disp_roi_median = np.nanmedian(disp_roi)
        else:
            disp_roi_median = np.nan

        if np.isfinite(z):
            pixel_text = f"{z:.3f} m"
        else:
            pixel_text = "invalid"

        if np.isfinite(roi_median):
            roi_text = f"{roi_median:.3f} m"
        else:
            roi_text = "invalid"

        if np.isfinite(d) and d > 0:
            disp_text = f"{d:.2f} px"
        else:
            disp_text = "invalid"

        if np.isfinite(disp_roi_median) and disp_roi_median > 0:
            disp_roi_text = f"{disp_roi_median:.2f} px"
        else:
            disp_roi_text = "invalid"

        self.get_logger().info(
            f"Clicked pixel ({x}, {y}) | "
            f"pixel depth: {pixel_text} | "
            f"{roi_size}x{roi_size} depth median: {roi_text} | "
            f"pixel disparity: {disp_text} | "
            f"{roi_size}x{roi_size} disparity median: {disp_roi_text}"
        )

        # 화면에도 표시하기 위해 저장
        self.last_click = {
            "x": x,
            "y": y,
            "pixel_text": pixel_text,
            "roi_text": roi_text,
            "disp_text": disp_text,
            "disp_roi_text": disp_roi_text,
            "roi": (x1, y1, x2, y2)
        }

    def show_image(self):
        if self.latest_overlay is None:
            return

        vis = self.latest_overlay.copy()

        if hasattr(self, "last_click"):
            x = self.last_click["x"]
            y = self.last_click["y"]
            pixel_text = self.last_click["pixel_text"]
            roi_text = self.last_click["roi_text"]
            disp_text = self.last_click["disp_text"]
            disp_roi_text = self.last_click["disp_roi_text"]
            x1, y1, x2, y2 = self.last_click["roi"]

            cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                vis,
                f"pixel ({x},{y}): {pixel_text}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                vis,
                f"ROI median: {roi_text}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                vis,
                f"Disparity: {disp_text}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            cv2.putText(
                vis,
                f"ROI disparity median: {disp_roi_text}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        cv2.imshow(self.window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DepthClickChecker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()