import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')

        self.bridge = CvBridge()

        # ============================================================
        # 1. 이미지 크기 설정
        # ============================================================
        self.width = 640
        self.height = 360
        self.image_size = (self.width, self.height)

        # ============================================================
        # 2. 사용자가 직접 넣을 캘리브레이션 값
        # ============================================================
        # 아래 값들은 예시입니다.
        # 반드시 본인 카메라로 측정한 값으로 바꿔야 합니다.

        # 왼쪽 카메라 intrinsic
        self.K1 = np.array([
            [395.39461,   0.     , 322.36372],
            [  0.     , 396.08816, 172.76244],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)

        # 왼쪽 카메라 distortion
        self.D1 = np.array([
            -0.313120, 0.086347, -0.001000, -0.001350, 0.000000
        ], dtype=np.float64)

        # 오른쪽 카메라 intrinsic
        self.K2 = np.array([
            [396.47452,   0.     , 310.8965],
            [  0.     , 397.44982, 181.63141],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)

        # 오른쪽 카메라 distortion
        self.D2 = np.array([
            -0.326130, 0.092355, -0.000391, 0.001183, 0.000000
        ], dtype=np.float64)

        # 오른쪽 카메라가 왼쪽 카메라 기준으로 얼마나 회전했는지
        self.R = np.array([
            [0.99954904, -0.00461258, 0.02967218],
            [0.00468782, 0.99998597, -0.00246655],
            [-0.02966038, 0.00260453, 0.99955664]
        ], dtype=np.float64)

        # 오른쪽 카메라가 왼쪽 카메라 기준으로 얼마나 이동했는지
        # 예: baseline이 6cm면 0.06m
        # 일반적으로 오른쪽 카메라가 왼쪽 카메라의 +x 방향에 있으면 [-baseline, 0, 0] 형태가 많이 나옵니다.
        self.T = np.array([
            [-0.03802564],
            [-0.00080493],
            [-0.00858097]
        ], dtype=np.float64)

        # ============================================================
        # 3. StereoSGBM 파라미터
        # ============================================================
        self.min_disp = 0

        # 반드시 16의 배수
        self.num_disp = 64

        # 홀수. 3, 5, 7, 9 등
        self.block_size = 5

        # depth 시각화 최대 거리
        self.max_depth_m = 3.0

        # 너무 가까운/이상한 값 제거용
        self.min_valid_depth_m = 0.05

        # ============================================================
        # 4. Rectification map 생성
        # ============================================================
        self.create_rectification_maps()

        # ============================================================
        # 5. Stereo matcher 생성
        # ============================================================
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # ============================================================
        # 6. ROS2 QoS 설정
        # 카메라 토픽은 보통 sensor data QoS를 쓰는 게 좋습니다.
        # ============================================================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # ============================================================
        # 7. 왼쪽/오른쪽 이미지 동기화 구독
        # ============================================================
        self.left_sub = Subscriber(
            self,
            Image,
            '/left/image_raw',
            qos_profile=qos
        )

        self.right_sub = Subscriber(
            self,
            Image,
            '/right/image_raw',
            qos_profile=qos
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.image_callback)

        # ============================================================
        # 8. Publisher
        # ============================================================
        self.depth_pub = self.create_publisher(
            Image,
            '/stereo/depth/image_raw',
            qos
        )

        self.depth_vis_pub = self.create_publisher(
            Image,
            '/stereo/depth/vis',
            qos
        )

        self.overlay_pub = self.create_publisher(
            Image,
            '/stereo/depth/overlay',
            qos
        )

        self.disp_vis_pub = self.create_publisher(
            Image,
            '/stereo/disparity/vis',
            qos
        )

        self.left_rect_pub = self.create_publisher(
            Image,
            '/stereo/left_rect',
            qos
        )

        self.disp_raw_pub = self.create_publisher(
            Image,
            '/stereo/disparity/raw',
            qos
        )

        self.get_logger().info('Stereo Depth Node Started')
        self.get_logger().info('Subscribing: /left/image_raw, /right/image_raw')
        self.get_logger().info('Publishing: /stereo/depth/image_raw')
        self.get_logger().info(f'Baseline: {self.baseline_m:.4f} m')
        self.get_logger().info(f'Rectified focal length: {self.f_px:.2f} px')

    def create_rectification_maps(self):
        """
        stereoRectify를 이용해 왼쪽/오른쪽 이미지를 같은 수평 에피폴라 라인에 맞추기 위한 map 생성
        """

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K1,
            self.D1,
            self.K2,
            self.D2,
            self.image_size,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2
        self.Q = Q

        # rectified focal length
        self.f_px = float(P1[0, 0])

        # baseline
        self.baseline_m = float(np.linalg.norm(self.T))

        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(
            self.K1,
            self.D1,
            R1,
            P1,
            self.image_size,
            cv2.CV_32FC1
        )

        self.map2_x, self.map2_y = cv2.initUndistortRectifyMap(
            self.K2,
            self.D2,
            R2,
            P2,
            self.image_size,
            cv2.CV_32FC1
        )

    def image_callback(self, left_msg, right_msg):
        try:
            left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        # 이미지 크기 확인 및 리사이즈
        if left_img.shape[1] != self.width or left_img.shape[0] != self.height:
            left_img = cv2.resize(left_img, self.image_size)

        if right_img.shape[1] != self.width or right_img.shape[0] != self.height:
            right_img = cv2.resize(right_img, self.image_size)

        # ============================================================
        # 1. Rectification
        # ============================================================
        left_rect = cv2.remap(
            left_img,
            self.map1_x,
            self.map1_y,
            interpolation=cv2.INTER_LINEAR
        )

        right_rect = cv2.remap(
            right_img,
            self.map2_x,
            self.map2_y,
            interpolation=cv2.INTER_LINEAR
        )

        # ============================================================
        # 2. Gray 변환
        # ============================================================
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        # ============================================================
        # 3. Disparity 계산
        # OpenCV StereoSGBM 결과는 16배 scale된 int16
        # ============================================================
        disp_raw = self.stereo.compute(left_gray, right_gray)
        disp = disp_raw.astype(np.float32) / 16.0

        # ============================================================
        # 4. Disparity -> Depth
        # Z = f * B / d
        # ============================================================
        depth_m = np.full(disp.shape, np.nan, dtype=np.float32)

        valid = disp > 0.5
        depth_m[valid] = (self.f_px * self.baseline_m) / disp[valid]

        # 비정상 depth 제거
        depth_m[depth_m < self.min_valid_depth_m] = np.nan
        depth_m[depth_m > self.max_depth_m] = np.nan

        # ============================================================
        # 5. Depth Image publish
        # 32FC1, meter 단위
        # ============================================================
        depth_msg = self.bridge.cv2_to_imgmsg(depth_m, encoding='32FC1')
        depth_msg.header = left_msg.header
        depth_msg.header.frame_id = 'left_camera_frame'
        self.depth_pub.publish(depth_msg)

        # ============================================================
        # 6. 시각화 publish
        # ============================================================
        depth_color = self.make_depth_color(depth_m)
        overlay = cv2.addWeighted(left_rect, 0.6, depth_color, 0.4, 0)

        disp_vis = cv2.normalize(
            disp,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

        depth_vis_msg = self.bridge.cv2_to_imgmsg(depth_color, encoding='bgr8')
        depth_vis_msg.header = left_msg.header
        depth_vis_msg.header.frame_id = 'left_camera_frame'
        self.depth_vis_pub.publish(depth_vis_msg)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        overlay_msg.header = left_msg.header
        overlay_msg.header.frame_id = 'left_camera_frame'
        self.overlay_pub.publish(overlay_msg)

        disp_vis_msg = self.bridge.cv2_to_imgmsg(disp_vis, encoding='bgr8')
        disp_vis_msg.header = left_msg.header
        disp_vis_msg.header.frame_id = 'left_camera_frame'
        self.disp_vis_pub.publish(disp_vis_msg)

        left_rect_msg = self.bridge.cv2_to_imgmsg(left_rect, encoding='bgr8')
        left_rect_msg.header = left_msg.header
        left_rect_msg.header.frame_id = 'left_camera_frame'
        self.left_rect_pub.publish(left_rect_msg)

        disp_msg = self.bridge.cv2_to_imgmsg(disp.astype(np.float32), encoding='32FC1')
        disp_msg.header = left_msg.header
        disp_msg.header.frame_id = 'left_camera_frame'
        self.disp_raw_pub.publish(disp_msg)

    def make_depth_color(self, depth_m):
        """
        depth_m: meter 단위 32FC1
        가까운 곳은 밝고 강하게, 먼 곳은 약하게 보이도록 컬러맵 생성
        """
        depth_vis = depth_m.copy()

        invalid = ~np.isfinite(depth_vis)
        depth_vis[invalid] = self.max_depth_m

        depth_vis = np.clip(depth_vis, 0.0, self.max_depth_m)

        # 가까운 물체가 더 밝게 보이도록 반전
        depth_norm = (self.max_depth_m - depth_vis) / self.max_depth_m
        depth_norm = np.clip(depth_norm, 0.0, 1.0)

        depth_u8 = (depth_norm * 255.0).astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

        # invalid 영역은 검정색
        depth_color[invalid] = (0, 0, 0)

        return depth_color


def main(args=None):
    rclpy.init(args=args)

    node = StereoDepthNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()