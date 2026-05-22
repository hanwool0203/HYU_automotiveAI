import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class UndistortCaptureNode(Node):
    def __init__(self):
        super().__init__('undistort_capture_node')
        
        # 1. 원본 이미지 구독 (실제 카메라 토픽 이름에 맞게 변경하세요)
        self.subscription = self.create_subscription(
            Image,
            '/right/image_raw',  # <-- 이 부분을 확인하세요!
            self.image_callback,
            10)
            
        # 2. 보정된 이미지 퍼블리셔
        self.publisher_ = self.create_publisher(Image, '/left/undistorted_image', 10)
        
        self.br = CvBridge()
        
        # # 3. left.yaml 에서 가져온 Camera Matrix (내부 파라미터)

        # self.K = np.array([
        #     [395.39461,   0.     , 322.36372],
        #     [  0.     , 396.08816, 172.76244],
        #     [  0.     ,   0.     ,   1.     ]
        # ], dtype=np.float64)
        
        # # 4. left.yaml 에서 가져온 Distortion Coefficients (왜곡 계수)
        # self.D = np.array([-0.313120, 0.086347, -0.001000, -0.001350, 0.000000], dtype=np.float64)

        self.K = np.array([
            [396.47452,   0.     , 310.8965 ],
            [  0.     , 397.44982, 181.63141],
            [  0.     ,   0.     ,   1.     ]
        ], dtype=np.float64)
        
        # 4. left.yaml 에서 가져온 Distortion Coefficients (왜곡 계수)
        self.D = np.array([-0.326130, 0.092355, -0.000391, 0.001183, 0.000000], dtype=np.float64)
        
        self.get_logger().info('📷 왜곡 보정 노드가 시작되었습니다.')
        self.get_logger().info('👉 영상 창을 선택한 상태에서 "q"를 누르면 이미지가 캡처됩니다.')

    def image_callback(self, msg):
        # ROS2 Image 메시지를 OpenCV용 NumPy 배열로 변환
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 왜곡 보정 적용 (Undistort)
        undistorted_image = cv2.undistort(cv_image, self.K, self.D)
        
        # 보정된 이미지를 ROS2 메시지로 변환하여 퍼블리시
        undistorted_msg = self.br.cv2_to_imgmsg(undistorted_image, encoding='bgr8')
        self.publisher_.publish(undistorted_msg)
        
        # 화면 시각화
        cv2.imshow("Undistorted Left Camera", undistorted_image)
        
        # 키보드 입력 대기 (1ms)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            filename = 'undistorted_camera_image.jpg'
            cv2.imwrite(filename, undistorted_image)
            self.get_logger().info(f'✅ 이미지가 캡처되었습니다! 파일명: {os.path.abspath(filename)}')
            
            # [선택 사항] 캡처 후 창을 닫고 프로그램을 종료하고 싶다면 아래 주석을 푸세요.
            # cv2.destroyAllWindows()
            # rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = UndistortCaptureNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('종료 요청을 받았습니다.')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()