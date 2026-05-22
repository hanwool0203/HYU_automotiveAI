import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class BevTransformNode(Node):
    def __init__(self):
        super().__init__('bev_transform_node')
        
        # 1. 파라미터 및 설정
        # 왜곡 보정된 이미지를 구독합니다.
        self.subscription = self.create_subscription(
            Image,
            '/left/undistorted_image', 
            self.image_callback,
            10)
            
        # BEV 변환된 이미지를 발행합니다.
        self.publisher_ = self.create_publisher(Image, '/left/bev_image', 10)
        self.br = CvBridge()
        
        # 2. IPM 행렬 로드 (파일명이 다를 경우 수정하세요)
        matrix_path = 'ipm_matrix_left.npy'
        if os.path.exists(matrix_path):
            self.H = np.load(matrix_path)
            self.get_logger().info(f'✅ IPM 행렬 로드 완료: {matrix_path}')
        else:
            self.get_logger().error(f'❌ 행렬 파일을 찾을 수 없습니다: {matrix_path}')
            raise FileNotFoundError

        # BEV 출력 크기 설정 (get_ipm_matrix.py에서 설정한 캔버스 크기와 동일해야 함)
        self.bev_width = 1000
        self.bev_height = 1200

    def image_callback(self, msg):
        # ROS2 Image 메시지를 OpenCV 배열로 변환
        undistorted_img = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 3. 투시 변환 (Perspective Warp) 적용
        # 이 한 줄의 연산으로 차량 중심 기준의 BEV가 생성됩니다.
        bev_img = cv2.warpPerspective(
            undistorted_img, 
            self.H, 
            (self.bev_width, self.bev_height),
            flags=cv2.INTER_LINEAR
        )
        
        # 4. 결과 발행 및 시각화
        bev_msg = self.br.cv2_to_imgmsg(bev_img, encoding='bgr8')
        self.publisher_.publish(bev_msg)
        
        # 로버의 중심축(X=500)을 가이드 라인으로 표시 (선택 사항)
        cv2.line(bev_img, (500, 0), (500, self.bev_height), (0, 255, 0), 1)
        
        cv2.imshow("Real-time BEV (base_link)", bev_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = BevTransformNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('BEV 노드 종료 중...')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()