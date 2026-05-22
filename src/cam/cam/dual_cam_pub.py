import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np  # 화면 병합(hstack)을 위해 추가
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    """
    젯슨 나노 전용 GStreamer 파이프라인 생성 함수
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
    )

class DualCameraNode(Node):
    def __init__(self):
        super().__init__('dual_camera_node')

        # qos = QoSProfile(
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=1,
        #     reliability=ReliabilityPolicy.BEST_EFFORT,
        #     durability=DurabilityPolicy.VOLATILE
        # )
        
        # ROS2 Publisher 생성
        self.left_pub = self.create_publisher(Image, '/left/image_raw', 10)
        self.right_pub = self.create_publisher(Image, '/right/image_raw', 10)
        self.bridge = CvBridge()

        # 좌/우 카메라 OpenCV VideoCapture 객체 초기화
        self.get_logger().info("카메라 초기화 중... (sensor_id=0, 1)")
        self.cap_left = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=0), cv2.CAP_GSTREAMER)
        self.cap_right = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=0), cv2.CAP_GSTREAMER)

        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            self.get_logger().error("카메라를 열 수 없습니다. 케이블 연결이나 데몬 상태를 확인하세요.")
            raise SystemExit

        self.get_logger().info("듀얼 카메라 퍼블리싱 및 시각화 시작! (종료하려면 영상 창에서 ESC를 누르세요)")
        
        # #시각화 윈도우 생성
        # self.window_name = "Dual Camera Raw View"
        # cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # 30FPS 에 맞춰 타이머 설정 (1.0 / 30.0 초마다 콜백 실행)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

    def timer_callback(self):
        # 양쪽 카메라에서 프레임 읽기
        ret_l, frame_l = self.cap_left.read()
        ret_r, frame_r = self.cap_right.read()

        if ret_l and ret_r:
            # 스테레오 매칭을 위해 타임스탬프를 완벽히 일치시킴
            now = self.get_clock().now().to_msg()

            # OpenCV 이미지를 ROS2 Image 메시지로 변환
            msg_l = self.bridge.cv2_to_imgmsg(frame_l, encoding="bgr8")
            msg_l.header.stamp = now
            msg_l.header.frame_id = 'left_camera_frame'

            msg_r = self.bridge.cv2_to_imgmsg(frame_r, encoding="bgr8")
            msg_r.header.stamp = now
            msg_r.header.frame_id = 'right_camera_frame'

            # 토픽 발행
            self.left_pub.publish(msg_l)
            self.right_pub.publish(msg_r)

            # # ==========================================
            # # 시각화(Visualization) 코드 추가 부분
            # # ==========================================
            # # 좌/우 프레임 가로로 이어붙이기 (display_width가 640이므로 전체 너비는 1280)
            # display_frame = np.hstack((frame_l, frame_r))
            
            # # 각 카메라 화면 상단에 텍스트 표시
            # cv2.putText(display_frame, "LEFT CAMERA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(display_frame, "RIGHT CAMERA", (640 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # # 화면 출력
            # cv2.imshow(self.window_name, display_frame)

            # # 1ms 대기 및 키보드 입력 확인 (ESC 키의 아스키 코드는 27)
            # key = cv2.waitKey(1)
            # if key == 27:
            #     self.get_logger().info("ESC 키가 눌려 프로그램을 종료합니다.")
            #     raise SystemExit # ROS2 노드를 깔끔하게 종료하기 위한 예외 발생

    def destroy_node(self):
        # 노드 종료 시 카메라 및 윈도우 리소스 해제
        self.cap_left.release()
        self.cap_right.release()
        # cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DualCameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        # 터미널에서 Ctrl+C를 누르거나 창에서 ESC를 눌렀을 때 예외 처리
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()