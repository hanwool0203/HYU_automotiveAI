#!/usr/bin/env python3

import os
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DebugImageCaptureOnQ(Node):
    def __init__(self):
        super().__init__('debug_image_capture_on_q')

        self.bridge = CvBridge()

        self.declare_parameter('image_topic', '/lane/debug_image')
        self.declare_parameter('save_dir', '/home/ircv7/bev_debug_capture')
        self.declare_parameter('window_name', 'lane_debug_image')

        self.image_topic = self.get_parameter('image_topic').value
        self.save_dir = self.get_parameter('save_dir').value
        self.window_name = self.get_parameter('window_name').value

        os.makedirs(self.save_dir, exist_ok=True)

        self.latest_image = None
        self.frame_count = 0

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        # OpenCV 키 입력 확인용 timer
        self.timer = self.create_timer(0.03, self.keyboard_loop)

        self.get_logger().info('Debug Image Capture Node Started')
        self.get_logger().info(f'Subscribing: {self.image_topic}')
        self.get_logger().info(f'Save directory: {self.save_dir}')
        self.get_logger().info('Press q on the OpenCV window to capture and quit.')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        self.latest_image = cv_image
        self.frame_count += 1

        cv2.imshow(self.window_name, cv_image)

    def keyboard_loop(self):
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.capture_image()
            self.get_logger().info('q pressed. Capture done. Shutting down.')
            rclpy.shutdown()

    def capture_image(self):
        if self.latest_image is None:
            self.get_logger().warn('No image received yet. Nothing to save.')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'lane_debug_capture_{timestamp}.png'
        save_path = os.path.join(self.save_dir, filename)

        success = cv2.imwrite(save_path, self.latest_image)

        if success:
            self.get_logger().info(f'Saved image: {save_path}')
        else:
            self.get_logger().error(f'Failed to save image: {save_path}')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = DebugImageCaptureOnQ()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        else:
            node.destroy_node()


if __name__ == '__main__':
    main()