import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TTCRouteNode(Node):
    def __init__(self):
        super().__init__('ttc_route_node')

        self.sub = self.create_subscription(
            String,
            '/yolo/left_detections',
            self.detection_callback,
            10
        )

        self.pub = self.create_publisher(
            String,
            '/route_select',
            10
        )

        self.conf_threshold = 0.45

        # 0.0 = left edge, 0.5 = center, 1.0 = right edge
        self.center_min = 0.20
        self.center_max = 0.80

        # Bounding box height ratio threshold
        self.height_threshold = 0.25

        self.danger_count = 0
        self.required_frames = 3

        self.current_route = 'right_lane'
        self.last_danger_time = 0.0
        self.return_delay = 5.0

        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info('TTC route node started')
        self.get_logger().info('Detect box -> publish left_lane')

    def publish_route(self, route, force=False):
        if not force and route == self.current_route:
            return

        msg = String()
        msg.data = route
        self.pub.publish(msg)

        if route != self.current_route:
            self.get_logger().warn(f'Route command published: {route}')

        self.current_route = route

    def detection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            image_width = float(data.get('image_width', 640))
            image_height = float(data.get('image_height', 360))
            detections = data.get('detections', [])
        except Exception as e:
            self.get_logger().warn(f'Failed to parse YOLO JSON: {e}')
            return

        found_danger = False

        for det in detections:
            class_name = det.get('class_name', '')
            confidence = float(det.get('confidence', 0.0))

            center = det.get('center', {})
            size = det.get('size', {})

            center_x_px = float(center.get('x', 0.0))
            box_h_px = float(size.get('h', 0.0))

            center_x = center_x_px / image_width
            box_height_ratio = box_h_px / image_height

            is_front_rover = (
                class_name == 'box' and
                confidence >= self.conf_threshold and
                self.center_min <= center_x <= self.center_max and
                box_height_ratio >= self.height_threshold
            )

            if is_front_rover:
                found_danger = True

                self.get_logger().warn(
                    f'box detected | '
                    f'conf={confidence:.2f}, '
                    f'center_x={center_x:.2f}, '
                    f'height_ratio={box_height_ratio:.2f}'
                )
                break

        if found_danger:
            self.danger_count += 1
            self.last_danger_time = time.time()
        else:
            self.danger_count = 0

        if self.danger_count >= self.required_frames:
            self.publish_route('left_lane')

    def timer_callback(self):
        now = time.time()

        # Return to the original route if box is not detected for a while
        if self.current_route == 'left_lane':
            elapsed = now - self.last_danger_time
            if elapsed >= self.return_delay:
                self.publish_route('right_lane', force=True)
                return

        # 현재 route를 계속 publish
        self.publish_route(self.current_route, force=True)


def main(args=None):
    rclpy.init(args=args)
    node = TTCRouteNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
