from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='cam',
            executable='dual_cam_pub_node',
            name='dual_cam_pub_node',
            output='screen',
        ),

        Node(
            package='cam',
            executable='lane_seg_node',
            name='lane_seg_node',
            output='screen',
        ),

        Node(
            package='cam',
            executable='stereo_depth_node',
            name='stereo_depth_node',
            output='screen',
        ),

        Node(
            package='cam',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
        ),
    ])