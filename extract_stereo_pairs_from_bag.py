import os
import cv2
import argparse
import numpy as np

import rclpy
from rclpy.serialization import deserialize_message

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

from cv_bridge import CvBridge


def get_stamp_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bag", required=True, help="ros2 bag folder path")
    parser.add_argument("--out", default="./stereo_calib_images", help="output folder")
    parser.add_argument("--left_topic", default="/left/image_raw")
    parser.add_argument("--right_topic", default="/right/image_raw")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--sync_slop", type=float, default=0.01, help="left/right max timestamp diff sec")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    left_dir = os.path.join(args.out, "left")
    right_dir = os.path.join(args.out, "right")

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    rclpy.init()

    bridge = CvBridge()

    storage_options = StorageOptions(
        uri=args.bag,
        storage_id="sqlite3"
    )

    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    if args.left_topic not in type_map:
        print(f"Left topic not found: {args.left_topic}")
        print("Available topics:")
        for t in type_map:
            print(" ", t)
        return

    if args.right_topic not in type_map:
        print(f"Right topic not found: {args.right_topic}")
        print("Available topics:")
        for t in type_map:
            print(" ", t)
        return

    left_msg_type = get_message(type_map[args.left_topic])
    right_msg_type = get_message(type_map[args.right_topic])

    latest_left = None
    latest_right = None

    saved_count = 0
    last_saved_time = -1e9
    save_interval = 1.0 / args.fps

    print("====================================")
    print("Bag:", args.bag)
    print("Left topic:", args.left_topic)
    print("Right topic:", args.right_topic)
    print("Output:", args.out)
    print("Save FPS:", args.fps)
    print("Sync slop:", args.sync_slop)
    print("====================================")

    while reader.has_next():
        topic, data, bag_time = reader.read_next()

        if topic == args.left_topic:
            msg = deserialize_message(data, left_msg_type)
            latest_left = msg

        elif topic == args.right_topic:
            msg = deserialize_message(data, right_msg_type)
            latest_right = msg

        else:
            continue

        if latest_left is None or latest_right is None:
            continue

        t_left = get_stamp_sec(latest_left)
        t_right = get_stamp_sec(latest_right)

        dt = abs(t_left - t_right)

        if dt > args.sync_slop:
            continue

        pair_time = min(t_left, t_right)

        if pair_time - last_saved_time < save_interval:
            continue

        try:
            left_img = bridge.imgmsg_to_cv2(latest_left, desired_encoding="bgr8")
            right_img = bridge.imgmsg_to_cv2(latest_right, desired_encoding="bgr8")
        except Exception as e:
            print("cv_bridge error:", e)
            continue

        if left_img.shape[1] != args.width or left_img.shape[0] != args.height:
            left_img = cv2.resize(left_img, (args.width, args.height))

        if right_img.shape[1] != args.width or right_img.shape[0] != args.height:
            right_img = cv2.resize(right_img, (args.width, args.height))

        left_path = os.path.join(left_dir, f"left_{saved_count:04d}.png")
        right_path = os.path.join(right_dir, f"right_{saved_count:04d}.png")

        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)

        print(
            f"[SAVE {saved_count:04d}] "
            f"left={t_left:.3f}, right={t_right:.3f}, dt={dt:.4f}s"
        )

        saved_count += 1
        last_saved_time = pair_time

    print()
    print("Done.")
    print("Saved pairs:", saved_count)
    print("Left images:", left_dir)
    print("Right images:", right_dir)

    rclpy.shutdown()


if __name__ == "__main__":
    main()