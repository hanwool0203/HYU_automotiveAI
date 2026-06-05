import os
import cv2
from cv_bridge import CvBridge

# ROS2 Bag 읽기 관련 라이브러리
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def main():
    # ==========================================
    # ⚙️ 설정 부분
    # ==========================================
    bag_path = 'bag/524_sign4'  
    storage_id = 'sqlite3' 
    topic_left = '/left/image_raw'
    topic_right = '/right/image_raw'
    
    output_dir_left = 'extracted_images/traffic_left'
    output_dir_right = 'extracted_images/sign_right'

    # 🌟 [다운샘플링 설정] 🌟
    # 추출하고 싶은 초당 프레임 수(FPS)를 설정하세요. (예: 5.0이면 0.2초 간격으로 추출)
    target_extract_fps = 2.0 
    
    # 목표 FPS를 기반으로 필요한 시간 간격을 나노초(ns) 단위로 계산합니다. (1초 = 1,000,000,000 나노초)
    extract_interval_ns = int(1_000_000_000 / target_extract_fps)
    # ==========================================

    os.makedirs(output_dir_left, exist_ok=True)
    os.makedirs(output_dir_right, exist_ok=True)

    print(f"📦 Bag 파일 읽기를 시작합니다: {bag_path}")
    print(f"🎯 타겟 추출 속도: {target_extract_fps} FPS (간격: {extract_interval_ns} ns)")

    storage_options = rosbag2_py._storage.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py._storage.ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr')

    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"❌ Bag 파일을 열 수 없습니다: {e}")
        return

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    bridge = CvBridge()
    count_l, count_r = 0, 0
    
    # 마지막으로 이미지를 저장한 시간을 기록할 변수
    last_saved_time_l = 0
    last_saved_time_r = 0

    print("🚀 다운샘플링 이미지 추출 진행 중...")

    while reader.has_next():
        # t는 메시지가 기록된 시간(나노초)입니다.
        (topic, data, t) = reader.read_next()

        if topic == topic_left:
            # 현재 메시지 시간(t)과 마지막 저장 시간의 차이가 설정한 간격보다 크거나 같을 때만 추출
            if t - last_saved_time_l >= extract_interval_ns:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                filename = os.path.join(output_dir_left, f"{count_l:06d}.jpg")
                cv2.imwrite(filename, cv_img)
                count_l += 1
                
                # 방금 저장한 시간을 업데이트
                last_saved_time_l = t

        elif topic == topic_right:
            if t - last_saved_time_r >= extract_interval_ns:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                filename = os.path.join(output_dir_right, f"{count_r:06d}.jpg")
                cv2.imwrite(filename, cv_img)
                count_r += 1
                
                last_saved_time_r = t

    print("-" * 40)
    print("✅ 다운샘플링 추출 완료!")
    print(f"왼쪽 카메라: {count_l}장")
    print(f"오른쪽 카메라: {count_r}장")
    print("-" * 40)

if __name__ == '__main__':
    main()