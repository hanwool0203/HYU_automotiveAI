import cv2
import numpy as np

# 전역 변수
src_points = []

def mouse_callback(event, x, y, flags, param):
    global src_points, img_copy
    # 왼쪽 마우스 클릭 시 좌표 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < 4:
            src_points.append([x, y])
            print(f"📍 점 획득: ({x}, {y}) - 현재 {len(src_points)}/4")
            
            # 클릭한 위치에 빨간색 점과 번호 표시
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img_copy, str(len(src_points)), (x+10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Click 4 points", img_copy)
            
            if len(src_points) == 4:
                print("\n✅ 4개의 점이 모두 수집되었습니다. 아무 키나 누르면 계산을 시작합니다.")

def main():
    global img_copy
    
    # 1. 왜곡이 보정된 이미지 로드 (경로를 본인 환경에 맞게 수정하세요)
    img_path = 'undistorted_left.jpg' 
    img = cv2.imread(img_path)
    if img is None:
        print("❌ 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        return
        
    img_copy = img.copy()

    # 창 생성 및 마우스 콜백 함수 등록
    cv2.namedWindow("Click 4 points")
    cv2.setMouseCallback("Click 4 points", mouse_callback)

    print("=====================================================")
    print(" 1. 왼쪽 아래(Bottom-Left) 부터 시작하여")
    print(" 2. 시계 방향 또는 반시계 방향으로 순서대로 4개의 점을 클릭하세요.")
    print("    (예: 좌하 -> 좌상 -> 우상 -> 우하)")
    print("=====================================================")

    cv2.imshow("Click 4 points", img_copy)
    cv2.waitKey(0) # 4점 클릭 후 키 입력 대기
    cv2.destroyAllWindows()

    if len(src_points) != 4:
        print("❌ 4개의 점이 모두 선택되지 않았습니다. 프로그램을 종료합니다.")
        return

    # 2. Source Points (이미지 상의 픽셀 좌표)
    src_pts = np.float32(src_points)

    # ---------------------------------------------------------
    # 3. Destination Points (BEV 상의 목표 좌표 설정)
    # ---------------------------------------------------------
    # [주의!] 클릭한 순서와 동일한 위치에 매칭되어야 합니다.
    # 예시: 바닥에 가로 1.0m, 세로 2.0m 짜리 직사각형을 그렸고, 
    #       1픽셀을 1cm(0.01m)로 매핑하고 싶다면 -> 가로 100px, 세로 200px
    
    # BEV 이미지의 캔버스 크기 (필요에 따라 조절)
    bev_width = 1000
    bev_height = 1200
    
    # 직사각형이 그려질 BEV 캔버스 내의 위치 (중앙 하단 쯤으로 배치)
    # src_points를 [좌하, 좌상, 우상, 우하] 순서로 클릭했다고 가정
    rect_width = 590  # 1m (100 px)
    rect_height = 800 # 2m (200 px)
    
    x_offset = (bev_width - rect_width) // 2
    y_offset = bev_height - rect_height - 485 # 하단에서 50px 띄움

    dst_pts = np.float32([
        [x_offset, y_offset + rect_height],               # 좌하
        [x_offset, y_offset],                             # 좌상
        [x_offset + rect_width, y_offset],                # 우상
        [x_offset + rect_width, y_offset + rect_height]   # 우하
    ])

    # 4. 투시 변환 행렬 (Homography Matrix) 계산
    H_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print("\n✅ 투시 변환 행렬 (H) 계산 완료:\n", H_matrix)

    # 5. 결과 테스트 (원본 이미지를 BEV로 변환하여 보여주기)
    bev_result = cv2.warpPerspective(img, H_matrix, (bev_width, bev_height))
    
    cv2.imshow("Original Image", img)
    cv2.imshow("BEV Result", bev_result)
    print("\n결과 창이 떴습니다. 변환이 잘 되었는지 확인하고 아무 키나 누르세요.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6. 행렬 저장 (ROS2 노드에서 불러다 쓰기 위함)
    np.save('ipm_matrix.npy', H_matrix)
    print("\n💾 'ipm_matrix.npy' 파일로 행렬이 성공적으로 저장되었습니다!")

if __name__ == "__main__":
    main()