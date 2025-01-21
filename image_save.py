import cv2
import numpy as np
import mediapipe as mp
import os

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 데이터 저장 경로 및 동작 정의
DATA_PATH = 'C:/Users/Admin/Desktop/motion_data'  # 여유 공간이 충분한 드라이브로 변경
os.makedirs(DATA_PATH, exist_ok=True)

actions = ['팔 벌리기', '양손 앞으로', '한손 올리기']
num_samples = 100  # 각 동작당 수집할 샘플 수

def collect_data(action, samples=100):
    cap = cv2.VideoCapture(0)  # 카메라 활성화
    collected_data = []

    print(f"'{action}' 데이터를 수집합니다. 준비하세요!")

    # 저장 디렉토리 생성
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

    for i in range(samples):
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("프레임이 유효하지 않습니다. (frame: None or empty)")
            continue

        # MediaPipe로 관절(Keypoints) 추출
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 관절 데이터가 있을 때만 저장
        if results.pose_landmarks:
            keypoints = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
            collected_data.append(keypoints)

            # 포즈 포인트를 현재 프레임에 그리기
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # 이미지 저장
            img_file_name = f"{action}_{i+1:03d}.jpg"  # 확장자를 .png로 변경 가능
            img_file_path = os.path.join(action_path, img_file_name)
            
            # 저장 결과 확인
            save_result = cv2.imwrite(img_file_path, frame)
            if save_result:
                print(f"이미지 저장 성공: {img_file_path}")
            else:
                print(f"이미지 저장 실패: {img_file_path}, 프레임 크기: {frame.shape}")

        # 현재 프레임 표시
        cv2.putText(frame, f"{action} 데이터 수집 중: {i+1}/{samples}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 관절 데이터 저장
    if collected_data:
        npy_file_name = f"{action}_keypoints.npy"
        npy_file_path = os.path.join(DATA_PATH, npy_file_name)
        np.save(npy_file_path, np.array(collected_data))
        print(f"'{npy_file_name}' 파일 저장 완료!")


# 각 동작에 대해 데이터 수집
for action in actions:
    collect_data(action, num_samples)

# MediaPipe Pose 객체 종료
pose.close()
