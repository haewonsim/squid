import scipy.io
import os
import numpy as np
import cv2

# 데이터셋 경로 설정
dataset_path = "EgoHands"

# .mat 파일 로드
annotations = scipy.io.loadmat(os.path.join(dataset_path, 'egohands_labels.mat'))

# 이미지 경로 및 주석 추출 예제
for idx, img_data in enumerate(annotations['video']):
    img_path = os.path.join(dataset_path, img_data[0]['path'][0])
    masks = img_data[0]['label'][0]  # 마스크 데이터
    print(f"Processing {img_path}")

    # 이미지 불러오기
    img = cv2.imread(img_path)
    # 마스크 적용 및 시각화 예제
    for mask in masks:
        cv2.addWeighted(img, 0.8, mask, 0.2, 0)
