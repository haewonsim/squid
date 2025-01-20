import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# MoveNet 모델 불러오기
model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
movenet = hub.load(model_url)

def detect_pose(image):
    input_image = tf.image.resize_with_pad(image, 192, 192)
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet.signatures["serving_default"](input_image)
    keypoints = outputs["output_0"].numpy()  # 신체 구조 키포인트 좌표
    return keypoints

# 데이터 전처리 함수
def preprocess_data(images, labels):
    processed_images = []
    for img in images:
        keypoints = detect_pose(img)
        processed_images.append(keypoints)
    return np.array(processed_images), np.array(labels)
