import cv2
import numpy as np
import mediapipe as mp
import os
import glob

# 初期化
mp_face_mesh = mp.solutions.face_mesh
try:
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
except Exception as e:
    print(f"MediaPipe initialization failed: {e}")
    exit(1)

def process_image(image_path, output_dir):
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # MediaPipeでの処理
    try:
        mp_results = face_mesh.process(image_rgb)
        if not mp_results.multi_face_landmarks:
            print(f"No face detected in {image_path}")
            return
        
        landmarks = mp_results.multi_face_landmarks[0].landmark
        mp_landmarks = []
        
        # 全ランドマークを取得
        for landmark in landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            mp_landmarks.append((x, y))
        
    except Exception as e:
        print(f"MediaPipe processing failed for {image_path}: {e}")
        return
    
    # ランドマークを描画
    for point in mp_landmarks:
        cv2.circle(image, point, 2, (0, 255, 0), -1)
    
    # ラベル追加
    cv2.putText(image, 'MediaPipe (All Landmarks)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存
    output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, image)

if __name__ == '__main__':
    input_dir = 'input_images'
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png'))
    
    for image_path in image_paths:
        print(f"Processing {image_path}")
        process_image(image_path, output_dir)