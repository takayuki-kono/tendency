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

# ランドマークインデックス
NOSE_TIP = 1  # 鼻の頂点
LEFT_EYE_INNER = 133  # 左目の内側
RIGHT_EYE_INNER = 362  # 右目の内側
# ご指定の輪郭インデックス（左右に分割）
LEFT_CONTOUR_INDICES = [10, 93]  # 左側輪郭
RIGHT_CONTOUR_INDICES = [152, 323]  # 右側輪郭

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
        
        # 鼻の頂点
        nose_tip = landmarks[NOSE_TIP]
        nose_x = nose_tip.x * image.shape[1]
        nose_y = nose_tip.y * image.shape[0]
        mp_landmarks.append((int(nose_x), int(nose_y)))
        
        # 左右の輪郭点から鼻の頂点にy座標が近い点（各1点）
        left_contour_points = [(i, landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in LEFT_CONTOUR_INDICES]
        right_contour_points = [(i, landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in RIGHT_CONTOUR_INDICES]
        left_y_closest = min(left_contour_points, key=lambda p: abs(p[2] - nose_y))
        right_y_closest = min(right_contour_points, key=lambda p: abs(p[2] - nose_y))
        mp_landmarks.extend([(int(left_y_closest[1]), int(left_y_closest[2])), (int(right_y_closest[1]), int(right_y_closest[2]))])
        
        # 左右の輪郭点から鼻の頂点にx座標が近い点（各1点）
        left_x_closest = min(left_contour_points, key=lambda p: abs(p[1] - nose_x))
        right_x_closest = min(right_contour_points, key=lambda p: abs(p[1] - nose_x))
        mp_landmarks.extend([(int(left_x_closest[1]), int(left_x_closest[2])), (int(right_x_closest[1]), int(right_x_closest[2]))])
        
        # 目の内側2点
        left_eye_inner = landmarks[LEFT_EYE_INNER]
        right_eye_inner = landmarks[RIGHT_EYE_INNER]
        mp_landmarks.append((int(left_eye_inner.x * image.shape[1]), int(left_eye_inner.y * image.shape[0])))
        mp_landmarks.append((int(right_eye_inner.x * image.shape[1]), int(right_eye_inner.y * image.shape[0])))
        
    except Exception as e:
        print(f"MediaPipe processing failed for {image_path}: {e}")
        return
    
    # ランドマークを描画
    for point in mp_landmarks:
        cv2.circle(image, point, 2, (0, 255, 0), -1)
    
    # ラベル追加
    cv2.putText(image, 'MediaPipe (Selected Landmarks)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
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