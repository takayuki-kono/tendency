import cv2
import numpy as np
import mediapipe as mp
import os
import glob

# 初期化
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
try:
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
except Exception as e:
    print(f"MediaPipe initialization failed: {e}")
    exit(1)

# ランドマークインデックス
NOSE_TIP = 1  # 鼻の頂点
CHIN = 152    # 顎の先端
FOREHEAD_INDICES = [105, 107, 336, 334]  # 眉の上端

def process_image(image_path, output_dir):
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. ランドマークベースの切り抜き（FaceMesh）
    try:
        mp_results = face_mesh.process(image_rgb)
        if not mp_results.multi_face_landmarks:
            print(f"No face detected by FaceMesh in {image_path}")
            return
        
        landmarks = mp_results.multi_face_landmarks[0].landmark
        
        # 必要なランドマーク座標
        nose_tip = landmarks[NOSE_TIP]
        chin = landmarks[CHIN]
        
        # 眉の上端の中で最も高い（y座標が小さい）点を選択
        forehead_points = [(landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in FOREHEAD_INDICES]
        forehead_y = min(forehead_points, key=lambda p: p[1])[1]
        
        # 座標をピクセルに変換
        nose_x = int(nose_tip.x * image.shape[1])
        chin_y = int(chin.y * image.shape[0])
        forehead_y = int(forehead_y)
        
        # 顔の幅（鼻の頂点を中心に上端と下端の半分の距離）
        face_height = abs(chin_y - forehead_y)
        half_height = face_height // 2
        
        # 切り抜き範囲
        left_x = nose_x - half_height
        right_x = nose_x + half_height
        top_y = forehead_y
        bottom_y = chin_y
        
        # 画像拡張（画面端を超えた場合）
        pad_left = max(0, -left_x)
        pad_right = max(0, right_x - image.shape[1])
        pad_top = max(0, -top_y)
        pad_bottom = max(0, bottom_y - image.shape[0])
        
        # パディング追加
        padded_image = image.copy()
        if pad_left or pad_right or pad_top or pad_bottom:
            padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # 座標調整
            left_x += pad_left
            right_x += pad_left
            top_y += pad_top
            bottom_y += pad_top
        
        # ランドマークベースの切り抜き
        cropped_image = padded_image[top_y:bottom_y, left_x:right_x]
        if cropped_image.size == 0:
            print(f"Invalid crop dimensions for FaceMesh in {image_path}")
            return
        
        # ランドマーク切り抜き画像を保存
        output_path = os.path.join(output_dir, f"cropped_facemesh_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, cropped_image)
        
    except Exception as e:
        print(f"FaceMesh processing failed for {image_path}: {e}")
        return
    
    # 2. バウンディングボックスベースの切り抜き（FaceDetection）
    try:
        fd_results = face_detection.process(image_rgb)
        if not fd_results.detections:
            print(f"No face detected by FaceDetection in {image_path}")
            return
        
        detection = fd_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        
        # バウンディングボックスの座標をピクセルに変換
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        x_max = x_min + width
        y_max = y_min + height
        
        # バウンディングボックスベースの切り抜き
        pad_left_fd = max(0, -x_min)
        pad_right_fd = max(0, x_max - image.shape[1])
        pad_top_fd = max(0, -y_min)
        pad_bottom_fd = max(0, y_max - image.shape[0])
        
        # パディング追加
        padded_image_fd = image.copy()
        if pad_left_fd or pad_right_fd or pad_top_fd or pad_bottom_fd:
            padded_image_fd = cv2.copyMakeBorder(image, pad_top_fd, pad_bottom_fd, pad_left_fd, pad_right_fd, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            x_min += pad_left_fd
            x_max += pad_left_fd
            y_min += pad_top_fd
            y_max += pad_top_fd
        
        # バウンディングボックスで切り抜き
        cropped_image_fd = padded_image_fd[y_min:y_max, x_min:x_max]
        if cropped_image_fd.size == 0:
            print(f"Invalid crop dimensions for FaceDetection in {image_path}")
            return
        
        # バウンディングボックス切り抜き画像を保存
        output_path_fd = os.path.join(output_dir, f"cropped_facedetection_{os.path.basename(image_path)}")
        cv2.imwrite(output_path_fd, cropped_image_fd)
        
    except Exception as e:
        print(f"FaceDetection processing failed for {image_path}: {e}")
        return

if __name__ == '__main__':
    input_dir = 'input_images'
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png'))
    
    for image_path in image_paths:
        print(f"Processing {image_path}")
        process_image(image_path, output_dir)