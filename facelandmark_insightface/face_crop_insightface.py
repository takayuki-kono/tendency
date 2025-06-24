import cv2
import numpy as np
import os
import glob
import insightface
from insightface.app import FaceAnalysis

# 初期化
try:
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    print(f"InsightFace initialization failed: {e}")
    exit(1)

# ランドマークインデックス
NOSE_TIP = 86    # 鼻の頂点
CHIN = 0         # 顎の先端
FOREHEAD_INDICES = [49, 104]  # 右眉上端、左眉上端

def process_image(image_path, output_dir):
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # InsightFaceで処理
    try:
        faces = app.get(image_rgb)
        if not faces:
            print(f"No face detected in {image_path}")
            return
        
        face = faces[0]  # 最初の顔のみ処理
        landmarks = face.landmark_2d_106  # 106点ランドマーク
        if landmarks.shape[0] < 106:
            print(f"Insufficient landmarks in {image_path}")
            return
        
        # 必要なランドマーク座標
        nose_tip = landmarks[NOSE_TIP]
        chin = landmarks[CHIN]
        
        # 眉の上端の中で最も高い（y座標が小さい）点を選択
        forehead_points = [(landmarks[i][0], landmarks[i][1]) for i in FOREHEAD_INDICES]
        forehead_y = min(forehead_points, key=lambda p: p[1])[1]
        
        # 座標を整数に変換
        nose_x = int(nose_tip[0])
        chin_y = int(chin[1])
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
        if pad_left or pad_right or pad_top or pad_bottom:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # 座標調整
            left_x += pad_left
            right_x += pad_left
            top_y += pad_top
            bottom_y += pad_top
        
        # 切り抜き
        cropped_image = image[top_y:bottom_y, left_x:right_x]
        if cropped_image.size == 0:
            print(f"Invalid crop dimensions for {image_path}")
            return
        
        # 保存
        output_path = os.path.join(output_dir, f"cropped_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, cropped_image)
        
    except Exception as e:
        print(f"InsightFace processing failed for {image_path}: {e}")
        return

if __name__ == '__main__':
    input_dir = 'input_images'
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png'))
    
    for image_path in image_paths:
        print(f"Processing {image_path}")
        process_image(image_path, output_dir)