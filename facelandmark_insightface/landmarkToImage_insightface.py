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
        
        # すべてのランドマークを描画
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        
        # ラベル追加
        cv2.putText(image, 'InsightFace (All 106 Landmarks)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"InsightFace processing failed for {image_path}: {e}")
        return
    
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