import cv2
import numpy as np
import face_alignment
import os
import glob

# 初期化
try:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')
except Exception as e:
    print(f"face-alignment initialization failed: {e}")
    exit(1)

def process_image(image_path, output_dir):
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # face-alignmentでの処理
    try:
        fa_landmarks = fa.get_landmarks(image_rgb)
        fa_points = [(int(x), int(y)) for x, y in fa_landmarks[0]] if fa_landmarks is not None else []
    except Exception as e:
        print(f"face-alignment processing failed for {image_path}: {e}")
        return
    
    # ランドマークを描画
    for point in fa_points:
        cv2.circle(image, point, 2, (0, 0, 255), -1)
    
    # ラベル追加
    cv2.putText(image, 'face-alignment', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
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