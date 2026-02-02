import cv2
import numpy as np
import torch
from decalib.deca import DECA
from decalib.utils import util
import os
import glob

# 初期化
try:
    deca = DECA(device='cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"DECA initialization failed: {e}")
    exit(1)

def process_image(image_path, output_dir):
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # DECAで3Dランドマーク取得
    try:
        # DECAの入力形式に変換
        image_tensor = util.load_image(image_path, deca)
        codedict = deca.encode(image_tensor)
        opdict = deca.decode(codedict)
        landmarks = opdict['landmarks2d']  # 2D投影された3Dランドマーク
        fa_points = [(int(x), int(y)) for x, y in landmarks[0].cpu().numpy()]
    except Exception as e:
        print(f"DECA processing failed for {image_path}: {e}")
        return
    
    # ランドマークを描画
    for point in fa_points:
        cv2.circle(image, point, 2, (255, 0, 0), -1)
    
    # ラベル追加
    cv2.putText(image, 'DECA', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # 保存
    output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, image)

if __name__ == '__main__':
    input_dir = 'input_images'
    output_dir = 'output_images_deca'
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png'))
    
    for image_path in image_paths:
        print(f"Processing {image_path}")
        process_image(image_path, output_dir)