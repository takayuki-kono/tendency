import os
import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN

# ====== 設定 ======
image_dir = "input_images"  # 例: "./images"
tolerance = 0.349 # 類似顔認識のしきい値
output_dir = "grouped_faces"

import shutil

# ====== 出力フォルダ初期化 ======
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# ====== 顔特徴抽出 ======
encodings = []
file_paths = []

from PIL import Image

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(image_dir, filename)

        # face_recognition推奨の読み込み方法を使用
        try:
            image = face_recognition.load_image_file(path)
        except Exception as e:
            print(f"読み込み失敗: {path}, 理由: {e}")
            continue

        if image.dtype != np.uint8:
            print(f"dtype不正: {path}")
            continue
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"形式不正: {path}")
            continue

        print(f"image dtype: {image.dtype}, shape: {image.shape}")

        # 顔検出と特徴抽出
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            encodings.append(encoding)
            file_paths.append(path)

# ====== クラスタリング ======
if len(encodings) == 0:
    print("顔が検出されませんでした。")
    exit()

encodings_np = np.array(encodings)
clustering = DBSCAN(metric='euclidean', eps=tolerance, min_samples=1).fit(encodings_np)
labels = clustering.labels_

# ====== グループ分け結果保存 ======
for label, path in zip(labels, file_paths):
    person_dir = os.path.join(output_dir, f"person_{label}")
    os.makedirs(person_dir, exist_ok=True)

    filename = os.path.basename(path)
    output_path = os.path.join(person_dir, filename)
    cv2.imwrite(output_path, cv2.imread(path))

print(f"グループ分け完了: {len(set(labels))} 人物グループ")
