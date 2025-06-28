import os
import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN

# ====== 設定 ======
image_dir = "input_images"  # 例: "./images"
tolerance = 0.349 # 類似顔認識のしきい値
output_dir = "grouped_faces" # 分類済みフォルダ
unclassified_dir = os.path.join(output_dir, "unclassified") # 未分類フォルダ

import shutil

# ====== 出力フォルダ初期化 ======
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(unclassified_dir, exist_ok=True)

# ====== 顔特徴抽出 ======
encodings = []
file_paths = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(image_dir, filename)

        # OpenCVで画像を読み込み、BGRからRGBに変換します。
        # face_recognitionライブラリはRGB順の画像を期待するためです。
        print(f"{filename}")
        bgr_image = cv2.imread(path)
        if bgr_image is None:
            print(f"画像の読み込みに失敗しました: {path}")
            continue
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if image.dtype != np.uint8:
            print(f"dtype不正: {path}")
            continue

        print(f"image dtype: {image.dtype}, shape: {image.shape}")

        # 顔検出と特徴抽出
        # HOGモデル(デフォルト)よりも高精度なCNNモデルを使用します。
        # 処理に時間がかかりますが、検出率が向上します。
        # GPUが利用可能な環境では高速に動作します。
        face_locations = face_recognition.face_locations(image, model="cnn")

        if not face_locations:
            print(f"顔が検出されませんでした: {path}")
            shutil.copy(path, os.path.join(unclassified_dir, filename))
            continue

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
    # 元のファイルをコピーする方が効率的です。
    # cv2.imwrite(output_path, cv2.imread(path))
    shutil.copy(path, output_path)

print(f"グループ分け完了: {len(set(labels))} 人物グループ")
