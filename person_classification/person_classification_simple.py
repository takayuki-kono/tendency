import os
import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
import shutil

# ====== 設定 ======
image_dir = "input_images"  # 画像が保存されているフォルダ
tolerance = 0.349           # 顔の類似度を判断するための距離のしきい値
output_dir = "grouped_faces_simple" # 分類結果の出力先フォルダ
unclassified_dir = os.path.join(output_dir, "unclassified") # 未分類フォルダ

# ====== 出力フォルダ初期化 ======
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(unclassified_dir, exist_ok=True)

# ====== 顔特徴抽出 ======
print("顔特徴の抽出を開始します...")
encodings = []
file_paths = []

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for i, filename in enumerate(image_files):
    path = os.path.join(image_dir, filename)
    print(f"[{i+1}/{len(image_files)}] 処理中: {filename}")

    try:
        # OpenCVで画像を読み込み、face_recognitionで使えるようにRGBに変換
        bgr_image = cv2.imread(path)
        if bgr_image is None:
            print(f"  -> 画像の読み込みに失敗しました。スキップします。")
            continue
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # 高速なHOGモデルで顔を検出 (CNNモデルより精度は低いが高速)
        face_locations = face_recognition.face_locations(image, model="hog")

        if not face_locations:
            print(f"  -> 顔が検出されませんでした。未分類にコピーします。")
            shutil.copy(path, os.path.join(unclassified_dir, filename))
            continue

        # 検出した顔から特徴量を抽出
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # 1つの画像に複数の顔がある場合も対応
        for encoding in face_encodings:
            encodings.append(encoding)
            file_paths.append(path) # 各特徴量に対応するファイルパスを保存
        print(f"  -> {len(face_encodings)} 個の顔を検出しました。")

    except Exception as e:
        print(f"  -> エラーが発生しました: {e}")
        continue

# ====== クラスタリング ======
# 検出された顔が2つ未満の場合はクラスタリングを行わない
if len(encodings) < 2:
    print("\nクラスタリング対象の顔が1つ以下です。処理を終了します。")
    if len(encodings) == 1:
        # 顔が1つだけの場合は、person_0として分類
        person_dir = os.path.join(output_dir, "person_0")
        os.makedirs(person_dir, exist_ok=True)
        filename = os.path.basename(file_paths[0])
        output_path = os.path.join(person_dir, filename)
        if not os.path.exists(output_path):
             shutil.copy(file_paths[0], output_path)
        print("1人の人物を分類しました。")
    exit()

print("\n顔特徴のクラスタリングを実行します (DBSCAN)...")
encodings_np = np.array(encodings)
clustering = DBSCAN(metric='euclidean', eps=tolerance, min_samples=1).fit(encodings_np)
labels = clustering.labels_

# ====== グループ分け結果保存 ======
print("\nグループ分け結果を保存します...")
copied_files = set()
for label, path in zip(labels, file_paths):
    person_dir = os.path.join(output_dir, f"person_{label}")
    os.makedirs(person_dir, exist_ok=True)

    filename = os.path.basename(path)
    output_path = os.path.join(person_dir, filename)

    if output_path not in copied_files:
        shutil.copy(path, output_path)
        copied_files.add(output_path)

num_clusters = len(set(labels))
print(f"\nグループ分け完了: {num_clusters} 人物グループに分類されました。")
print(f"結果は '{output_dir}' フォルダに保存されています。")