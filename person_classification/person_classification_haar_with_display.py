import os
import cv2
import face_recognition
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import shutil
from collections import defaultdict
import math

# ====== 設定 ======
image_dir = "input_images"  # 画像が保存されているフォルダ
tolerance = 0.35         # 顔の類似度を判断するための距離のしきい値
output_dir = "grouped_faces_haar" # 分類結果の出力先フォルダ
unclassified_dir = os.path.join(output_dir, "unclassified") # 未分類フォルダ

# Haar Cascade分類器の読み込み
# このファイルはOpenCVのインストールに含まれています
try:
    haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Haar Cascadeファイルが読み込めませんでした: {haar_cascade_path}")
except (IOError, AttributeError) as e:
    print(f"エラー: Haar Cascade分類器を読み込めません。")
    print("OpenCVが正しくインストールされているか、または 'opencv-python' パッケージを確認してください。")
    print(f"詳細: {e}")
    exit()

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
        # OpenCVで画像を読み込み
        bgr_image = cv2.imread(path)
        if bgr_image is None:
            print(f"  -> 画像の読み込みに失敗しました。スキップします。")
            continue

        # Haar Cascadeでの顔検出 (HOGの代替)
        # 1. グレースケールに変換
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # 2. 顔を検出
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 3. face_recognitionで使えるように座標形式を変換 (x, y, w, h) -> (top, right, bottom, left)
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

        if not face_locations:
            print(f"  -> 顔が検出されませんでした。未分類にコピーします。")
            shutil.copy(path, os.path.join(unclassified_dir, filename))
            continue

        # 検出した顔から特徴量を抽出 (face_recognitionの機能を利用)
        # RGB画像が必要なため、元のbgr_imageから変換する
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

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

print("\n顔特徴のクラスタリングを実行します (階層的クラスタリング)...")
encodings_np = np.array(encodings)
# 階層的クラスタリングを実行
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='euclidean',
    linkage='average',
    distance_threshold=tolerance
).fit(encodings_np)
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


# ====== グループ分け結果表示 (OpenCV) ======
def display_grouped_images_cv(labels, file_paths):
    print("\nグループ分け結果を画面に表示します...")
    grouped_images = defaultdict(list)
    for label, path in zip(labels, file_paths):
        grouped_images[label].append(path)

    if not grouped_images:
        print("表示する分類済みグループがありません。")
        return

    IMG_SIZE = 150
    COLS = 5
    key = 0
    sorted_labels = sorted(grouped_images.keys())
    for label in sorted_labels:
        image_paths = sorted(list(set(grouped_images[label])))
        num_images = len(image_paths)
        if num_images == 0: continue

        rows = math.ceil(num_images / COLS)
        montage = np.zeros((rows * IMG_SIZE, COLS * IMG_SIZE, 3), dtype=np.uint8)

        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) if img is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            row_idx, col_idx = divmod(i, COLS)
            montage[row_idx*IMG_SIZE:(row_idx+1)*IMG_SIZE, col_idx*IMG_SIZE:(col_idx+1)*IMG_SIZE] = img

        window_title = f"Person {label} ({num_images} images) - Press any key for next, ESC to quit"
        cv2.imshow(window_title, montage)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_title)
        if key == 27:
            print("表示を中断しました。")
            break

    cv2.destroyAllWindows()
    if key != 27:
        print("全てのグループの表示が完了しました。")

# 表示関数を呼び出し
display_grouped_images_cv(labels, file_paths)