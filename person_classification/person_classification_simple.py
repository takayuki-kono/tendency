import os
import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
import shutil
from collections import defaultdict
import math
# ====== 設定 ======
image_dir = "input_images"  # 画像が保存されているフォルダ
tolerance = 0.35         # 顔の類似度を判断するための距離のしきい値
# 0.0 76group 71unclassified
# 0.2 78group 63unclassified
# 0.3 81group 68unclassified
# 0.4 85group 70unclassified
# 0.8 79group 68unclassified
padding_ratio = 0.4      # 画像の端を拡張する割合 (例: 0.2 = 上下左右に20%ずつ余白を追加)
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

        # ★★★ 前処理: 画像の周囲にパディングを追加 ★★★
        h, w = bgr_image.shape[:2]
        pad_h = int(h * padding_ratio)
        pad_w = int(w * padding_ratio)
        padded_bgr_image = cv2.copyMakeBorder(bgr_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
        print(f"  -> パディング追加: 元サイズ({w}, {h}) -> 新サイズ({padded_bgr_image.shape[1]}, {padded_bgr_image.shape[0]})")
        # ★★★ 前処理ここまで ★★★

        image = cv2.cvtColor(padded_bgr_image, cv2.COLOR_BGR2RGB)

        # 高速なHOGモデルで顔を検出 (CNNモデルより精度は低いが高速)
        face_locations = face_recognition.face_locations(image, model="hog") # パディングされた画像を使用

        if not face_locations:
            print(f"  -> 顔が検出されませんでした。未分類にコピーします。")
            shutil.copy(path, os.path.join(unclassified_dir, filename))
            continue

        # forループの中、face_encodingsを計算する前あたりに追加

        # --- デバッグ用: 検出した顔領域を描画して保存 ---
        debug_image = padded_bgr_image.copy()
        # face_locationsは (top, right, bottom, left) のリスト
        for (top, right, bottom, left) in face_locations:
            # OpenCVのrectangleは (left, top) と (right, bottom) の順
            cv2.rectangle(debug_image, (left, top), (right, bottom), (0, 255, 0), 2)

        # デバッグ用のファイル名を作成して保存
        debug_dir = os.path.join(output_dir, "debug_detection")
        os.makedirs(debug_dir, exist_ok=True)
        debug_filename = f"detected_{filename}"
        cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_image)
        # --- デバッグ用ここまで ---

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
    # DBSCANでノイズと判定されたもの(-1)はここでは扱わない
    if label == -1:
        continue
    person_dir = os.path.join(output_dir, f"person_{label}")
    os.makedirs(person_dir, exist_ok=True)

    filename = os.path.basename(path)
    output_path = os.path.join(person_dir, filename)

    if output_path not in copied_files:
        shutil.copy(path, output_path)
        copied_files.add(output_path)

num_clusters = len(set(l for l in labels if l != -1))
print(f"\nグループ分け完了: {num_clusters} 人物グループに分類されました。")
print(f"結果は '{output_dir}' フォルダに保存されています。")


# ====== グループ分け結果表示 (OpenCV) ======
def display_grouped_images_cv(labels, file_paths):
    """
    グループ分けされた画像をOpenCVウィンドウでグリッド表示する。
    キー操作で次のグループに進み、Escで終了できる。
    """
    print("\nグループ分け結果を画面に表示します...")

    # グループごとにファイルパスをまとめる
    grouped_images = defaultdict(list)
    for label, path in zip(labels, file_paths):
        # DBSCANでノイズ(-1)と判定されたものは表示しない
        if label != -1:
            grouped_images[label].append(path)

    # 表示するグループがない場合は終了
    if not grouped_images:
        print("表示する分類済みグループがありません。")
        return

    # 表示設定
    IMG_SIZE = 150  # 表示する各画像のサイズ
    COLS = 5        # グリッドの列数

    key = 0
    sorted_labels = sorted(grouped_images.keys())
    for label in sorted_labels:
        # グループの画像パスを重複なく取得
        image_paths = sorted(list(set(grouped_images[label])))
        num_images = len(image_paths)

        if num_images == 0:
            continue

        # グリッドの行数を計算
        rows = math.ceil(num_images / COLS)

        # グリッド表示用の大きな画像（モンタージュ）を作成
        montage = np.zeros((rows * IMG_SIZE, COLS * IMG_SIZE, 3), dtype=np.uint8)

        # 各画像を読み込み、リサイズしてモンタージュに配置
        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                cv2.putText(img, "Error", (10, IMG_SIZE // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            row_idx, col_idx = divmod(i, COLS)
            montage[row_idx*IMG_SIZE:(row_idx+1)*IMG_SIZE, col_idx*IMG_SIZE:(col_idx+1)*IMG_SIZE] = img

        window_title = f"Person {label} ({num_images} images) - Press any key for next, ESC to quit"
        cv2.imshow(window_title, montage)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_title)

        if key == 27:  # ESCキー
            print("表示を中断しました。")
            break

    cv2.destroyAllWindows()
    if key != 27:
        print("全てのグループの表示が完了しました。")

# 表示関数を呼び出し
display_grouped_images_cv(labels, file_paths)
