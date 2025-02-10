import os
import shutil
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ディレクトリのパス
TRAIN_DIR = 'train'
VALIDATION_DIR = 'validation'
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed/validation'

# 画像サイズ設定
img_size = 112  

# MediaPipe の顔検出 + ランドマーク抽出モデルの初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# 顔のランドマーク点の選択（輪郭関連のみ）
CONTOUR_LANDMARKS = list(range(0, 17))  # 輪郭のランドマーク（0~16番）

# 距離計算用関数
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# **顔の切り抜き + ランドマーク取得**
def preprocess_and_extract_features(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    features = []
    labels = []

    for category in ['category1', 'category2']:
        category_input_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)

        for filename in os.listdir(category_input_dir):
            img_path = os.path.join(category_input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Could not read image {img_path}")
                continue

            HEIGHT, WIDTH, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = []
                    landmark_points = []

                    for idx in CONTOUR_LANDMARKS:
                        x = face_landmarks.landmark[idx].x
                        y = face_landmarks.landmark[idx].y
                        landmark_points.append((x, y))

                    # すべてのランドマークペアの距離を計算
                    for i in range(len(landmark_points)):
                        for j in range(i + 1, len(landmark_points)):
                            dist = euclidean_distance(landmark_points[i], landmark_points[j])
                            landmarks.append(dist)

                    features.append(landmarks)
                    labels.append(category)

    return np.array(features), np.array(labels)

# **前処理を実行**
X_train, y_train = preprocess_and_extract_features(TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
X_val, y_val = preprocess_and_extract_features(VALIDATION_DIR, PREPROCESSED_VALIDATION_DIR)

# ラベルエンコーディング
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

# **SVM モデル**
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)

# **ランダムフォレストモデル**
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# **特徴量の重要度を表示**
feature_importances = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

print("\n=== ランダムフォレストの特徴量の重要度 ===")
for i in sorted_indices[:10]:  # 上位10個を表示
    print(f"特徴 {i}: 重要度 {feature_importances[i]:.4f}")

# **検証データでのスコア**
rf_train_acc = rf_model.score(X_train, y_train)
rf_val_acc = rf_model.score(X_val, y_val)
print(f"ランダムフォレスト - 訓練精度: {rf_train_acc:.4f}, 検証精度: {rf_val_acc:.4f}")
