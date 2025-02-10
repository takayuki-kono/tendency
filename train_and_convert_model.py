import os
import shutil
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
            # print(HEIGHT)

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
                        # landmarks.append(x)
                        # landmarks.append(y)

                    # すべてのランドマークペアの距離を計算
                    for i in range(len(landmark_points)):
                        for j in range(i + 1, len(landmark_points)):  # 同じペアを繰り返さない
                            dist = euclidean_distance(landmark_points[0], landmark_points[4])
                            landmarks.append(dist)  # ユークリッド距離を追加

                    # 特徴量として保存
                    features.append(landmarks)
                    # print(landmarks)
                    labels.append(category)

                    # 顔を切り抜いて保存
                    bbox = results.multi_face_landmarks[0].landmark
                    x_min = int(min([p.x for p in bbox]) * img.shape[1])
                    x_max = int(max([p.x for p in bbox]) * img.shape[1])
                    y_min = int(min([p.y for p in bbox]) * img.shape[0])
                    y_max = int(max([p.y for p in bbox]) * img.shape[0])

                    face_image = img[y_min:y_max, x_min:x_max]
                    if face_image is not None and face_image.size > 0:
                        face_image_resized = cv2.resize(face_image, (img_size, img_size))
                        output_path = os.path.join(category_output_dir, filename)
                        cv2.imwrite(output_path, face_image_resized)

    return np.array(features), np.array(labels)

# **前処理を実行**
X_train, y_train = preprocess_and_extract_features(TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
X_val, y_val = preprocess_and_extract_features(VALIDATION_DIR, PREPROCESSED_VALIDATION_DIR)

# print(X_train)
# print(y_train)
# print(X_val)
# print(y_val)
# print('from sklearn.svm import SVC')

# **機械学習モデル (SVM)**
from sklearn.svm import SVC
# print('from sklearn.model_selection import train_test_split')
from sklearn.model_selection import train_test_split
# print('from sklearn.preprocessing import LabelEncoder')
from sklearn.preprocessing import LabelEncoder
# print('LabelEncoder')

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
# print('SVC')

svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
# print('create_cnn_model')


# **ニューラルネットワークモデル (CNN)**
def create_cnn_model():
    base_model = tf.keras.applications.VGG16(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # print('create_cnn_model done')
    return model

cnn_model = create_cnn_model()

# **データジェネレーター**
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    PREPROCESSED_TRAIN_DIR, target_size=(img_size, img_size), batch_size=2, class_mode='sparse')

validation_generator = validation_datagen.flow_from_directory(
    PREPROCESSED_VALIDATION_DIR, target_size=(img_size, img_size), batch_size=2, class_mode='sparse')

# **CNN の訓練**
try:
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = cnn_model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        callbacks=[model_checkpoint, early_stopping]
    )

    print(f"Training accuracy: {history.history['accuracy'][-1]}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")

    converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

except Exception as e:
    print(f"Error processing training : {e}")
