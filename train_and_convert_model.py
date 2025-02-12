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
# 要調整
# 20 0.68 0.52
# 40 0.7  0.5
# 10 0.72 0.55
#  0 0.72 0.47
# 15 0.72 0.44
ROTATION_RANGE = 20
# 要調整
# 2  0.72
# 4  0.74
# 6  0.72
# 7  0.76
# 8  0.79
# 12 0.7
# 16 0.73
BATCH_SIZE = 8

# 画像サイズ設定
# 要調整
# 56  0.63 0.48
# 84  0.67 0.43
# 112 0.68 0.53
# 140 0.68 0.51
# 168 0.71 0.47
# 224 0.68 0.41
img_size = 112

# MediaPipe の顔検出
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# **顔の切り抜き + グレースケール変換 + エッジ検出**
def preprocess_and_cut_faces(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

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

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = img.shape
                    x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    face_image = img[y:y + height, x:x + width]

                    if face_image is not None and face_image.size > 0:
                        # **グレースケール変換**
                        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        # # **エッジ検出**
                        edges = cv2.Canny(gray, 100, 200)
                        # **リサイズ**
                        face_image_resized = cv2.resize(gray, (img_size, img_size))

                        output_path = os.path.join(category_output_dir, filename)
                        cv2.imwrite(output_path, face_image_resized)
                    else:
                        print(f"Skipping {filename} due to empty face image.")
            else:
                print(f"No face detected in {filename}, skipping.")

# **前処理を実行**
preprocess_and_cut_faces(TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
preprocess_and_cut_faces(VALIDATION_DIR, PREPROCESSED_VALIDATION_DIR)

# **ニューラルネットワークモデル (CNN)**
def create_cnn_model():
    base_model = tf.keras.applications.VGG16(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),  # 1チャンネル（グレースケール + エッジ）
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    from tensorflow.keras.optimizers import Adam
    # 要調整
    # 0.002 0.78
    model.compile(optimizer=Adam(learning_rate=0.001),  # 学習率を0.001に設定
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

cnn_model = create_cnn_model()

# データジェネレーターの作成
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # # 0.95 1.05
    # # 0.7  1.3  0.77 0.65
    # # 0.95  1.05  0.77 0.65
    # brightness_range=[0.5, 2],  # 明るさを 0.7～1.3 倍に変化
    # # contrast_stretching=True,  # コントラストを変化させる
    # rotation_range=ROTATION_RANGE,  # 回転
    # width_shift_range=0.2,  # 横方向のずれ
    # height_shift_range=0.2,  # 縦方向のずれ
    # shear_range=0.2,  # 斜め方向の変形
    # zoom_range=0.2,  # 拡大・縮小
    # horizontal_flip=True  # 左右反転
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    PREPROCESSED_TRAIN_DIR, target_size=(img_size, img_size), batch_size=2, class_mode='sparse', color_mode='grayscale')

validation_generator = validation_datagen.flow_from_directory(
    PREPROCESSED_VALIDATION_DIR, target_size=(img_size, img_size), batch_size=2, class_mode='sparse', color_mode='grayscale')

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

    print("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")

except Exception as e:
    print(f"Error processing training : {e}")
