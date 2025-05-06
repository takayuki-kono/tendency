import os
import shutil
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

# ディレクトリのパス
TRAIN_DIR = 'train'
VALIDATION_DIR = 'validation'
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed/validation'

# 設定
ROTATION_RANGE = 10
BATCH_SIZE = 16
img_size = 112
TARGET_NOSE_X = img_size / 2  # 鼻の目標x座標（56）
TARGET_NOSE_Y = img_size / 2  # 鼻の目標y座標（56）
X_DIFF_THRESHOLD = 0.11 * img_size / 2 # 5% of image width (5.6px)
Y_DIFF_THRESHOLD = 0.11 * img_size / 2  # 5% of image height (5.6px)

# ランドマークID
RIGHT_EYE_LANDMARKS = [7, 33, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_LANDMARKS = [13, 14]
NOSE_INDEX = 4  # 鼻
CHIN_INDEX = 152  # あご
RIGHT_CONTOUR_INDEX = 137  # 右側ほほ/輪郭
LEFT_CONTOUR_INDEX = 366  # 左側ほほ/輪郭

# MediaPipe の顔検出とランドマーク
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 顔画像の切り抜き + 輪郭チェック + 移動 + 回転 + y座標乖離チェック + グレースケール変換 + 座標表示
def preprocess_and_cut_faces(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for category in ['category1', 'category2']:
        category_input_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)

        for root, dirs, files in os.walk(category_input_dir):
            for filename in files:
                img_path = os.path.join(root, filename)
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

                        if face_image is None or face_image.size == 0:
                            print(f"Skipping {filename} due to empty face image.")
                            continue

                        # FaceMeshで鼻、あご、輪郭のランドマークを取得
                        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                        results_mesh = face_mesh.process(face_rgb)
                        
                        if results_mesh.multi_face_landmarks:
                            landmarks = results_mesh.multi_face_landmarks[0].landmark
                            nose = (landmarks[NOSE_INDEX].x * width, landmarks[NOSE_INDEX].y * height)
                            chin = (landmarks[CHIN_INDEX].x * width, landmarks[CHIN_INDEX].y * height)
                            right_contour = (landmarks[RIGHT_CONTOUR_INDEX].x * width, landmarks[RIGHT_CONTOUR_INDEX].y * height)
                            left_contour = (landmarks[LEFT_CONTOUR_INDEX].x * width, landmarks[LEFT_CONTOUR_INDEX].y * height)
                            
                            # 1. 輪郭チェック：鼻と輪郭のx座標差の差
                            nose_x = nose[0] * img_size / width
                            right_contour_x = right_contour[0] * img_size / width
                            left_contour_x = left_contour[0] * img_size / width
                            diff_right = abs(nose_x - right_contour_x)
                            diff_left = abs(nose_x - left_contour_x)
                            diff_of_diffs = abs(diff_right - diff_left)
                            
                            if diff_of_diffs >= X_DIFF_THRESHOLD:
                                print(f"Skipping {filename} due to large contour x-diff difference (Diff: {diff_of_diffs:.1f})")
                                continue
                            
                            # 2. 移動：鼻のx=56, y=56に
                            shift_x = (TARGET_NOSE_X * width / img_size) - nose[0]
                            shift_y = (TARGET_NOSE_Y * height / img_size) - nose[1]
                            M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                            shifted_image = cv2.warpAffine(face_image, M_shift, (width, height))
                            
                            # 移動後のランドマーク更新
                            nose_x = nose[0] + shift_x
                            nose_y = nose[1] + shift_y
                            chin_x = chin[0] + shift_x
                            chin_y = chin[1] + shift_y
                            nose = (nose_x, nose_y)
                            chin = (chin_x, chin_y)
                            
                            # 3. 回転：鼻-あごのx座標を揃える
                            # なし　0.61
                            # rotated_image = shifted_image
                            # あり　0.72
                            x_diff = nose_x - chin_x
                            if abs(x_diff) < 1e-2:
                                rotated_image = shifted_image
                                angle = 0.0
                            else:
                                dx = chin_x - nose_x
                                dy = chin_y - nose_y
                                angle_rad = math.atan2(dx, dy)
                                angle = -angle_rad * 180 / math.pi
                                center = (width / 2, height / 2)
                                M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
                                rotated_image = cv2.warpAffine(shifted_image, M_rotate, (width, height))

                            # 4. y座標乖離チェック：鼻とほほのy座標最大差
                            rotated_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
                            results_rotated = face_mesh.process(rotated_rgb)
                            if results_rotated.multi_face_landmarks:
                                rotated_landmarks = results_rotated.multi_face_landmarks[0].landmark
                                nose_y_rotated = rotated_landmarks[NOSE_INDEX].y * img_size
                                right_contour_y = rotated_landmarks[RIGHT_CONTOUR_INDEX].y * img_size
                                left_contour_y = rotated_landmarks[LEFT_CONTOUR_INDEX].y * img_size
                                
                                y_coords = [nose_y_rotated, right_contour_y, left_contour_y]
                                y_diff_max = max(y_coords) - min(y_coords)
                                
                                if y_diff_max >= Y_DIFF_THRESHOLD:
                                    print(f"Skipping {filename} due to large y-coordinate difference (Max Diff: {y_diff_max:.1f})")
                                    continue
                                
                                # 5. グレースケール変換、リサイズ
                                gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
                                face_image_resized = cv2.resize(gray, (img_size, img_size))
                                
                                # 6. 座標表示：右ほほ、左ほほ、あご（1px赤い点）
                                # あり　0.72 0.6
                                # なし　0.66 0.7
                                face_image_with_dots = cv2.cvtColor(face_image_resized, cv2.COLOR_GRAY2BGR)
                                landmarks_to_draw = [
                                    (RIGHT_CONTOUR_INDEX, (rotated_landmarks[RIGHT_CONTOUR_INDEX].x * img_size, rotated_landmarks[RIGHT_CONTOUR_INDEX].y * img_size)),
                                    (LEFT_CONTOUR_INDEX, (rotated_landmarks[LEFT_CONTOUR_INDEX].x * img_size, rotated_landmarks[LEFT_CONTOUR_INDEX].y * img_size)),
                                    (CHIN_INDEX, (rotated_landmarks[CHIN_INDEX].x * img_size, rotated_landmarks[CHIN_INDEX].y * img_size))
                                ]
                                
                                for idx, (x, y) in landmarks_to_draw:
                                    cv2.circle(face_image_with_dots, (int(x), int(y)), 1, (255, 0, 0), -1)  # 1px赤い点
                                
                                # 保存（座標表示付き）
                                output_path = os.path.join(category_output_dir, filename)
                                cv2.imwrite(output_path, face_image_resized)
                                # cv2.imwrite(output_path, face_image_with_dots)
                            
                            else:
                                print(f"No landmarks detected in rotated {filename}, skipping.")
                                continue
                        else:
                            print(f"No landmarks detected in {filename}, skipping.")
                else:
                    print(f"No face detected in {filename}, skipping.")

# 前処理を実行
preprocess_and_cut_faces(TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
preprocess_and_cut_faces(VALIDATION_DIR, PREPROCESSED_VALIDATION_DIR)

# CNNモデル
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), input_shape=(img_size, img_size, 1)),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3)),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

cnn_model = create_cnn_model()

# データジェネレーターと訓練
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=ROTATION_RANGE)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    PREPROCESSED_TRAIN_DIR, target_size=(img_size, img_size), batch_size=BATCH_SIZE, class_mode='sparse', color_mode='grayscale')

validation_generator = validation_datagen.flow_from_directory(
    PREPROCESSED_VALIDATION_DIR, target_size=(img_size, img_size), batch_size=BATCH_SIZE, class_mode='sparse', color_mode='grayscale')

try:
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=12)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=12, min_lr=1e-6)
    history = cnn_model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )

    train_generator = train_datagen.flow_from_directory(
        PREPROCESSED_TRAIN_DIR, target_size=(img_size, img_size),
        batch_size=BATCH_SIZE, class_mode='sparse', color_mode='grayscale', shuffle=False)

    print(f"Training accuracy: {history.history['accuracy'][-1]}")
    print(f"Validation accuracy: {max(history.history['val_accuracy'])}")

    converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")

except Exception as e:
    print(f"Error processing training: {e}")