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
import logging

# ログ設定
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(message)s')
multiple_faces_logger = logging.getLogger('multiple_faces')
multiple_faces_handler = logging.FileHandler('multiple_faces_log.txt')
multiple_faces_handler.setLevel(logging.INFO)
multiple_faces_handler.setFormatter(logging.Formatter('%(message)s'))
multiple_faces_logger.addHandler(multiple_faces_handler)

# ディレクトリのパス
TRAIN_DIR = 'train'
VALIDATION_DIR = 'validation'
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed/validation'

# 設定
VAN_RATIO = 0.35
ROTATION_RANGE = 1
img_size = 112
TARGET_NOSE_X = img_size / 2  # 鼻の目標x座標（56）
TARGET_NOSE_Y = img_size / 2  # 鼻の目標y座標（56）
Y_DIFF_THRESHOLD = VAN_RATIO * img_size / 2  # 5% of image height (11.2px)

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

# データ量をカウントする関数
def count_images(directory):
    total_images = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
    return total_images

# 顔画像の切り抜き + 頬-鼻距離チェック + 移動 + 回転 + y座標乖離チェック + グレースケール変換
def preprocess_and_cut_faces(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    skip_counters = {
        'no_face': 0, 'empty_face': 0, 'no_landmarks': 0, 'cheek_nose_distance': 0,
        'y_coordinate_diff': 0, 'no_rotated_landmarks': 0, 'small_chin_y': 0,
        'multiple_faces': 0,
        'deleted_no_face': 0, 'deleted_empty_face': 0, 'deleted_no_landmarks': 0,
        'deleted_cheek_nose_distance': 0, 'deleted_y_coordinate_diff': 0,
        'deleted_no_rotated_landmarks': 0, 'deleted_small_chin_y': 0,
        'deleted_multiple_faces': 0
    }
    total_images = 0

    for category in ['category1', 'category2']:
        category_input_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)

        for root, dirs, files in os.walk(category_input_dir):
            for filename in files:
                total_images += 1
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    skip_counters['no_face'] += 1
                    print(f"Could not read image {img_path}")
                    logging.info(f"Could not read image {img_path}")
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters['deleted_no_face'] += 1
                            print(f"Deleted {img_path} due to no_face")
                            logging.info(f"Deleted {img_path} due to no_face")
                        else:
                            logging.info(f"File {img_path} not found for deletion (no_face)")
                    except Exception as e:
                        logging.error(f"Error deleting {img_path} (no_face): {e}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_detection.process(img_rgb)

                if results.detections:
                    if len(results.detections) > 1:
                        skip_counters['multiple_faces'] += 1
                        multiple_faces_logger.info(f"Multiple faces detected in {filename}")
                        print(f"Skipping {filename} due to multiple faces detected")
                        logging.info(f"Skipping {filename} due to multiple faces detected")
                        try:
                            if os.path.exists(img_path):
                                os.remove(img_path)
                                skip_counters['deleted_multiple_faces'] += 1
                                print(f"Deleted {img_path} due to multiple_faces")
                                logging.info(f"Deleted {img_path} due to multiple_faces")
                            else:
                                logging.info(f"File {img_path} not found for deletion (multiple_faces)")
                        except Exception as e:
                            logging.error(f"Error deleting {img_path} (multiple_faces): {e}")
                        continue

                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = img.shape
                    x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    face_image = img[y:y + height, x:x + width]

                    if face_image is None or face_image.size == 0:
                        skip_counters['empty_face'] += 1
                        try:
                            if os.path.exists(img_path):
                                os.remove(img_path)
                                skip_counters['deleted_empty_face'] += 1
                                print(f"Deleted {img_path} due to empty_face")
                                logging.info(f"Deleted {img_path} due to empty_face")
                            else:
                                logging.info(f"File {img_path} not found for deletion (empty_face)")
                        except Exception as e:
                            logging.error(f"Error deleting {img_path} (empty_face): {e}")
                        continue

                    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    results_mesh = face_mesh.process(face_rgb)
                    
                    if results_mesh.multi_face_landmarks:
                        landmarks = results_mesh.multi_face_landmarks[0].landmark
                        nose = (landmarks[NOSE_INDEX].x * width, landmarks[NOSE_INDEX].y * height)
                        chin = (landmarks[CHIN_INDEX].x * width, landmarks[CHIN_INDEX].y * height)
                        right_contour = (landmarks[RIGHT_CONTOUR_INDEX].x * width, landmarks[RIGHT_CONTOUR_INDEX].y * height)
                        left_contour = (landmarks[LEFT_CONTOUR_INDEX].x * width, landmarks[LEFT_CONTOUR_INDEX].y * height)
                        
                        nose_x = nose[0] * img_size / width
                        nose_y = nose[1] * img_size / height
                        right_contour_x = right_contour[0] * img_size / width
                        right_contour_y = right_contour[1] * img_size / height
                        left_contour_x = left_contour[0] * img_size / width
                        left_contour_y = left_contour[1] * img_size / height
                        dist_right = math.sqrt((nose_x - right_contour_x)**2 + (nose_y - right_contour_y)**2)
                        dist_left = math.sqrt((nose_x - left_contour_x)**2 + (nose_y - left_contour_y)**2)
                        
                        if dist_right > 0:
                            ratio = dist_left / dist_right
                            if not (1 - VAN_RATIO <= ratio <= 1 + VAN_RATIO):
                                skip_counters['cheek_nose_distance'] += 1
                                print(f"Skipping {filename} due to cheek-nose distance imbalance (Right: {dist_right:.1f}, Left: {dist_left:.1f}, Ratio: {ratio:.3f})")
                                logging.info(f"Skipping {filename} due to cheek-nose distance imbalance (Right: {dist_right:.1f}, Left: {dist_left:.1f}, Ratio: {ratio:.3f})")
                                try:
                                    if os.path.exists(img_path):
                                        os.remove(img_path)
                                        skip_counters['deleted_cheek_nose_distance'] += 1
                                        print(f"Deleted {img_path} due to cheek_nose_distance")
                                        logging.info(f"Deleted {img_path} due to cheek_nose_distance")
                                    else:
                                        logging.info(f"File {img_path} not found for deletion (cheek_nose_distance)")
                                except Exception as e:
                                    logging.error(f"Error deleting {img_path} (cheek_nose_distance): {e}")
                                continue
                        else:
                            skip_counters['cheek_nose_distance'] += 1
                            print(f"Skipping {filename} due to zero right cheek-nose distance")
                            logging.info(f"Skipping {filename} due to zero right cheek-nose distance")
                            try:
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                                    skip_counters['deleted_cheek_nose_distance'] += 1
                                    print(f"Deleted {img_path} due to cheek_nose_distance")
                                    logging.info(f"Deleted {img_path} due to cheek_nose_distance")
                                else:
                                    logging.info(f"File {img_path} not found for deletion (cheek_nose_distance)")
                            except Exception as e:
                                logging.error(f"Error deleting {img_path} (cheek_nose_distance): {e}")
                            continue
                        
                        shift_x = (TARGET_NOSE_X * width / img_size) - nose[0]
                        shift_y = (TARGET_NOSE_Y * height / img_size) - nose[1]
                        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                        shifted_image = cv2.warpAffine(face_image, M_shift, (width, height))
                        
                        nose_x = nose[0] + shift_x
                        nose_y = nose[1] + shift_y
                        chin_x = chin[0] + shift_x
                        chin_y = chin[1] + shift_y
                        nose = (nose_x, nose_y)
                        chin = (chin_x, chin_y)
                        
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
                                skip_counters['y_coordinate_diff'] += 1
                                print(f"Skipping {filename} due to large y-coordinate difference (Max Diff: {y_diff_max:.1f})")
                                logging.info(f"Skipping {filename} due to large y-coordinate difference (Max Diff: {y_diff_max:.1f})")
                                try:
                                    if os.path.exists(img_path):
                                        os.remove(img_path)
                                        skip_counters['deleted_y_coordinate_diff'] += 1
                                        print(f"Deleted {img_path} due to y_coordinate_diff")
                                        logging.info(f"Deleted {img_path} due to y_coordinate_diff")
                                    else:
                                        logging.info(f"File {img_path} not found for deletion (y_coordinate_diff)")
                                except Exception as e:
                                    logging.error(f"Error deleting {img_path} (y_coordinate_diff): {e}")
                                continue
                            
                            if not (1 - VAN_RATIO/2 < rotated_landmarks[CHIN_INDEX].y < 1 + VAN_RATIO/2):
                                skip_counters['small_chin_y'] += 1
                                print(f"Skipping {filename} due to small chin y-coordinate (Chin Y: {chin_y:.1f})")
                                logging.info(f"Skipping {filename} due to small chin y-coordinate (Chin Y: {chin_y:.1f})")
                                try:
                                    if os.path.exists(img_path):
                                        os.remove(img_path)
                                        skip_counters['deleted_small_chin_y'] += 1
                                        print(f"Deleted {img_path} due to small_chin_y")
                                        logging.info(f"Deleted {img_path} due to small_chin_y")
                                    else:
                                        logging.info(f"File {img_path} not found for deletion (small_chin_y)")
                                except Exception as e:
                                    logging.error(f"Error deleting {img_path} (small_chin_y): {e}")
                                continue
                            
                            gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
                            face_image_resized = cv2.resize(gray, (img_size, img_size))
                            
                            face_image_with_dots = cv2.cvtColor(face_image_resized, cv2.COLOR_GRAY2BGR)
                            landmarks_to_draw = [
                                (RIGHT_CONTOUR_INDEX, (rotated_landmarks[RIGHT_CONTOUR_INDEX].x * img_size, rotated_landmarks[RIGHT_CONTOUR_INDEX].y * img_size)),
                                (LEFT_CONTOUR_INDEX, (rotated_landmarks[LEFT_CONTOUR_INDEX].x * img_size, rotated_landmarks[LEFT_CONTOUR_INDEX].y * img_size)),
                                (CHIN_INDEX, (rotated_landmarks[CHIN_INDEX].x * img_size, rotated_landmarks[CHIN_INDEX].y * img_size))
                            ]
                            
                            for idx, (x, y) in landmarks_to_draw:
                                cv2.circle(face_image_with_dots, (int(x), int(y)), 1, (255, 0, 0), -1)
                            
                            output_path = os.path.join(category_output_dir, filename)
                            cv2.imwrite(output_path, face_image_resized)
                        
                        else:
                            skip_counters['no_rotated_landmarks'] += 1
                            print(f"No landmarks detected in rotated {filename}, skipping.")
                            logging.info(f"No landmarks detected in rotated {filename}, skipping.")
                            try:
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                                    skip_counters['deleted_no_rotated_landmarks'] += 1
                                    print(f"Deleted {img_path} due to no_rotated_landmarks")
                                    logging.info(f"Deleted {img_path} due to no_rotated_landmarks")
                                else:
                                    logging.info(f"File {img_path} not found for deletion (no_rotated_landmarks)")
                            except Exception as e:
                                logging.error(f"Error deleting {img_path} (no_rotated_landmarks): {e}")
                            continue
                    else:
                        skip_counters['no_landmarks'] += 1
                        print(f"No landmarks detected in {filename}, skipping.")
                        logging.info(f"No landmarks detected in {filename}, skipping.")
                        try:
                            if os.path.exists(img_path):
                                os.remove(img_path)
                                skip_counters['deleted_no_landmarks'] += 1
                                print(f"Deleted {img_path} due to no_landmarks")
                                logging.info(f"Deleted {img_path} due to no_landmarks")
                            else:
                                logging.info(f"File {img_path} not found for deletion (no_landmarks)")
                        except Exception as e:
                            logging.error(f"Error deleting {img_path} (no_landmarks): {e}")
                        continue
                else:
                    skip_counters['no_face'] += 1
                    print(f"No face detected in {filename}, skipping.")
                    logging.info(f"No face detected in {filename}, skipping.")
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters['deleted_no_face'] += 1
                            print(f"Deleted {img_path} due to no_face")
                            logging.info(f"Deleted {img_path} due to no_face")
                        else:
                            logging.info(f"File {img_path} not found for deletion (no_face)")
                    except Exception as e:
                        logging.error(f"Error deleting {img_path} (no_face): {e}")
                    continue

    for reason, count in skip_counters.items():
        if total_images > 0:
            rate = count / total_images * 100
            print(f"{input_dir}: {reason} {count}/{total_images} ({rate:.1f}%)")
            logging.info(f"{input_dir}: {reason} {count}/{total_images} ({rate:.1f}%)")

# 前処理を実行
preprocess_and_cut_faces(TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
preprocess_and_cut_faces(VALIDATION_DIR, PREPROCESSED_VALIDATION_DIR)

# 訓練データ量をカウント
train_image_count = count_images(PREPROCESSED_TRAIN_DIR)
# 動的バッチサイズの計算
BATCH_SIZE = max(4, min(32, train_image_count // 32))
batch_count = math.ceil(train_image_count / BATCH_SIZE)
print(f"Dynamic BATCH_SIZE set to {BATCH_SIZE} for {train_image_count} training images ({batch_count} batches per epoch)")
logging.info(f"Dynamic BATCH_SIZE set to {BATCH_SIZE} for {train_image_count} training images ({batch_count} batches per epoch)")

# クラス重みの計算
def compute_class_weights(train_dir):
    class_counts = {}
    for category in ['category1', 'category2']:
        category_dir = os.path.join(train_dir, category)
        if os.path.exists(category_dir):
            class_counts[category] = len([f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))])
        else:
            class_counts[category] = 0
    
    total_images = sum(class_counts.values())
    n_classes = len(class_counts)
    
    if total_images == 0 or n_classes == 0:
        print("No training data found, using equal weights.")
        logging.info("No training data found, using equal weights.")
        return {0: 1.0, 1: 1.0}
    
    class_weights = {}
    for idx, category in enumerate(['category1', 'category2']):
        if class_counts[category] > 0:
            class_weights[idx] = total_images / (n_classes * class_counts[category])
        else:
            class_weights[idx] = 1.0
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")
    logging.info(f"Class counts: {class_counts}")
    logging.info(f"Class weights: {class_weights}")
    
    return class_weights

# カスタム損失関数
def weighted_sparse_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        weights = tf.gather([class_weights[0], class_weights[1]], y_true)
        unweighted_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        weighted_loss = unweighted_loss * weights
        return tf.reduce_mean(weighted_loss)
    return loss

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
    return model

# クラス重みの適用
train_class_weights = compute_class_weights(PREPROCESSED_TRAIN_DIR)
validation_class_weights = compute_class_weights(PREPROCESSED_VALIDATION_DIR)

# モデルコンパイル
cnn_model = create_cnn_model()
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=weighted_sparse_categorical_crossentropy(train_class_weights),
    metrics=['accuracy']
)

# データジェネレーターと訓練
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=ROTATION_RANGE)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    PREPROCESSED_TRAIN_DIR, target_size=(img_size, img_size), batch_size=BATCH_SIZE, class_mode='sparse', color_mode='grayscale')

validation_generator = validation_datagen.flow_from_directory(
    PREPROCESSED_VALIDATION_DIR, target_size=(img_size, img_size), batch_size=BATCH_SIZE, class_mode='sparse', color_mode='grayscale')

try:
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=24)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=24, min_lr=1e-6)
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
    logging.info(f"Training accuracy: {history.history['accuracy'][-1]}")
    logging.info(f"Validation accuracy: {max(history.history['val_accuracy'])}")

    converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")
    logging.info("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")

except Exception as e:
    print(f"Error processing training: {e}")
    logging.error(f"Error processing training: {e}")