import os
import shutil
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# ディレクトリのパス
TRAIN_DIR = 'train'
VALIDATION_DIR = 'validation'
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed/validation'

# 設定
ROTATION_RANGE = 1
BATCH_SIZE = 16
img_size = 112
NUM_SPLITS = 20
EPOCHS_PER_SPLIT = 10

# 右目・左目のランドマークID
RIGHT_EYE_LANDMARKS = [7, 33, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_LANDMARKS = [13, 14]

# MediaPipe の顔検出とランドマーク
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 顔画像の切り抜き + グレースケール変換
def preprocess_and_cut_faces(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for category in ['category1', 'category2']:
    # for category in ['category1', 'category2', 'category3', 'category4', 'category5', 'category6']:
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

                        if face_image is not None and face_image.size > 0:
                            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                            face_image_resized = cv2.resize(gray, (img_size, img_size))
                            output_path = os.path.join(category_output_dir, filename)
                            cv2.imwrite(output_path, face_image_resized)
                        else:
                            print(f"Skipping {filename} due to empty face image.")
                else:
                    print(f"No face detected in {filename}, skipping.")

# CNNモデル定義
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
        # layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 訓練データのval_accuへの寄与を分析（ランダム分割ベース）
def analyze_train_contributions_to_val_accu(train_dir, validation_generator, img_size, num_splits=NUM_SPLITS, epochs=EPOCHS_PER_SPLIT, top_n=50):
    # 訓練データのファイルリストを取得
    image_records = []
    for category in ['category1', 'category2']:
    # for category in ['category1', 'category2', 'category3', 'category4', 'category5', 'category6']:
        cat_dir = os.path.join(train_dir, category)
        if not os.path.exists(cat_dir):
            print(f"Directory {cat_dir} not found")
            continue
        for filename in os.listdir(cat_dir):
            img_path = os.path.join(cat_dir, filename)
            image_records.append({
                'path': img_path,
                'category': category,
                'val_accu_sum': 0.0,
                'count': 0
            })
    
    print(f"Found {len(image_records)} images")
    
    # データジェネレーター
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=ROTATION_RANGE)
    
    # ランダム分割と訓練
    for split_idx in range(num_splits):
        print(f"\nSplit {split_idx+1}/{num_splits}")
        
        # 各カテゴリをランダムに2分割
        group_a_paths = []
        group_b_paths = []
        for category in ['category1', 'category2']:
        # for category in ['category1', 'category2', 'category3', 'category4', 'category5', 'category6']:
            cat_files = [r['path'] for r in image_records if r['category'] == category]
            random.shuffle(cat_files)
            split_point = len(cat_files) // 2
            group_a_paths.extend(cat_files[:split_point])
            group_b_paths.extend(cat_files[split_point:])
        
        # グループAとBの訓練ディレクトリを一時的に作成
        temp_dir_a = f'temp_train_a_{split_idx}'
        temp_dir_b = f'temp_train_b_{split_idx}'
        
        for temp_dir, paths in [(temp_dir_a, group_a_paths), (temp_dir_b, group_b_paths)]:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            for cat in ['category1', 'category2']:
            # for cat in ['category1', 'category2', 'category3', 'category4', 'category5', 'category6']:
                os.makedirs(os.path.join(temp_dir, cat))
            for path in paths:
                if 'category1' in path:
                    cat = 'category1' 
                if 'category2' in path:
                    cat = 'category2' 
                # if 'category3' in path:
                #     cat = 'category3' 
                # if 'category4' in path:
                #     cat = 'category4' 
                # if 'category5' in path:
                #     cat = 'category5' 
                # if 'category6' in path:
                #     cat = 'category6' 
                shutil.copy(path, os.path.join(temp_dir, cat, os.path.basename(path)))
        
        # グループAとBで訓練
        for temp_dir, group_paths in [(temp_dir_a, group_a_paths), (temp_dir_b, group_b_paths)]:
            train_generator = train_datagen.flow_from_directory(
                temp_dir, target_size=(img_size, img_size),
                batch_size=BATCH_SIZE, class_mode='sparse', color_mode='grayscale')
            
            model = create_cnn_model()
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=[
                    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
                    ModelCheckpoint(f'best_model_split_{split_idx}.keras', monitor='val_accuracy', save_best_only=True)
                ],
                verbose=0
            )
            val_accu = max(history.history['val_accuracy'])
            print(f"  {temp_dir}: val_accu = {val_accu:.4f}")
            # if val_accu < 0.55:
            #     continue
            
            # 含まれた画像にval_accuを加算
            for img_record in image_records:                
                if img_record['path'] in group_paths:
                    img_record['val_accu_sum'] += val_accu
                    img_record['count'] += 1
        
        shutil.rmtree(temp_dir_a)
        shutil.rmtree(temp_dir_b)
    
    # 平均val_accuを計算
    contributions = []
    for img_record in image_records:
        if img_record['count'] > 0:
            avg_val_accu = img_record['val_accu_sum'] / img_record['count']
        else:
            avg_val_accu = 0.0
        contributions.append({
            'path': img_record['path'],
            'category': img_record['category'],
            'val_accu_sum': img_record['val_accu_sum'],
            'count': img_record['count'],
            'avg_val_accu': avg_val_accu
        })
    
    # 低い順にソート（avg_val_accuで）
    contributions = sorted(contributions, key=lambda x: x['val_accu_sum'], reverse=False)
    
    # トップ50をログ出力
    print(f"\nTop {min(top_n, len(contributions))} training images with lowest contribution to val_accu:")
    for i, contrib in enumerate(contributions[:top_n]):
        print(f"Image {i+1}: {contrib['path']}")
        print(f"  Category: {contrib['category']}, val_accu_sum: {contrib['val_accu_sum']:.4f}, Count: {contrib['count']}, Avg val_accu: {contrib['avg_val_accu']:.4f}")

def main():
    # 前処理
    # preprocess_and_cut_faces(TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
    # preprocess_and_cut_faces(VALIDATION_DIR, PREPROCESSED_VALIDATION_DIR)
    
    # データジェネレーター
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        PREPROCESSED_VALIDATION_DIR, target_size=(img_size, img_size),
        batch_size=BATCH_SIZE, class_mode='sparse', color_mode='grayscale')
    
    # 寄与分析
    try:
        analyze_train_contributions_to_val_accu(
            PREPROCESSED_TRAIN_DIR, validation_generator, img_size,
            num_splits=NUM_SPLITS, epochs=EPOCHS_PER_SPLIT, top_n=50)
    except Exception as e:
        print(f"Error in contribution analysis: {e}")

if __name__ == "__main__":
    main()