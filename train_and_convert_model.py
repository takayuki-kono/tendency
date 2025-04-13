import os
import shutil
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import confusion_matrix, classification_report

# ディレクトリのパス
TRAIN_DIR = 'train'
VALIDATION_DIR = 'validation'
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed/validation'

# 設定
# 要調整
#  0 0.71 0.76
#  5 0.49 0.52
# 10 0.81 0.72
# 20 0.7  0.51
ROTATION_RANGE = 10
# 16 0.63
# 8 0.51 0.51
# BATCH_SIZE = 8
# 224 0.65 0.6
# 112 0.65 0.63
img_size = 112

# 右目・左目のランドマークID（MediaPipe FaceMeshの目の周辺ランドマーク）
RIGHT_EYE_LANDMARKS = [7, 33, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_LANDMARKS = [13, 14]

# MediaPipe の顔検出とランドマーク
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

yaw_angles = []
pitch_angles = []
roll_angles = []

# def get_average_landmark(face_landmarks, landmark_indices):

#     """ 指定したランドマークの平均値を取得 """
#     x_list, y_list, z_list = [], [], []

#     for idx in landmark_indices:
#         landmark = face_landmarks.landmark[idx]

#         x_list.append(landmark.x)  # 画像座標系に変換
#         y_list.append(landmark.y)
#         z_list.append(landmark.z)

#     avg_x = np.mean(x_list)
#     avg_y = np.mean(y_list)
#     avg_z = np.mean(z_list)

#     return avg_x, avg_y, avg_z

# def calculate_face_angles(mouth, right_eye, left_eye):
#     """ 右目と左目のランドマークの平均値を基準に顔の向きを計算 """

#     right_x, right_y, right_z = right_eye
#     left_x, left_y, left_z = left_eye
#     mouth_x, mouth_y, mouth_z = mouth

#     # ヨー角（左右の傾き）: 目のX座標の差を使う
#     yaw = np.arctan2(1, left_z - right_z) * 180 / np.pi

#     # ピッチ角（上下の傾き）: 目のY座標の平均とZ軸の変化
#     avg_eye_y = (right_y + left_y) / 2
#     avg_eye_z = (right_z + left_z) / 2
#     pitch = np.arctan2(1, avg_eye_z - mouth_z) * 180 / np.pi

#     # ロール角（回転）: 右目と左目の高さの差を使用
#     roll = np.arctan2(1, left_y - right_y) * 180 / np.pi

#     yaw_angles.append(yaw)
#     pitch_angles.append(pitch)
#     roll_angles.append(roll)

#     return yaw, pitch, roll

# 顔画像の切り抜き + 傾きチェック + グレースケール変換
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

                        if face_image is not None and face_image.size > 0:
                            # FaceMeshによる顔のランドマークを取得
                            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                            results_mesh = face_mesh.process(face_rgb)
                            # skip = False
                            
                            # if results_mesh.multi_face_landmarks:
                            #     for landmarks in results_mesh.multi_face_landmarks:

                            #         right_eye = get_average_landmark(landmarks, RIGHT_EYE_LANDMARKS)
                            #         left_eye = get_average_landmark(landmarks, LEFT_EYE_LANDMARKS)
                            #         mouth = get_average_landmark(landmarks, MOUTH_LANDMARKS)

                            #         # 顔の傾きを計算
                            #         yaw, pitch, roll = calculate_face_angles(mouth, right_eye, left_eye)

                            #         # 傾きが平均以下でなかったら
                            #         # 全なし 0.74 0.65　0.63
                            #         # 全あり 0.77 0.79　0.67

                            #         # yawあり 0.64 0.74 0.71
                            #         # yawなし 0.68 0.74 0.71
                            #         # if yaw <= 81 or yaw >= 99:
                            #         #     print(f"Skipping {filename} due to large yaw ({yaw:.2f} degrees).")
                            #         #     skip = True

                            #         # pitchあり 0.70 0.75
                            #         # pitchなし 0.78 0.62
                            #         # if pitch <= 81 or pitch >= 91:
                            #         #     print(f"Skipping {filename} due to large pitch ({pitch:.2f} degrees).")
                            #         #     skip = True

                            #         # rollあり 0.60 0.70
                            #         # rollなし 0.51
                            #         # if roll <= 81 or roll >= 98:
                            #         #     print(f"Skipping {filename} due to large roll ({roll:.2f} degrees).")
                            #         #     skip = True

                            # if skip:
                            #     continue
                            # グレースケール変換
                            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                            # あり 0.55 0.6
                            # なし 0.64 0.64
                            # gray = cv2.equalizeHist(gray)  # ヒストグラム平坦化
                            # # エッジ検出
                            # edges = cv2.Canny(gray, 100, 200)
                            # リサイズ
                            face_image_resized = cv2.resize(gray, (img_size, img_size))

                            output_path = os.path.join(category_output_dir, filename)
                            cv2.imwrite(output_path, face_image_resized)
                        else:
                            print(f"Skipping {filename} due to empty face image.")
                else:
                    print(f"No face detected in {filename}, skipping.")
    # print(f"angle min yaw {min(yaw_angles)}")
    # print(f"angle min pitch {min(pitch_angles)}")
    # print(f"angle min roll {min(roll_angles)}")
    # print(f"angle yaw {np.mean(yaw_angles)}")
    # print(f"angle pitch {np.mean(pitch_angles)}")
    # print(f"angle roll {np.mean(roll_angles)}")
    # print(f"angle max yaw {max(yaw_angles)}")
    # print(f"angle max pitch {max(pitch_angles)}")
    # print(f"angle max roll {max(roll_angles)}")

# 前処理を実行
preprocess_and_cut_faces(TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
preprocess_and_cut_faces(VALIDATION_DIR, PREPROCESSED_VALIDATION_DIR)

# ニューラルネットワークモデル (CNN) は変更なし
def create_cnn_model():
    print('A')
    base_model = tf.keras.applications.VGG16(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([

        # 16あり　0.76 0.74
        # 16なし　0.66 0.81 0.74
        # layers.Conv2D(16, (3,3)),
        # # layers.BatchNormalization(),
        # layers.Activation('relu'),  
        # layers.MaxPooling2D((2,2)),

        # 32あり　0.66 0.81 0.74
        # 32なし　0.66 0.61
        layers.Conv2D(32, (3,3), input_shape=(img_size, img_size, 1)),
        #BNあり　0.72 0.72 0.79 0.51
        #BNなし　0.66 0.81 0.74
        # layers.BatchNormalization(),
        layers.Activation('relu'),  
        layers.MaxPooling2D((2,2)),

        #64あり　0.66 0.81
        #64なし　0.72 0.68
        layers.Conv2D(64, (3,3)),
        # layers.BatchNormalization(),
        layers.Activation('relu'),  
        layers.MaxPooling2D((2,2)),

        #128あり　0.63 0.49
        #128なし　0.66 0.81
        # layers.Conv2D(128, (3,3)),
        # layers.BatchNormalization(),
        # layers.Activation('relu'),  
        # layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # layers.Dropout(0.5),  # 50%のニューロンを無効化して過学習防止
        layers.Dense(2, activation='softmax')
    ])

    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    from tensorflow.keras.optimizers import Adam
    # 要調整
    # 0.002 0.78
    # 0.0025 0.58
    model.compile(optimizer=Adam(learning_rate=0.002),  # 学習率を0.001に設定
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

cnn_model = create_cnn_model()

# データジェネレーター設定と訓練部分は変更なし
# データジェネレーターの作成
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # brightness_range=[0.5, 2],  # 明るさを 0.7～1.3 倍に変化
    # contrast_stretching=True,  # コントラストを変化させる
    rotation_range=ROTATION_RANGE,  # 回転
    # width_shift_range=0.2,  # 横方向のずれ
    # height_shift_range=0.2,  # 縦方向のずれ
    # shear_range=0.2,  # 斜め方向の変形
    # zoom_range=0.2,  # 拡大・縮小
    # horizontal_flip=True  # 左右反転
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    #  4 0.64
    #  8 0.63
    # 16 0.63
    PREPROCESSED_TRAIN_DIR, target_size=(img_size, img_size), batch_size=32, class_mode='sparse', color_mode='grayscale')

validation_generator = validation_datagen.flow_from_directory(
    PREPROCESSED_VALIDATION_DIR, target_size=(img_size, img_size), batch_size=32, class_mode='sparse', color_mode='grayscale')

# 訓練データのval_accuへの寄与を分析（低い順）
def analyze_train_contributions_to_val_accu(model, train_generator, img_size, top_n=10):
    """
    訓練データの各画像がval_accuに間接的に寄与する度合いを分析し、寄与率の低い順に表示。
    
    Args:
        model: 学習済みモデル
        train_generator: 訓練データのジェネレーター
        img_size: 画像のサイズ
        top_n: 表示する画像の数
    """
    contributions = []
    train_generator.shuffle = False
    train_generator.reset()
    
    total_images = len(train_generator.filenames)
    correct_predictions = 0
    
    for _ in range(len(train_generator)):
        batch = next(train_generator)
        images = batch[0]
        true_labels = batch[1].astype(int)
        
        preds = model.predict(images, verbose=0)
        
        for i in range(len(images)):
            idx = train_generator.batch_index - len(images) + i
            if idx >= total_images:
                break
            img_path = train_generator.filepaths[idx]
            true_label = true_labels[i]
            pred = preds[i]
            pred_class = np.argmax(pred)
            pred_prob = pred[true_label]
            
            is_correct = pred_class == true_label
            if is_correct:
                correct_predictions += 1
                contribution = pred_prob
            else:
                contribution = 0.0
            
            contributions.append({
                'img_path': img_path,
                'true_label': true_label,
                'pred_class': pred_class,
                'pred_prob': pred_prob,
                'contribution': contribution,
                'is_correct': is_correct
            })
    
    train_accu = correct_predictions / total_images if total_images > 0 else 0.0
    print(f"Train Prediction Accuracy: {train_accu:.4f}")
    print(f"Correct Predictions on Train: {correct_predictions}/{total_images}")
    
    # 低い順にソート（誤分類を優先、続いて低確信度の正しい予測）
    contributions = sorted(contributions, key=lambda x: x['contribution'], reverse=False)
    
    print(f"\nTop {top_n} training images with lowest contribution to model learning:")
    plt.figure(figsize=(15, 5))
    displayed = 0
    for i, contrib in enumerate(contributions):
        if displayed >= top_n:
            break
        
        img_path = contrib['img_path']
        true_label = contrib['true_label']
        pred_class = contrib['pred_class']
        contribution = contrib['contribution']
        is_correct = contrib['is_correct']
        
        try:
            img = load_img(img_path, color_mode='grayscale', target_size=(img_size, img_size))
            img = np.array(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
        
        plt.subplot(2, 5, displayed + 1)
        plt.imshow(img, cmap='gray')
        if is_correct:
            plt.title(f"Conf: {contribution:.4f}\nTrue: {true_label}, Pred: {pred_class}")
        else:
            plt.title(f"Misclassified\nTrue: {true_label}, Pred: {pred_class}")
        plt.axis('off')
        
        print(f"Image {displayed+1}: {img_path}")
        if is_correct:
            print(f"  True Label: {true_label}, Predicted: {pred_class}, Confidence: {contribution:.4f}")
        else:
            print(f"  True Label: {true_label}, Predicted: {pred_class}, Misclassified")
        
        displayed += 1
    
    plt.tight_layout()
    plt.show()
    
    # 統計情報
    misclassified = [c for c in contributions if not c['is_correct']]
    print(f"\nTotal misclassified training images: {len(misclassified)}")
    correct_low_conf = [c for c in contributions if c['is_correct'] and c['contribution'] < 0.7]
    print(f"Correct predictions with low confidence (<0.7): {len(correct_low_conf)}")

# CNN の訓練
try:
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    from tensorflow.keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)
    history = cnn_model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )

    train_generator = train_datagen.flow_from_directory(
        PREPROCESSED_TRAIN_DIR, target_size=(img_size, img_size), 
        batch_size=2, class_mode='sparse', color_mode='grayscale', shuffle=False)
    analyze_train_contributions_to_val_accu(cnn_model, train_generator, img_size, top_n=10)    
    # history = cnn_model.fit(
    #     train_generator,
    #     validation_data=validation_generator,
    #     epochs=20,
    #     callbacks=[model_checkpoint, early_stopping]
    # )

    print(f"Training accuracy: {history.history['accuracy'][-1]}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")

    converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")

except Exception as e:
    print(f"Error processing training : {e}")
