import os
import shutil
from timeit import repeat
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import math
import pathlib
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# try:
#     tf.get_logger().setLevel(tf.get_log('ERROR'))  # 'ERROR'で情報メッセージを非表示に
# except Exception as e:
#     print(f"Error processing training : {e}")

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
ROTATION_RANGE = 10
# 要調整
# 2 0.69 0.63
BATCH_SIZE = 2

# 要調整
# 56  0.63 0.48
# 84  0.67 0.43
# 112 0.68 0.53
# 140 0.68 0.51
# 168 0.71 0.47
# 224 0.68 0.41
img_size = 112

# MediaPipeの顔検出モデルの初期化
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# 顔の切り抜きとリサイズの処理
def preprocess_and_cut_faces(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in ['category1', 'category2']:
        category_input_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)

        for filename in os.listdir(category_input_dir):
            try:
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(category_input_dir, filename)
                    img = cv2.imread(img_path)

                    if img is None:
                        print(f"Could not read image {img_path}")
                        continue

                    # MediaPipeを使用して顔を検出
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(img_rgb)

                    if results.detections:
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            h, w, _ = img.shape
                            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                            # 顔の部分を切り抜いてリサイズ
                            face_image = img[y:y + height, x:x + width]

                            # face_image_resized = cv2.resize(face_image, (img_size, img_size))

                            # # 保存
                            # output_path = os.path.join(category_output_dir, filename)
                            # cv2.imwrite(output_path, face_image_resized)
                            if face_image is not None and face_image.size > 0:
                                face_image_resized = cv2.resize(face_image, (img_size, img_size))
                                output_path = os.path.join(category_output_dir, filename)
                                cv2.imwrite(output_path, face_image_resized)
                            else:
                                print(f"Skipping {filename} due to empty face image.")
                    else:
                        print(f"No face detected in {filename}, skipping.")
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

# 前処理を実行
preprocess_and_cut_faces(TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
preprocess_and_cut_faces(VALIDATION_DIR, PREPROCESSED_VALIDATION_DIR)

# モデルの構築
def create_model():

    from tensorflow.keras.applications import VGG16, ResNet50, MobileNet

    # 要調整
    # # VGG16 0.68 0.63
    base_model = tf.keras.applications.VGG16(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')

    # # ResNet50 0.54 0.44
    # base_model = tf.keras.applications.ResNet50(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')

    # InceptionV3 299 0.65 0.5
    # base_model = tf.keras.applications.InceptionV3(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')

    # # MobileNetV2 0.73 0.5
    # base_model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')

    # # DenseNet121 0.67 0.55
    # base_model = tf.keras.applications.DenseNet121(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')

    # # EfficientNetB0 0.49 0.47
    # base_model = tf.keras.applications.EfficientNetB0(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')

    base_model.trainable = False  # 転移学習のため、ベースモデルは固定
    # base_model.trainable = True
    # for layer in base_model.layers[:-20]:  # 最後の20層以外を凍結
    #     layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(2, activation='softmax')  # カテゴリ数は2
    ])

    # 要調整
    # 1. Adamオプティマイザ 0.67
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=0.001),  # 学習率を0.001に設定
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 2. SGDオプティマイザ 0.57
    # from tensorflow.keras.optimizers import SGD
    # model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),  # 学習率を0.01、モーメンタム0.9
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    # 3. RMSpropオプティマイザ 0.62

    # from tensorflow.keras.optimizers import RMSprop
    # model.compile(optimizer=RMSprop(learning_rate=0.0005),  # 学習率を0.0005に設定
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])


    # 4. Adagradオプティマイザ 0.57
    # from tensorflow.keras.optimizers import Adagrad
    # model.compile(optimizer=Adagrad(learning_rate=0.01),  # 学習率を0.01
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    # 5. 学習率のスケジューリング
    # from tensorflow.keras.callbacks import LearningRateScheduler
    # def scheduler(epoch, lr):
    #     if epoch < 10:
    #         return lr
    #     else:
    #         return lr * tf.math.exp(-0.1)  # エポックが進むごとに学習率を減少
    # lr_scheduler = LearningRateScheduler(scheduler)
    # history = model.fit(train_generator,
    #                     validation_data=validation_generator,
    #                     epochs=30,
    #                     callbacks=[lr_scheduler])

    return model

# データジェネレーターの作成
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # 0.95 1.05
    # 0.7  1.3  0.77 0.65
    # 0.95  1.05  0.77 0.65
    brightness_range=[1, 1],  # 明るさを 0.7～1.3 倍に変化
    # contrast_stretching=True,  # コントラストを変化させる
    rotation_range=ROTATION_RANGE,  # 回転
    width_shift_range=0.2,  # 横方向のずれ
    height_shift_range=0.2,  # 縦方向のずれ
    shear_range=0.2,  # 斜め方向の変形
    zoom_range=0.2,  # 拡大・縮小
    horizontal_flip=True  # 左右反転
)

# validation_datagen = ImageDataGenerator(
#     rescale=1./255,
#     brightness_range=[0.7, 1.3],  # 明るさを 0.7～1.3 倍に変化
#     # contrast_stretching=True,  # コントラストを変化させる
#     rotation_range=ROTATION_RANGE,  # 回転
#     width_shift_range=0.2,  # 横方向のずれ
#     height_shift_range=0.2,  # 縦方向のずれ
#     shear_range=0.2,  # 斜め方向の変形
#     zoom_range=0.2,  # 拡大・縮小
#     horizontal_flip=True  # 左右反転
# )

validation_datagen = ImageDataGenerator(rescale=1./255)  # 検証データは変更しない

train_generator = train_datagen.flow_from_directory(
    PREPROCESSED_TRAIN_DIR,
    target_size=(img_size, img_size),
    # 要調整
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

validation_generator = validation_datagen.flow_from_directory(
    PREPROCESSED_VALIDATION_DIR,
    target_size=(img_size, img_size),
    # 要調整
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

# モデルの訓練
try:
    model = create_model()

    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=12)

    # # `repeat()`を使って無限ループを設定
    # train_dataset = tf.data.Dataset.from_generator(
    #     lambda: train_generator,
    #     output_types=(tf.float32, tf.float32)
    # ).repeat()

    # validation_dataset = tf.data.Dataset.from_generator(
    #     lambda: validation_generator,
    #     output_types=(tf.float32, tf.float32)
    # ).repeat()

    # # モデルの学習
    # history = model.fit(
    #     train_dataset,
    #     steps_per_epoch=math.ceil(train_generator.samples / train_generator.batch_size),
    #     validation_data=validation_dataset,
    #     validation_steps=math.ceil(validation_generator.samples / validation_generator.batch_size),
    #     epochs=20,
    #     callbacks=[model_checkpoint, early_stopping]
    # )

    print("Samples in train generator:", train_generator.samples)
    print("Batch size in train generator:", train_generator.batch_size)
    print("Steps per epoch:", math.ceil(train_generator.samples / train_generator.batch_size))

    history = model.fit(
        train_generator,
        # steps_per_epoch=math.ceil(train_generator.samples // train_generator.batch_size),
        validation_data=validation_generator,
        # validation_steps=math.ceil(validation_generator.samples // validation_generator.batch_size),
        # validation_steps=math.ceil(validation_generator.samples / validation_generator.batch_size),
        # 要調整
        epochs=20,
        callbacks=[model_checkpoint, early_stopping]
    )

    # モデルの評価
    print(f"Training accuracy: {history.history['accuracy'][-1]}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")

    # TensorFlow Lite形式で保存
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")

except Exception as e:
    print(f"Error processing training : {e}")

