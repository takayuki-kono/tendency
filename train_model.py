import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # ターミナル出力
        logging.FileHandler('train_log.txt', mode='w')  # ファイル出力
    ],
    force=True
)
logger = logging.getLogger(__name__)

# ディレクトリのパス
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed/validation'

# 設定
ROTATION_RANGE = 1
img_size = 56

# データ量をカウントする関数
def count_images(directory):
    total_images = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
    return total_images

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
        logger.info("No training data found, using equal weights.")
        print("No training data found, using equal weights.")
        return {0: 1.0, 1: 1.0}
    
    class_weights = {}
    for idx, category in enumerate(['category1', 'category2']):
        if class_counts[category] > 0:
            class_weights[idx] = total_images / (n_classes * class_counts[category])
        else:
            class_weights[idx] = 1.0
    
    logger.info(f"Class counts: {class_counts}")
    print(f"Class counts: {class_counts}")
    logger.info(f"Class weights: {class_weights}")
    print(f"Class weights: {class_weights}")
    
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

def main():
    try:
        logger.info("Starting model training")
        print("Starting model training")
        
        # 訓練データ量をカウント
        train_image_count = count_images(PREPROCESSED_TRAIN_DIR)
        BATCH_SIZE = max(4, min(32, train_image_count // 32))
        batch_count = math.ceil(train_image_count / BATCH_SIZE)
        logger.info(f"Dynamic BATCH_SIZE set to {BATCH_SIZE} for {train_image_count} training images ({batch_count} batches per epoch)")
        print(f"Dynamic BATCH_SIZE set to {BATCH_SIZE} for {train_image_count} training images ({batch_count} batches per epoch)")

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

        logger.info(f"Training accuracy: {history.history['accuracy'][-1]}")
        print(f"Training accuracy: {history.history['accuracy'][-1]}")
        logger.info(f"Validation accuracy: {max(history.history['val_accuracy'])}")
        print(f"Validation accuracy: {max(history.history['val_accuracy'])}")

        converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
        tflite_model = converter.convert()

        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)

        logger.info("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")
        print("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")

    except Exception as e:
        logger.error(f"Error processing training: {e}")
        print(f"Error processing training: {e}")

if __name__ == "__main__":
    main()