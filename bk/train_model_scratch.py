
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
        logging.StreamHandler(),
        logging.FileHandler('train_log_scratch_4class.txt', mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

# ディレクトリのパス
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed/validation'

# 設定
ROTATION_RANGE = 1
img_size = 56 # 元の画像サイズに戻す

def count_images(directory):
    total_images = 0
    if not os.path.exists(directory):
        return 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
    return total_images

def get_num_classes(directory):
    if not os.path.exists(directory):
        return 0
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

def compute_class_weights(directory):
    num_classes = get_num_classes(directory)
    if num_classes == 0:
        logger.info("No class directories found, using equal weights.")
        return {}

    class_counts = {}
    categories = sorted([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])
    
    for category in categories:
        category_dir = os.path.join(directory, category)
        class_counts[category] = len([f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))])
    
    total_images = sum(class_counts.values())
    
    if total_images == 0:
        logger.info("No training data found, using equal weights.")
        return {i: 1.0 for i in range(num_classes)}
    
    class_weights = {}
    for idx, category in enumerate(categories):
        if class_counts[category] > 0:
            class_weights[idx] = total_images / (num_classes * class_counts[category])
        else:
            class_weights[idx] = 1.0
    
    logger.info(f"Class counts: {class_counts}")
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights

def weighted_sparse_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_flat = tf.reshape(y_true, [-1])
        
        weight_keys = sorted(class_weights.keys())
        weight_map = tf.constant([class_weights[k] for k in weight_keys], dtype=tf.float32)
        
        weights = tf.gather(weight_map, y_true_flat)
        
        unweighted_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        weighted_loss = unweighted_loss * weights
        return tf.reduce_mean(weighted_loss)
    return loss

# CNNモデル（ゼロから学習）
def create_cnn_model(input_shape=(56, 56, 1), num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), input_shape=input_shape),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3)),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax') # 出力層を4クラスに変更
    ])
    return model

def main():
    try:
        logger.info("Starting 4-class model training from scratch")
        
        num_classes = get_num_classes(PREPROCESSED_TRAIN_DIR)
        if num_classes < 2:
            logger.error(f"Not enough class directories found in {PREPROCESSED_TRAIN_DIR}. Found {num_classes}. Aborting.")
            return

        logger.info(f"Found {num_classes} classes.")

        train_image_count = count_images(PREPROCESSED_TRAIN_DIR)
        if train_image_count == 0:
            logger.error("No training images found. Aborting.")
            return
            
        BATCH_SIZE = max(4, min(32, train_image_count // 32 if train_image_count > 32 else 4))
        batch_count = math.ceil(train_image_count / BATCH_SIZE)
        logger.info(f"Dynamic BATCH_SIZE set to {BATCH_SIZE} for {train_image_count} training images ({batch_count} batches per epoch)")

        train_class_weights = compute_class_weights(PREPROCESSED_TRAIN_DIR)

        model = create_cnn_model(input_shape=(img_size, img_size, 1), num_classes=num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=weighted_sparse_categorical_crossentropy(train_class_weights),
            metrics=['accuracy']
        )
        model.summary(print_fn=logger.info)

        # rescaleのみのシンプルなジェネレータ
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=ROTATION_RANGE)
        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            PREPROCESSED_TRAIN_DIR, 
            target_size=(img_size, img_size), 
            batch_size=BATCH_SIZE, 
            class_mode='sparse', 
            color_mode='grayscale' # グレースケールに戻す
        )

        validation_generator = validation_datagen.flow_from_directory(
            PREPROCESSED_VALIDATION_DIR, 
            target_size=(img_size, img_size), 
            batch_size=BATCH_SIZE, 
            class_mode='sparse', 
            color_mode='grayscale' # グレースケールに戻す
        )

        model_checkpoint = ModelCheckpoint('best_model_scratch_4class.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=24, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=12, min_lr=1e-6, verbose=1)
        
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=100, # エポック数を増やす
            callbacks=[model_checkpoint, early_stopping, reduce_lr]
        )

        logger.info(f"Final Training accuracy: {history.history['accuracy'][-1]}")
        logger.info(f"Best Validation accuracy: {max(history.history['val_accuracy'])}")

        best_model = tf.keras.models.load_model('best_model_scratch_4class.keras', custom_objects={'loss': weighted_sparse_categorical_crossentropy(train_class_weights)})
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open('model_scratch_4class.tflite', 'wb') as f:
            f.write(tflite_model)

        logger.info("Model converted to TensorFlow Lite format and saved as 'model_scratch_4class.tflite'.")

    except Exception as e:
        logger.error(f"Error processing training: {e}", exc_info=True)

if __name__ == "__main__":
    main()
