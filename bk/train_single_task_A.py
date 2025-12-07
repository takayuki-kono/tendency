import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import numpy as np

# --- モデルのインポート ---
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# ===== モデル/ラベル/パス設定 =====
MODEL_TO_USE = 'DenseNet121'
TASK_A_LABELS = ['a', 'b', 'c']
NUM_CLASSES = len(TASK_A_LABELS)
TRAIN_DIR = 'preprocessed_multitask/train'
VALIDATION_DIR = 'preprocessed_multitask/validation'
img_size = 224
BATCH_SIZE = 32
# =================================================================

# ===== 学習設定 =====
HEAD_EPOCHS = 50
FINE_TUNE_EPOCHS = 50
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
FINE_TUNE_AT_LAYER = -40 # DenseNet121の下から40層をファインチューニング
# =================================

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'train_log_single_task_A.txt', mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

def calculate_class_weights(directory):
    """Task A専用のクラス重みを計算する"""
    logger.info(f"Calculating class weights for Task A from directory: {directory}")
    class_counts = {label: 0 for label in TASK_A_LABELS}
    total_images = 0
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return None

    for label_name in os.listdir(directory):
        label_path = os.path.join(directory, label_name)
        if os.path.isdir(label_path):
            task_a_label = label_name[0]
            if task_a_label in class_counts:
                count = len([f for f in os.listdir(label_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
                if count > 0:
                    class_counts[task_a_label] += count
                    total_images += count

    if total_images == 0:
        logger.warning("No images found for weight calculation.")
        return None

    logger.info(f"Found {total_images} total images. Class counts: {class_counts}")

    class_weights = {}
    for i, label in enumerate(TASK_A_LABELS):
        count = class_counts.get(label, 0)
        weight = total_images / (NUM_CLASSES * (count + 1e-6))
        class_weights[i] = weight
    
    logger.info(f"Calculated class weights: {class_weights}")
    return class_weights

def create_dataset(directory, augment=False):
    """Task A専用のデータセットを作成する"""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2)
    ])
    
    # 'a'->0, 'b'->1, 'c'->2
    label_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(TASK_A_LABELS), 
            values=tf.constant(list(range(NUM_CLASSES)))
        ), 
        -1
    )

    def parse_path(path):
        parts = tf.strings.split(path, os.sep)
        folder_name = parts[-2]
        # フォルダ名の最初の文字がTask Aのラベル
        task_a_char = tf.strings.substr(folder_name, 0, 1)
        label = label_table.lookup(task_a_char)
        
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        return image, label

    def process_image(image, label):
        if augment:
            image = data_augmentation(image, training=True)
        processed_image = densenet_preprocess(image)
        return processed_image, label

    AUTOTUNE = tf.data.AUTOTUNE
    list_ds = tf.data.Dataset.list_files(f'{directory}/*/*.jpg', shuffle=augment, seed=42)
    ds = list_ds.map(parse_path, num_parallel_calls=AUTOTUNE)
    ds = ds.map(process_image, num_parallel_calls=AUTOTUNE)
    return ds

def build_model(num_classes):
    """Task A専用のモデルを構築する"""
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
    
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False) # まずは全体を凍結
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model, base_model

def main():
    logger.info("--- Starting Single-Task Training for Task A ---")

    # --- データと重みの準備 ---
    class_weights = calculate_class_weights(TRAIN_DIR)
    if class_weights is None:
        return

    train_ds = create_dataset(TRAIN_DIR, augment=True)
    val_ds = create_dataset(VALIDATION_DIR, augment=False)

    train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- モデルの構築 ---
    model, base_model = build_model(NUM_CLASSES)
    
    # --- 1. ヘッドの学習 ---
    logger.info("--- Stage 1: Training the classification head ---")
    base_model.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary(print_fn=logger.info)

    head_callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    ]
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=head_callbacks,
        class_weight=class_weights
    )

    # --- 2. ファインチューニング ---
    logger.info("--- Stage 2: Fine-tuning the model ---")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False
    logger.info(f"Unfreezing from layer: {base_model.layers[FINE_TUNE_AT_LAYER].name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary(print_fn=logger.info)

    model_filename = f'best_model_single_task_A.keras'
    finetune_callbacks = [
        ModelCheckpoint(model_filename, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=finetune_callbacks,
        class_weight=class_weights
    )

    # --- 最終評価 ---
    logger.info(f"Loading best model from {model_filename} to report final accuracy.")
    model.load_weights(model_filename)
    results = model.evaluate(val_ds)
    
    final_acc = results[1]
    print("\n" + "="*60)
    print("SINGLE-TASK TRAINING FOR TASK A COMPLETE")
    logger.info("Final Validation Metrics after loading best model:")
    logger.info(f"Final Val Acc Task A: {final_acc:.4f}")
    print(f"Final Val Acc Task A: {final_acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
