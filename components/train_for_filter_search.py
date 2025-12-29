import os

# GPUメモリを必要な分だけ動的に割り当て（他アプリとの共存改善）
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

# GPU メモリ増分割り当てを有効化
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
import logging
import random
import numpy as np

# 再現性のためのシード固定
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 混合精度演算を有効化
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- 設定 ---
MODEL_NAME = 'EfficientNetV2B0'
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5  # 評価用なので少なく設定
TRAIN_DIR = 'preprocessed_multitask/train'
VALIDATION_DIR = 'preprocessed_multitask/validation'

# ラベル定義
TASK_A_LABELS = ['a', 'b', 'c']
TASK_B_LABELS = ['d', 'e']
TASK_C_LABELS = ['f', 'g']
TASK_D_LABELS = ['h', 'i']
ALL_TASK_LABELS = [TASK_A_LABELS, TASK_B_LABELS, TASK_C_LABELS, TASK_D_LABELS]

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_class_weights_as_tables(directory):
    # (既存のロジックと同じため省略可能だが、独立して動くように記述)
    if not os.path.exists(directory): return None
    
    multi_label_counts = {}
    total_images = 0
    for label_name in os.listdir(directory):
        label_path = os.path.join(directory, label_name)
        if os.path.isdir(label_path):
            count = len([f for f in os.listdir(label_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            logger.info(f"Label {label_name}: {count} images")
            if count > 0:
                multi_label_counts[label_name] = count
                total_images += count

    if total_images == 0: return None

    per_task_counts = [{label: 0 for label in task_labels} for task_labels in ALL_TASK_LABELS]
    for multi_label, count in multi_label_counts.items():
        for i, char_label in enumerate(multi_label):
            if i < len(per_task_counts) and char_label in per_task_counts[i]:
                per_task_counts[i][char_label] += count

    weight_tables = []
    for i, task_labels in enumerate(ALL_TASK_LABELS):
        counts = per_task_counts[i]
        num_classes = len(task_labels)
        class_indices = []
        class_weight_values = []
        for j, label in enumerate(task_labels):
            class_indices.append(j)
            class_count = counts.get(label, 0)
            weight = total_images / (num_classes * (class_count + 1e-6))
            class_weight_values.append(weight)

        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(class_indices, dtype=tf.int64),
                values=tf.constant(class_weight_values, dtype=tf.float32)
            ),
            default_value=tf.constant(1.0, dtype=tf.float32)
        )
        weight_tables.append(table)
    return weight_tables

def create_dataset(directory, task_labels, weight_tables=None):
    label_tables = [
        tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(labels),
                values=tf.constant(list(range(len(labels))))
            ), -1
        ) for labels in task_labels
    ]

    def parse_path(path):
        parts = tf.strings.split(path, os.sep)
        folder_name = parts[-2]
        chars = [tf.strings.substr(folder_name, i, 1) for i in range(len(task_labels))]
        labels = tuple(label_tables[i].lookup(chars[i]) for i in range(len(task_labels)))
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image, labels

    def apply_weights(image, labels):
        output_labels = {f'task_{chr(97+i)}_output': label for i, label in enumerate(labels)}
        if weight_tables:
            sample_weights = tuple(
                weight_tables[i].lookup(tf.cast(label, dtype=tf.int64))
                for i, label in enumerate(labels)
            )
            return image, output_labels, sample_weights
        else:
            return image, output_labels

    AUTOTUNE = tf.data.AUTOTUNE
    list_ds = tf.data.Dataset.list_files(f'{directory}/*/*.jpg', shuffle=True, seed=42)
    ds = list_ds.map(parse_path, num_parallel_calls=AUTOTUNE)
    ds = ds.map(apply_weights, num_parallel_calls=AUTOTUNE)
    return ds

def main():
    logger.info("Starting lightweight training for filter evaluation...")
    
    weight_tables = calculate_class_weights_as_tables(TRAIN_DIR)
    if weight_tables is None:
        print(f"Error: No training data found in {os.path.abspath(TRAIN_DIR)}")
        return

    val_weight_tables = calculate_class_weights_as_tables(VALIDATION_DIR)

    train_ds = create_dataset(TRAIN_DIR, ALL_TASK_LABELS, weight_tables=weight_tables)\
        .cache()\
        .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = create_dataset(VALIDATION_DIR, ALL_TASK_LABELS, weight_tables=val_weight_tables)\
        .cache()\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 固定モデル構築 (EfficientNetV2B0)
    from tensorflow.keras.applications import EfficientNetV2B0
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # データ拡張は最低限（評価のブレを防ぐため）
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.Lambda(preprocess_input)(x)

    base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False # 高速化のため凍結

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.2)(x)

    output_names = [f'task_{chr(97+i)}_output' for i in range(len(ALL_TASK_LABELS))]
    outputs = []
    for name, labels in zip(output_names, ALL_TASK_LABELS):
        # Mixed Precision安定化: 出力層はfloat32で行う
        x_out = layers.Dense(len(labels), name=name+'_logits')(x)
        outputs.append(layers.Activation('softmax', dtype='float32', name=name)(x_out))

    model = models.Model(inputs=inputs, outputs=outputs)

    loss_dict = {name: 'sparse_categorical_crossentropy' for name in output_names}
    loss_weights_dict = {name: 1.0 / len(ALL_TASK_LABELS) for name in output_names}
    metrics_dict = {name: 'accuracy' for name in output_names}

    model.compile(
        optimizer='adam',
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=metrics_dict,
        # jit_compile=True  # XLAコンパイルは libdevice.10.bc エラーのため無効化（GPUは引き続き使用される）
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    # 最終評価 (Balanced Accuracy を計算)
    print("Calculating Balanced Accuracy...")
    y_true_all = []
    y_pred_all = []
    
    # バリデーションデータから全予測を取得
    for batch_images, batch_labels_dict, *_ in val_ds: # 重みがある場合は *_ で吸収
        batch_preds = model.predict(batch_images, verbose=0)
        
        # Task A (index 0) に注目
        # batch_preds は [task_a_pred, task_b_pred, ...] のリスト
        task_a_pred_probs = batch_preds[0] 
        task_a_preds = np.argmax(task_a_pred_probs, axis=1)
        
        # 正解ラベル (Batchごとの辞書から取得)
        task_a_true = batch_labels_dict['task_a_output_accuracy' if 'task_a_output_accuracy' in batch_labels_dict else 'task_a_output'].numpy()

        y_true_all.extend(task_a_true)
        y_pred_all.extend(task_a_preds)
        
    # 混同行列作成
    from sklearn.metrics import balanced_accuracy_score
    final_score = balanced_accuracy_score(y_true_all, y_pred_all)
    
    print(f"FINAL_SCORE: {final_score}") # Balanced Accuracyを出力

if __name__ == "__main__":
    main()
