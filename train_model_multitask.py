import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging

# --- モデルのインポート ---
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# ===== モデル選択 =====
# NOTE: このスクリプトは現在DenseNet121に最適化されています
MODEL_TO_USE = 'DenseNet121'
# =====================

# ===== ラベル設定 =====
# 各タスクのラベルを、フォルダ名で使う文字で定義してください
# 例: 拡張ゾーンが 'a', 'b', 'c' の3クラスなら ['a', 'b', 'c']
TASK_A_LABELS = ['a', 'b', 'c'] # Task A: 拡張ゾーン (3クラス)
TASK_B_LABELS = ['d', 'e']     # Task B: 内向/外向 (2クラス)
TASK_C_LABELS = ['f', 'g']     # Task C: 水平/垂直思考 (2クラス)
TASK_D_LABELS = ['h', 'i']     # Task D: 耳横拡張 (2クラス)
# =====================

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'train_log_4task_flat_{MODEL_TO_USE}.txt', mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

# ディレクトリのパス
TRAIN_DIR = 'preprocessed_multitask/train'
VALIDATION_DIR = 'preprocessed_multitask/validation'

# 設定
img_size = 224
BATCH_SIZE = 32

def create_multitask_model(input_shape, num_classes_a, num_classes_b, num_classes_c, num_classes_d):
    """4つのタスク出力を持つモデルを構築する"""
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    shared_features = layers.GlobalAveragePooling2D()(x)

    def create_head(shared_features, num_classes, name):
        head = layers.Dropout(0.3)(shared_features)
        return layers.Dense(num_classes, activation='softmax', name=name)(head)

    task_a_output = create_head(shared_features, num_classes_a, 'task_a_output')
    task_b_output = create_head(shared_features, num_classes_b, 'task_b_output')
    task_c_output = create_head(shared_features, num_classes_c, 'task_c_output')
    task_d_output = create_head(shared_features, num_classes_d, 'task_d_output')

    model = models.Model(inputs=inputs, outputs=[task_a_output, task_b_output, task_c_output, task_d_output])
    return model

def create_dataset(directory, task_labels, preprocessing_function):
    """フォルダ名からラベルを解析してtf.data.Datasetを作成する"""
    
    tables = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys=tf.constant(labels), values=tf.constant(list(range(len(labels))))), -1) for labels in task_labels]
    def parse_path(path):
        parts = tf.strings.split(path, os.sep)
        folder_name = parts[-2] # e.g., 'adfh'
        
        chars = [tf.strings.substr(folder_name, i, 1) for i in range(len(task_labels))]
        labels = tuple(tables[i].lookup(chars[i]) for i in range(len(task_labels)))

        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        return image, labels
    def preprocess_image(image, labels):
        image = preprocessing_function(image)
        return image, {f'task_{chr(97+i)}_output': label for i, label in enumerate(labels)}
    AUTOTUNE = tf.data.AUTOTUNE
    list_ds = tf.data.Dataset.list_files(f'{directory}/*/*.jpg', shuffle=True)
    ds = list_ds.map(parse_path, num_parallel_calls=AUTOTUNE)
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    
    return ds

def main():
    try:
        logger.info(f"Starting 4-task training with flat directory structure for model: {MODEL_TO_USE}")

        all_task_labels = [TASK_A_LABELS, TASK_B_LABELS, TASK_C_LABELS, TASK_D_LABELS]
        task_names = ["拡張ゾーン", "内向/外向", "水平/垂直思考", "耳横拡張"]
        num_classes = [len(labels) for labels in all_task_labels]

        for i, name in enumerate(task_names):
            logger.info(f"Task {chr(65+i)} ({name}): {num_classes[i]} classes -> {all_task_labels[i]}")

        train_ds = create_dataset(TRAIN_DIR, all_task_labels, densenet_preprocess)
        val_ds = create_dataset(VALIDATION_DIR, all_task_labels, densenet_preprocess)

        train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = create_multitask_model((img_size, img_size, 3), *num_classes)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={f'task_{chr(97+i)}_output': 'sparse_categorical_crossentropy' for i in range(len(all_task_labels))},
            loss_weights={f'task_{chr(97+i)}_output': 1.0/len(all_task_labels) for i in range(len(all_task_labels))}, 
            metrics={f'task_{chr(97+i)}_output': 'accuracy' for i in range(len(all_task_labels))}
        )
        model.summary(print_fn=logger.info)

        model_filename = f'best_4task_flat_model_{MODEL_TO_USE}.keras'
        callbacks = [
            ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=15, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        ]

        history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)

        best_epoch = history.history['val_loss'].index(min(history.history['val_loss']))
        logger.info(f"Best Epoch (based on val_loss): {best_epoch + 1}")
        
        print("\n" + "="*50)
        print("4-TASK TRAINING COMPLETE (FLAT DIRECTORY)")
        for i, name in enumerate(task_names):
            acc = history.history[f'val_task_{chr(97+i)}_output_accuracy'][best_epoch]
            logger.info(f"Val Acc Task {chr(65+i)} ({name}): {acc:.4f}")
            print(f"Val Acc Task {chr(65+i)} ({name}): {acc:.4f}")
        print("="*50)

    except Exception as e:
        logger.error(f"Error processing training: {e}", exc_info=True)

if __name__ == "__main__":
    main()