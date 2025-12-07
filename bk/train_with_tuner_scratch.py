import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import logging
import keras_tuner as kt

# ===== モデル設定 =====
MODEL_NAME = 'ScratchCNN'
# =======================

# ===== ラベル/パス設定 =====
TASK_A_LABELS = ['a', 'b', 'c']
TASK_B_LABELS = ['d', 'e']
TASK_C_LABELS = ['f', 'g']
TASK_D_LABELS = ['h', 'i']
ALL_TASK_LABELS = [TASK_A_LABELS, TASK_B_LABELS, TASK_C_LABELS, TASK_D_LABELS]
TRAIN_DIR = 'preprocessed_multitask/train'
VALIDATION_DIR = 'preprocessed_multitask/validation'
img_size = 224
BATCH_SIZE = 32
# =================================================================

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'tuner_log_{MODEL_NAME}.txt', mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

def calculate_class_weights_as_tables(directory):
    logger.info(f"Calculating class weights from directory: {directory}")
    multi_label_counts = {}
    total_images = 0
    if not os.path.exists(directory): return None
    for label_name in os.listdir(directory):
        label_path = os.path.join(directory, label_name)
        if os.path.isdir(label_path):
            count = len([f for f in os.listdir(label_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            if count > 0:
                multi_label_counts[label_name] = count
                total_images += count
    if total_images == 0: return None
    logger.info(f"Found {total_images} total images across {len(multi_label_counts)} classes.")
    per_task_counts = [ {label: 0 for label in task_labels} for task_labels in ALL_TASK_LABELS ]
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
            tf.lookup.KeyValueTensorInitializer(keys=tf.constant(class_indices, dtype=tf.int64), values=tf.constant(class_weight_values, dtype=tf.float32)),
            default_value=tf.constant(1.0, dtype=tf.float32)
        )
        weight_tables.append(table)
    logger.info("Class weight tables calculated.")
    return weight_tables

def create_dataset(directory, task_labels, weight_tables=None):
    label_tables = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys=tf.constant(labels), values=tf.constant(list(range(len(labels))))), -1) for labels in task_labels]
    def parse_path(path):
        parts = tf.strings.split(path, os.sep)
        folder_name = parts[-2]
        chars = [tf.strings.substr(folder_name, i, 1) for i in range(len(task_labels))]
        labels = tuple(label_tables[i].lookup(chars[i]) for i in range(len(task_labels)))
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        return image, labels
    def apply_weights(image, labels):
        output_labels = {f'task_{chr(97+i)}_output': label for i, label in enumerate(labels)}
        if weight_tables:
            sample_weights = tuple(weight_tables[i].lookup(tf.cast(label, dtype=tf.int64)) for i, label in enumerate(labels))
            return image, output_labels, sample_weights
        else:
            return image, output_labels
    AUTOTUNE = tf.data.AUTOTUNE
    list_ds = tf.data.Dataset.list_files(f'{directory}/*/*.jpg', shuffle=True, seed=42)
    ds = list_ds.map(parse_path, num_parallel_calls=AUTOTUNE)
    ds = ds.map(apply_weights, num_parallel_calls=AUTOTUNE)
    return ds

def build_model(hp):
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Data Augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(hp.Float('aug_rotation', min_value=0.0, max_value=0.2, step=0.05))(x)
    x = layers.RandomZoom(hp.Float('aug_zoom', min_value=0.0, max_value=0.2, step=0.05))(x)
    x = layers.RandomContrast(hp.Float('aug_contrast', min_value=0.0, max_value=0.2, step=0.05))(x)

    # Rescaling
    x = layers.Rescaling(1./255)(x)

    # Convolutional Base
    for i in range(hp.Int('num_conv_blocks', 2, 4)):
        x = layers.Conv2D(
            filters=hp.Int(f'filters_{i}', 32, 128, step=32),
            kernel_size=3,
            padding='same',
            activation='relu'
        )(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    shared_features = layers.GlobalAveragePooling2D()(x)

    # Top (Head)
    head = shared_features
    head = layers.Dropout(hp.Float('head_dropout', min_value=0.2, max_value=0.6, step=0.1))(head)
    for i in range(hp.Int('num_dense_layers', 1, 2)):
        head = layers.Dense(
            units=hp.Int(f'units_{i}', min_value=128, max_value=512, step=128),
            activation='relu'
        )(head)
        head = layers.Dropout(hp.Float(f'dense_dropout_{i}', min_value=0.2, max_value=0.5, step=0.1))(head)

    # Output Layers
    outputs = []
    num_classes_list = [len(labels) for labels in ALL_TASK_LABELS]
    output_names = ['task_a_output', 'task_b_output', 'task_c_output', 'task_d_output']
    for i, num_classes in enumerate(num_classes_list):
        output_name = output_names[i]
        outputs.append(layers.Dense(num_classes, activation='softmax', name=output_name)(head))

    model = models.Model(inputs=inputs, outputs=outputs)
    lr = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    loss_dict = {}
    loss_weights_dict = {}
    metrics_dict = {}
    for i in range(len(ALL_TASK_LABELS)):
        name = f'task_{chr(97+i)}_output'
        loss_dict[name] = 'sparse_categorical_crossentropy'
        loss_weights_dict[name] = 1.0 / len(ALL_TASK_LABELS)
        metrics_dict[name] = 'accuracy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=metrics_dict
    )
    return model

def main():
    logger.info(f"--- KerasTuner Hyperparameter Search for {MODEL_NAME} ---")
    weight_tables = calculate_class_weights_as_tables(TRAIN_DIR)
    if weight_tables is None:
        logger.error("Could not calculate class weights. Aborting.")
        return

    train_ds = create_dataset(TRAIN_DIR, ALL_TASK_LABELS, weight_tables=weight_tables)
    val_ds = create_dataset(VALIDATION_DIR, ALL_TASK_LABELS)

    train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective("val_loss", direction="min"),
        max_epochs=30,  # Increased for scratch model
        factor=3,
        directory='kerastuner_dir',
        project_name=f'multitask_tuning_{MODEL_NAME}',
        overwrite=True
    )

    tuner.search_space_summary()
    logger.info("--- Starting Tuner Search ---")
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=30, # Increased for scratch model
        callbacks=[EarlyStopping(monitor='val_loss', patience=7)] # Increased patience
    )

    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"Best Hyperparameters Found for {MODEL_NAME}:")
    for hp, value in best_hps.values.items():
        logger.info(f"- {hp}: {value}")

    logger.info("--- Evaluating the best model ---")
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Re-compile with the best learning rate
    best_lr = best_hps.get('learning_rate')
    loss_dict = {}
    loss_weights_dict = {}
    metrics_dict = {}
    for i in range(len(ALL_TASK_LABELS)):
        name = f'task_{chr(97+i)}_output'
        loss_dict[name] = 'sparse_categorical_crossentropy'
        loss_weights_dict[name] = 1.0 / len(ALL_TASK_LABELS)
        metrics_dict[name] = 'accuracy'

    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=metrics_dict
    )

    best_model.summary(print_fn=logger.info)
    
    results = best_model.evaluate(val_ds, return_dict=True)
    logger.info(f"Final validation metrics for {MODEL_NAME}: {results}")

    print("\n" + "="*60)
    print(f"TUNER COMPLETE for {MODEL_NAME}")
    logger.info(f"Final Validation Metrics for {MODEL_NAME}:")
    for i, name in enumerate(["拡張ゾーン", "内向/外向", "水平/垂直思考", "耳横拡張"]):
        acc_key = f'task_{chr(97+i)}_output_accuracy'
        final_acc = results.get(acc_key, 0.0)
        logger.info(f"Final Val Acc Task {chr(65+i)} ({name}): {final_acc:.4f}")
        print(f"Final Val Acc Task {chr(65+i)} ({name}): {final_acc:.4f}")
    print("="*60)

    # Save the best model
    best_model.save(f'best_model_{MODEL_NAME}.keras')
    logger.info(f"Best model saved to best_model_{MODEL_NAME}.keras")


if __name__ == "__main__":
    main()
