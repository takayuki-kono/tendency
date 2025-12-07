import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import numpy as np
import keras_tuner as kt
from tensorflow.keras import mixed_precision

# 混合精度演算を有効化（GPU高速化）
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- モデルのインポート ---
from tensorflow.keras.applications import DenseNet121, EfficientNetV2B0, ResNet50V2
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess

# ===== モデル選択 =====
MODEL_TO_USE = 'EfficientNetV2B0'
# =======================

# ===== 探索スキップ設定 =====
SKIP_STRUCTURE_TUNING = False    # Stage 1: 構造探索をスキップ
SKIP_AUGMENTATION_TUNING = False # Stage 2: データ拡張探索をスキップ

# ===== 継続探索設定 =====
CONTINUE_SEARCH = False    # True: 既存の探索結果に追記, False: 新規探索（上書き）
ADDITIONAL_TRIALS = 10     # 継続する場合に追加する試行回数
# ============================

# ===== ベイズ最適化チューニング設定 =====
# Stage 1: モデル構造探索（データ拡張は固定）
STAGE1_MAX_TRIALS = 30     # 試行回数を増加（10 -> 30）
STAGE1_INITIAL_POINTS = 5  # 初期ランダム探索回数も少し増加
STAGE1_EPOCHS = 10         # 各試行のエポック数

# Stage 2: データ拡張探索（構造は固定）
STAGE2_MAX_TRIALS = 30     # 試行回数を増加（10 -> 30）
STAGE2_INITIAL_POINTS = 5  # 初期ランダム探索回数も少し増加
STAGE2_EPOCHS = 10         # 各試行のエポック数

# Stage 3: 最終ファインチューニング
FINAL_EPOCHS = 50          # 本格学習
FINE_TUNE_AT_LAYER = -40   # ファインチューニングで解凍する層
# ====================================

# モデルごとの設定を定義
MODEL_CONFIG = {
    'DenseNet121': {'class': DenseNet121, 'preprocess': densenet_preprocess, 'base_name': 'densenet121_base'},
    'EfficientNetV2B0': {'class': EfficientNetV2B0, 'preprocess': efficientnet_preprocess, 'base_name': 'efficientnetv2b0_base'},
    'ResNet50V2': {'class': ResNet50V2, 'preprocess': resnet_preprocess, 'base_name': 'resnet50v2_base'},
}

if MODEL_TO_USE not in MODEL_CONFIG:
    raise ValueError(f"Unsupported model: {MODEL_TO_USE}. Please choose from {list(MODEL_CONFIG.keys())}")

SelectedModelClass = MODEL_CONFIG[MODEL_TO_USE]['class']
selected_preprocess_func = MODEL_CONFIG[MODEL_TO_USE]['preprocess']
selected_base_model_name = MODEL_CONFIG[MODEL_TO_USE]['base_name']

# ===== ラベル/パス設定 =====
TASK_A_LABELS = ['a', 'b', 'c']
TASK_B_LABELS = ['d', 'e']
TASK_C_LABELS = ['f', 'g']
TASK_D_LABELS = ['h', 'i']
ALL_TASK_LABELS = [TASK_A_LABELS, TASK_B_LABELS, TASK_C_LABELS, TASK_D_LABELS]
TRAIN_DIR = 'preprocessed_multitask/train'
VALIDATION_DIR = 'preprocessed_multitask/validation'
img_size = 224
BATCH_SIZE = 32  # 精度優先のため32に戻す
# =================================================================

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'bayesian_tuner_log_{MODEL_TO_USE}.txt', mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

def calculate_class_weights_as_tables(directory):
    logger.info(f"Calculating class weights from directory: {directory}")
    multi_label_counts = {}
    total_images = 0
    if not os.path.exists(directory):
        return None

    for label_name in os.listdir(directory):
        label_path = os.path.join(directory, label_name)
        if os.path.isdir(label_path):
            count = len([f for f in os.listdir(label_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            if count > 0:
                multi_label_counts[label_name] = count
                total_images += count

    if total_images == 0:
        return None

    logger.info(f"Found {total_images} total images across {len(multi_label_counts)} classes.")

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

    logger.info("Class weight tables calculated.")
    return weight_tables

def create_dataset(directory, task_labels, weight_tables=None):
    label_tables = [
        tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(labels),
                values=tf.constant(list(range(len(labels))))
            ),
            -1
        ) for labels in task_labels
    ]

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

class MultiStageHyperModel(kt.HyperModel):
    """
    段階的探索用HyperModel
    - Stage 1: モデル構造のみ探索（データ拡張は固定）
    - Stage 2: データ拡張のみ探索（構造は固定）
    """
    def __init__(self, tuning_stage='structure', fixed_hps=None):
        super().__init__()
        self.tuning_stage = tuning_stage
        self.fixed_hps = fixed_hps if fixed_hps else kt.HyperParameters()

    def build(self, hp):
        # ===== Stage 1: モデル構造の探索 =====
        if self.tuning_stage == 'structure':
            # 構造パラメータを探索
            num_dense_layers = hp.Int('num_dense_layers', 1, 2)
            head_dropout = hp.Float('head_dropout', min_value=0.3, max_value=0.6, step=0.1)
            lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

            # Dense層の設定
            dense_units = []
            dense_dropouts = []
            for i in range(2):  # 最大2層分を定義
                units = hp.Int(f'units_{i}', min_value=64, max_value=256, step=64)
                dropout = hp.Float(f'dense_dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)
                dense_units.append(units)
                dense_dropouts.append(dropout)

            # データ拡張は固定（デフォルト値）
            aug_rotation = 0.1
            aug_zoom = 0.1
            aug_contrast = 0.1
            aug_brightness = 0.1
            aug_trans_height = 0.05
            aug_trans_width = 0.05

        # ===== Stage 2: データ拡張の探索 =====
        elif self.tuning_stage == 'augmentation':
            # 構造パラメータは固定
            num_dense_layers = self.fixed_hps.get('num_dense_layers')
            head_dropout = self.fixed_hps.get('head_dropout')
            lr = self.fixed_hps.get('learning_rate')

            dense_units = []
            dense_dropouts = []
            for i in range(2):
                if f'units_{i}' in self.fixed_hps.values:
                    dense_units.append(self.fixed_hps.get(f'units_{i}'))
                    dense_dropouts.append(self.fixed_hps.get(f'dense_dropout_{i}'))
                else:
                    dense_units.append(128)
                    dense_dropouts.append(0.3)

            # データ拡張パラメータを探索
            aug_rotation = hp.Float('aug_rotation', min_value=0.0, max_value=0.2, step=0.05)
            aug_zoom = hp.Float('aug_zoom', min_value=0.0, max_value=0.2, step=0.05)
            aug_contrast = hp.Float('aug_contrast', min_value=0.0, max_value=0.2, step=0.05)
            aug_brightness = hp.Float('aug_brightness', min_value=0.0, max_value=0.2, step=0.05)
            aug_trans_height = hp.Float('aug_trans_height', min_value=0.0, max_value=0.1, step=0.02)
            aug_trans_width = hp.Float('aug_trans_width', min_value=0.0, max_value=0.1, step=0.02)

        # ===== 最終モデル（全て固定） =====
        else:  # 'final'
            num_dense_layers = self.fixed_hps.get('num_dense_layers')
            head_dropout = self.fixed_hps.get('head_dropout')
            lr = self.fixed_hps.get('learning_rate')

            dense_units = []
            dense_dropouts = []
            for i in range(2):
                if f'units_{i}' in self.fixed_hps.values:
                    dense_units.append(self.fixed_hps.get(f'units_{i}'))
                    dense_dropouts.append(self.fixed_hps.get(f'dense_dropout_{i}'))
                else:
                    dense_units.append(128)
                    dense_dropouts.append(0.3)

            aug_rotation = self.fixed_hps.get('aug_rotation')
            aug_zoom = self.fixed_hps.get('aug_zoom')
            aug_contrast = self.fixed_hps.get('aug_contrast')
            aug_brightness = self.fixed_hps.get('aug_brightness')
            aug_trans_height = self.fixed_hps.get('aug_trans_height')
            aug_trans_width = self.fixed_hps.get('aug_trans_width')

        # ===== モデル構築 =====
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(aug_rotation),
            layers.RandomZoom(aug_zoom),
            layers.RandomContrast(aug_contrast),
            layers.RandomBrightness(aug_brightness),
            layers.RandomTranslation(height_factor=aug_trans_height, width_factor=aug_trans_width),
        ])

        inputs = layers.Input(shape=(img_size, img_size, 3))
        x = data_augmentation(inputs)
        x = layers.Lambda(selected_preprocess_func)(x)

        core_base_model = SelectedModelClass(
            include_top=False,
            weights='imagenet',
            input_shape=(img_size, img_size, 3)
        )
        base_model_wrapper = models.Model(
            inputs=core_base_model.input,
            outputs=core_base_model.output,
            name=selected_base_model_name
        )
        base_model_wrapper.trainable = False

        x = base_model_wrapper(x, training=False)
        shared_features = layers.GlobalAveragePooling2D()(x)
        head = layers.Dropout(head_dropout)(shared_features)

        # Dense層を追加
        for i in range(num_dense_layers):
            head = layers.Dense(units=dense_units[i])(head)
            head = layers.BatchNormalization()(head)
            head = layers.Activation('relu')(head)
            head = layers.Dropout(dense_dropouts[i])(head)

        # 出力層
        output_names = [f'task_{chr(97+i)}_output' for i in range(len(ALL_TASK_LABELS))]
        outputs = [
            layers.Dense(len(labels), activation='softmax', name=name)(head)
            for name, labels in zip(output_names, ALL_TASK_LABELS)
        ]

        model = models.Model(inputs=inputs, outputs=outputs)

        # コンパイル
        loss_dict = {name: 'sparse_categorical_crossentropy' for name in output_names}
        loss_weights_dict = {name: 1.0 / len(ALL_TASK_LABELS) for name in output_names}
        metrics_dict = {name: 'accuracy' for name in output_names}

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=loss_dict,
            loss_weights=loss_weights_dict,
            metrics=metrics_dict,
            jit_compile=False  # XLAコンパイル無効化（データ拡張との互換性のため）
        )

        return model

def save_hyperparameters(hps, filename):
    """ハイパーパラメータをJSONファイルに保存"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(hps.values, f, indent=2, ensure_ascii=False)
    logger.info(f"Hyperparameters saved to {filename}")

def load_hyperparameters(filename):
    """JSONファイルからハイパーパラメータを読み込み"""
    if not os.path.exists(filename):
        logger.warning(f"Hyperparameter file not found: {filename}")
        return None

    with open(filename, 'r', encoding='utf-8') as f:
        values = json.load(f)

    # kt.HyperParametersオブジェクトを作成
    hps = kt.HyperParameters()
    for key, value in values.items():
        hps.values[key] = value

    logger.info(f"Hyperparameters loaded from {filename}")
    logger.info(f"Loaded values: {hps.values}")
    return hps

def get_existing_trial_count(project_name, directory='kerastuner_bayesian'):
    """指定されたプロジェクトの既存試行回数をカウント"""
    project_dir = os.path.join(directory, project_name)
    if not os.path.exists(project_dir):
        return 0
    
    # trial_で始まるディレクトリ数をカウント
    count = 0
    for name in os.listdir(project_dir):
        if name.startswith('trial_') and os.path.isdir(os.path.join(project_dir, name)):
            count += 1
    return count

def main():
    logger.info(f"=== Starting Bayesian Optimization Tuning for {MODEL_TO_USE} ===")
    logger.info(f"Stage 1: Structure tuning - {STAGE1_MAX_TRIALS} trials")
    logger.info(f"Stage 2: Augmentation tuning - {STAGE2_MAX_TRIALS} trials")
    logger.info(f"Stage 3: Final fine-tuning - {FINAL_EPOCHS} epochs")

    weight_tables = calculate_class_weights_as_tables(TRAIN_DIR)
    if weight_tables is None:
        logger.error("Could not calculate class weights for training. Aborting.")
        return

    # 検証用データの重み計算（評価を均等にするため）
    val_weight_tables = calculate_class_weights_as_tables(VALIDATION_DIR)
    if val_weight_tables is None:
        logger.warning("Could not calculate class weights for validation. Proceeding without validation weights.")

    train_ds = create_dataset(TRAIN_DIR, ALL_TASK_LABELS, weight_tables=weight_tables)\
        .cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = create_dataset(VALIDATION_DIR, ALL_TASK_LABELS, weight_tables=val_weight_tables)\
        .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ===== Stage 1: モデル構造の探索 =====
    logger.info("\n" + "="*60)
    logger.info("STAGE 1: MODEL STRUCTURE TUNING (データ拡張は固定)")
    logger.info("="*60)

    structure_hp_file = f'best_hps_structure_bayesian_{MODEL_TO_USE}.json'

    if SKIP_STRUCTURE_TUNING:
        logger.info("Skipping structure tuning. Loading saved hyperparameters...")
        best_hps_structure = load_hyperparameters(structure_hp_file)

        if best_hps_structure is None:
            logger.error(f"Cannot skip structure tuning: No saved hyperparameters found at {structure_hp_file}")
            logger.error("Please set SKIP_STRUCTURE_TUNING = False to run tuning first.")
            return

        logger.info("Successfully loaded structure hyperparameters:")
        for key, value in best_hps_structure.values.items():
            logger.info(f"  {key}: {value}")
    else:
        project_name_s1 = f'{MODEL_TO_USE}_stage1_structure'
        
        if CONTINUE_SEARCH:
            existing_trials = get_existing_trial_count(project_name_s1)
            current_max_trials = existing_trials + ADDITIONAL_TRIALS
            overwrite_flag = False
            logger.info(f"Continuing search (Stage 1): {existing_trials} existing + {ADDITIONAL_TRIALS} new = {current_max_trials} trials")
        else:
            current_max_trials = STAGE1_MAX_TRIALS
            overwrite_flag = True
            logger.info(f"Starting fresh search (Stage 1) with {current_max_trials} trials")

        logger.info("Running structure tuning (Bayesian Optimization)...")

        tuner_structure = kt.BayesianOptimization(
            MultiStageHyperModel(tuning_stage='structure'),
            objective=kt.Objective('val_task_a_output_accuracy', direction='max'),
            max_trials=current_max_trials,
            num_initial_points=STAGE1_INITIAL_POINTS,
            directory='kerastuner_bayesian',
            project_name=project_name_s1,
            overwrite=overwrite_flag
        )

        tuner_structure.search(
            train_ds,
            validation_data=val_ds,
            epochs=STAGE1_EPOCHS,
            callbacks=[
                EarlyStopping(monitor='val_task_a_output_accuracy', mode='max', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_task_a_output_accuracy', mode='max', factor=0.5, patience=2, min_lr=1e-6)
            ]
        )

        best_hps_structure = tuner_structure.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best hyperparameters from Stage 1 (structure): {best_hps_structure.values}")

        # 結果を保存
        save_hyperparameters(best_hps_structure, structure_hp_file)

    # ===== Stage 2: データ拡張の探索 =====
    logger.info("\n" + "="*60)
    logger.info("STAGE 2: DATA AUGMENTATION TUNING (構造は固定)")
    logger.info("="*60)

    augmentation_hp_file = f'best_hps_augmentation_bayesian_{MODEL_TO_USE}.json'

    if SKIP_AUGMENTATION_TUNING:
        logger.info("Skipping augmentation tuning. Loading saved hyperparameters...")
        best_hps_augmentation = load_hyperparameters(augmentation_hp_file)

        if best_hps_augmentation is None:
            logger.error(f"Cannot skip augmentation tuning: No saved hyperparameters found at {augmentation_hp_file}")
            logger.error("Please set SKIP_AUGMENTATION_TUNING = False to run tuning first.")
            return

        logger.info("Successfully loaded augmentation hyperparameters:")
        for key, value in best_hps_augmentation.values.items():
            logger.info(f"  {key}: {value}")
    else:
        project_name_s2 = f'{MODEL_TO_USE}_stage2_augmentation'

        if CONTINUE_SEARCH:
            existing_trials = get_existing_trial_count(project_name_s2)
            current_max_trials = existing_trials + ADDITIONAL_TRIALS
            overwrite_flag = False
            logger.info(f"Continuing search (Stage 2): {existing_trials} existing + {ADDITIONAL_TRIALS} new = {current_max_trials} trials")
        else:
            current_max_trials = STAGE2_MAX_TRIALS
            overwrite_flag = True
            logger.info(f"Starting fresh search (Stage 2) with {current_max_trials} trials")

        logger.info("Running augmentation tuning (Bayesian Optimization)...")

        tuner_augmentation = kt.BayesianOptimization(
            MultiStageHyperModel(tuning_stage='augmentation', fixed_hps=best_hps_structure),
            objective=kt.Objective('val_task_a_output_accuracy', direction='max'),
            max_trials=current_max_trials,
            num_initial_points=STAGE2_INITIAL_POINTS,
            directory='kerastuner_bayesian',
            project_name=project_name_s2,
            overwrite=overwrite_flag
        )

        tuner_augmentation.search(
            train_ds,
            validation_data=val_ds,
            epochs=STAGE2_EPOCHS,
            callbacks=[
                EarlyStopping(monitor='val_task_a_output_accuracy', mode='max', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_task_a_output_accuracy', mode='max', factor=0.5, patience=2, min_lr=1e-6)
            ]
        )

        best_hps_augmentation = tuner_augmentation.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best hyperparameters from Stage 2 (augmentation): {best_hps_augmentation.values}")

        # 結果を保存
        save_hyperparameters(best_hps_augmentation, augmentation_hp_file)

    # ===== Stage 3: 最終モデルのファインチューニング =====
    logger.info("\n" + "="*60)
    logger.info("STAGE 3: FINAL FINE-TUNING with BEST HYPERPARAMETERS")
    logger.info("="*60)

    # 最終ハイパーパラメータを結合
    final_hps = best_hps_structure
    for hp_name, hp_value in best_hps_augmentation.values.items():
        if hp_name.startswith('aug_'):
            final_hps.values[hp_name] = hp_value

    logger.info(f"Final combined hyperparameters: {final_hps.values}")

    # 最終モデル構築
    final_hypermodel = MultiStageHyperModel(tuning_stage='final', fixed_hps=final_hps)
    best_model = final_hypermodel.build(kt.HyperParameters())

    # ベースモデルの凍結解除
    base_model = best_model.get_layer(selected_base_model_name)
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    logger.info(f"Unfreezing from layer: {base_model.layers[FINE_TUNE_AT_LAYER].name}")

    # ファインチューニング用の低学習率
    finetune_lr = final_hps.get('learning_rate') / 10
    logger.info(f"Using fine-tuning learning rate: {finetune_lr}")

    # 再コンパイル
    loss_dict = {f'task_{chr(97+i)}_output': 'sparse_categorical_crossentropy'
                 for i in range(len(ALL_TASK_LABELS))}
    loss_weights_dict = {f'task_{chr(97+i)}_output': 1.0 / len(ALL_TASK_LABELS)
                         for i in range(len(ALL_TASK_LABELS))}
    metrics_dict = {f'task_{chr(97+i)}_output': 'accuracy'
                    for i in range(len(ALL_TASK_LABELS))}

    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=metrics_dict,
        jit_compile=False  # XLAコンパイル無効化（データ拡張との互換性のため）
    )

    best_model.summary(print_fn=logger.info)

    # ファインチューニング
    callbacks = [
        EarlyStopping(monitor='val_task_a_output_accuracy', mode='max', patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_task_a_output_accuracy', mode='max', factor=0.2, patience=4, min_lr=1e-7, verbose=1)
    ]

    history = best_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINAL_EPOCHS,
        callbacks=callbacks
    )

    # ===== 最終評価 =====
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION")
    logger.info("="*60)

    results = best_model.evaluate(val_ds, return_dict=True)
    logger.info(f"Final validation metrics: {results}")

    print("\n" + "="*60)
    print(f"BAYESIAN TUNING + FINE-TUNING COMPLETE for {MODEL_TO_USE}")
    print("="*60)

    task_names = ["拡張ゾーン", "内向/外向", "水平/垂直思考", "耳横拡張"]
    for i, name in enumerate(task_names):
        acc_key = f'task_{chr(97+i)}_output_accuracy'
        final_acc = results.get(acc_key, 0.0)
        logger.info(f"Final Val Acc Task {chr(65+i)} ({name}): {final_acc:.4f}")
        print(f"Final Val Acc Task {chr(65+i)} ({name}): {final_acc:.4f}")

    print("="*60)

    # モデル保存（スコア比較）
    model_filename = f'best_bayesian_tuned_model_{MODEL_TO_USE}.keras'
    
    # 今回の主要スコア（Task A Accuracy）
    current_val_acc = results.get('task_a_output_accuracy', 0.0)
    
    should_save = True
    if os.path.exists(model_filename):
        try:
            logger.info(f"Existing model found. Comparing scores...")
            # 既存モデルをロードして評価するのは重いので、ファイル名や別ファイルにスコアを記録しておくのが定石だが、
            # ここでは簡易的に「既存モデルをロードして評価」するか、
            # または「前回までのベストスコア」をどこかに保存しておく必要がある。
            # 今回はシンプルに「既存モデルをロードして評価」する（時間はかかるが確実）。
            
            # ただし、ロードして評価はメモリも食うため、
            # 「best_score.txt」のようなファイルで管理する方式に変更する。
            pass
        except Exception as e:
            logger.warning(f"Failed to check existing model: {e}")

    score_filename = f'best_score_{MODEL_TO_USE}.txt'
    best_prev_score = -1.0
    
    if os.path.exists(score_filename):
        try:
            with open(score_filename, 'r') as f:
                best_prev_score = float(f.read().strip())
            logger.info(f"Previous best score: {best_prev_score}")
        except:
            pass

    if current_val_acc > best_prev_score:
        logger.info(f"New best model found! ({current_val_acc:.4f} > {best_prev_score:.4f}) Saving model...")
        best_model.save(model_filename)
        with open(score_filename, 'w') as f:
            f.write(str(current_val_acc))
        print(f"Model saved to {model_filename}")
    else:
        logger.info(f"Current model ({current_val_acc:.4f}) is not better than previous best ({best_prev_score:.4f}). NOT saving.")
        print(f"Current model ({current_val_acc:.4f}) is not better than previous best ({best_prev_score:.4f}). NOT saving.")

if __name__ == "__main__":
    main()
