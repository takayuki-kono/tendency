import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetV2B0, ResNet50V2, Xception, DenseNet121
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import logging
import sys

# 再現性のためのシード固定
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 混合精度演算
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- 設定 ---
IMG_SIZE = 224
# EPOCHS = 10 # 引数に変更
BATCH_SIZE = 32
PREPROCESSED_TRAIN_DIR = 'preprocessed_multitask/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed_multitask/validation'

# タスク定義
TASK_A_LABELS = ['a', 'b', 'c']
TASK_B_LABELS = ['d', 'e']
TASK_C_LABELS = ['f', 'g']
TASK_D_LABELS = ['h', 'i']
ALL_TASK_LABELS = [TASK_A_LABELS, TASK_B_LABELS, TASK_C_LABELS, TASK_D_LABELS]

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_class_weights_as_tables(directory):
    # train_with_bayesian.py から移植
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

def create_dataset(directory, task_labels, weight_tables=None, augment_params=None):
    label_tables = [
        tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=tf.constant(labels), values=tf.constant(list(range(len(labels))))),
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
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
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
    
    # Cache here to avoid re-reading/decoding images
    ds = ds.cache()

    # データ拡張 (GPUで行うためモデル内ではなくここでやるか、モデル内に入れるか。train_with_bayesianはモデル内)
    # ここではモデル内に任せるため、ここでは何もしない
    
    ds = ds.map(apply_weights, num_parallel_calls=AUTOTUNE)
    return ds

def get_preprocessing_function(model_name):
    preprocess_map = {
        'EfficientNetV2B0': efficientnet_preprocess,
        'ResNet50V2': resnet_preprocess,
        'Xception': xception_preprocess,
        'DenseNet121': densenet_preprocess
    }
    return preprocess_map[model_name]

def create_model(model_name, num_dense_layers, dense_units, dropout, head_dropout, learning_rate, augment_params):
    model_map = {
        'EfficientNetV2B0': EfficientNetV2B0,
        'ResNet50V2': ResNet50V2,
        'Xception': Xception,
        'DenseNet121': DenseNet121
    }
    BaseCnnModel = model_map[model_name]
    preprocess_func = get_preprocessing_function(model_name)

    # データ拡張層
    aug_layers = []
    if augment_params['horizontal_flip']:
        aug_layers.append(layers.RandomFlip("horizontal"))
        
    aug_layers.extend([
        layers.RandomRotation(augment_params['rotation_range']),
        layers.RandomZoom(augment_params['zoom_range']),
        layers.RandomTranslation(height_factor=augment_params['height_shift_range'], width_factor=augment_params['width_shift_range']),
    ])
    
    data_augmentation = tf.keras.Sequential(aug_layers)

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Lambda(preprocess_func)(x)

    base_model = BaseCnnModel(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False # Transfer Learning
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(head_dropout)(x)

    for _ in range(num_dense_layers):
        x = layers.Dense(dense_units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout)(x)

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=metrics_dict,
        jit_compile=False
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    # Model Params
    parser.add_argument('--model_name', type=str, default='EfficientNetV2B0')
    parser.add_argument('--num_dense_layers', type=int, default=1)
    parser.add_argument('--dense_units', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--head_dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    
    # Augmentation Params
    parser.add_argument('--rotation_range', type=float, default=0.0) # 0.0-1.0 (fraction of 2pi)
    parser.add_argument('--width_shift_range', type=float, default=0.0)
    parser.add_argument('--height_shift_range', type=float, default=0.0)
    parser.add_argument('--zoom_range', type=float, default=0.0)
    parser.add_argument('--horizontal_flip', type=str, default='False')
    
    # Mode
    parser.add_argument('--fine_tune', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    augment_params = {
        'rotation_range': args.rotation_range, 
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'zoom_range': args.zoom_range,
        'horizontal_flip': args.horizontal_flip.lower() == 'true'
    }
    
    logger.info(f"Starting trial with params: {args}")

    weight_tables = calculate_class_weights_as_tables(PREPROCESSED_TRAIN_DIR)
    val_weight_tables = calculate_class_weights_as_tables(PREPROCESSED_VALIDATION_DIR)

    train_ds = create_dataset(PREPROCESSED_TRAIN_DIR, ALL_TASK_LABELS, weight_tables=weight_tables)\
        .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = create_dataset(PREPROCESSED_VALIDATION_DIR, ALL_TASK_LABELS, weight_tables=val_weight_tables)\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = create_model(
        args.model_name, 
        args.num_dense_layers, 
        args.dense_units, 
        args.dropout, 
        args.head_dropout,
        args.learning_rate,
        augment_params
    )

    # コールバック生成関数
    def create_callbacks():
        return [
            EarlyStopping(monitor='val_task_a_output_accuracy', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_task_a_output_accuracy', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        ]

    # --- Phase 1: 初期学習 (Headのみ) ---
    # 探索時と条件を合わせるため、Warmupは常に10エポック確保する
    phase1_epochs = 10
    if args.fine_tune.lower() == 'true':
        logger.info(f"--- Phase 1: Warmup Training (Head only, {phase1_epochs} epochs) ---")
    else:
        logger.info(f"--- Training (Head only, {phase1_epochs} epochs) ---") # fine_tune=Falseならこれが本番

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=create_callbacks(),
        verbose=2
    )
    
    # Phase 1 のベストスコアを記録
    if 'val_task_a_output_accuracy' in history.history:
        warmup_best_score = max(history.history['val_task_a_output_accuracy'])
    else:
        warmup_best_score = 0.0
    
    # Backup weights (FTで悪化した場合の保険)
    temp_weights_path = 'temp_warmup_weights.weights.h5'
    model.save_weights(temp_weights_path)


    # --- Phase 2: Fine-tuning (全層解凍) ---
    final_val_acc = warmup_best_score  # Initialize with warmup score by default

    if args.fine_tune.lower() == 'true':
        logger.info(f"--- Phase 2: Fine-tuning ({args.epochs} epochs) ---")
        
        # ベースモデル解凍
        base_model_layer = None
        for layer in model.layers:
            if isinstance(layer, models.Model):
                 base_model_layer = layer
                 break
        
        if base_model_layer:
            base_model_layer.trainable = True
            # 下位層は再固定
            for layer in base_model_layer.layers[:-40]:
                layer.trainable = False
            
            output_names = [f'task_{chr(97+i)}_output' for i in range(len(ALL_TASK_LABELS))]
            loss_dict = {name: 'sparse_categorical_crossentropy' for name in output_names}
            loss_weights_dict = {name: 1.0 / len(ALL_TASK_LABELS) for name in output_names}
            metrics_dict = {name: 'accuracy' for name in output_names}
            
            # 再コンパイル (学習率をさらに下げる: 1/10 -> 1/100)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate / 100),
                loss=loss_dict,
                loss_weights=loss_weights_dict,
                metrics=metrics_dict,
                jit_compile=False
            )
            
            # 再学習
            history_ft = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                callbacks=create_callbacks(),
                verbose=2
            )
            
            # スコア比較とロールバック
            if 'val_task_a_output_accuracy' in history_ft.history:
                ft_best_score = max(history_ft.history['val_task_a_output_accuracy'])
            else:
                ft_best_score = 0.0
            
            logger.info(f"Warmup Best: {warmup_best_score:.4f}, FT Best: {ft_best_score:.4f}")
            
            if ft_best_score < warmup_best_score:
                logger.warning("Fine-tuning degraded performance. Reverting to Warmup model.")
                model.load_weights(temp_weights_path)
                # historyを書き換えて最終出力が正しくなるようにする (簡易的)
                history = history # そのまま(Phase1の結果を持つオブジェクト)には戻らないが、スコア変数は別途管理すべき
                # 下流で history.history を参照しているので、ここはどうしようもないが
                # 少なくとも保存されるモデル（model.save）は戻った状態になる。
                # 画面表示用のスコア出力は以下のロジックで対応
                final_val_acc = warmup_best_score
            else:
                history = history_ft # FTの結果を採用
                final_val_acc = ft_best_score
        else:
            logger.warning("Base model layer not found for fine-tuning.")
            # final_val_acc は既に warmup_best_score で初期化済み

    # クリーンアップ
    if os.path.exists(temp_weights_path):
        os.remove(temp_weights_path)

    # モデル保存 (Fine-tuning時のみ)
    if args.fine_tune.lower() == 'true':
        save_path = 'best_sequential_model.keras'
        model.save(save_path)
        logger.info(f"Fine-tuned model saved to {save_path}")

    # 最終結果出力
    # 上記ロジックで final_val_acc が計算されている場合がある
    # 最終結果出力 (全タスク)
    print(f"FINAL_VAL_ACCURACY: {final_val_acc}") # 互換性のため維持

    # 全タスクのスコアを表示
    if 'history' in locals() and hasattr(history, 'history'):
        for char_code, task_label in zip(range(ord('a'), ord('a') + len(ALL_TASK_LABELS)), ['A', 'B', 'C', 'D']):
            task_key = f"val_task_{chr(char_code)}_output_accuracy"
            if task_key in history.history:
                # best score (max) or final score? Usually max is better for "potential"
                best_task_score = max(history.history[task_key])
                print(f"TASK_{task_label}_ACCURACY: {best_task_score}")
    else:
        logger.warning("History object not found, cannot print individual task scores.")

if __name__ == "__main__":
    main()
