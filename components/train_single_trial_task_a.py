"""
Task A 単タスク学習スクリプト
既存のマルチラベルフォルダ構成（adfh, begi等）をそのまま使用し、
1文字目（a/b/c）のみをラベルとして学習する
"""
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
BATCH_SIZE = 32
PREPROCESSED_TRAIN_DIR = 'preprocessed_multitask/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed_multitask/validation'

# Task A のラベル定義
TASK_A_LABELS = ['a', 'b', 'c']

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_balanced_accuracy(model, dataset, num_classes):
    """バランス精度を計算（各クラスの正解率の平均）"""
    all_true = []
    all_pred = []
    
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        all_true.extend(labels.numpy())
        all_pred.extend(pred_classes)
    
    # 各クラスの正解率を計算
    per_class_acc = []
    for c in range(num_classes):
        class_indices = [i for i, t in enumerate(all_true) if t == c]
        if len(class_indices) > 0:
            correct = sum(1 for i in class_indices if all_pred[i] == c)
            per_class_acc.append(correct / len(class_indices))
    
    if len(per_class_acc) == 0:
        return 0.0
    
    balanced_acc = sum(per_class_acc) / len(per_class_acc)
    return balanced_acc, per_class_acc

def calculate_class_weights(directory):
    """Task A のクラス重みを計算"""
    logger.info(f"Calculating class weights for Task A from: {directory}")
    
    label_counts = {label: 0 for label in TASK_A_LABELS}
    
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path) and len(folder_name) > 0:
            first_char = folder_name[0]
            if first_char in label_counts:
                count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                label_counts[first_char] += count
    
    total = sum(label_counts.values())
    if total == 0:
        return None
    
    # クラス重み計算
    num_classes = len(TASK_A_LABELS)
    class_weights = {}
    for i, label in enumerate(TASK_A_LABELS):
        count = label_counts[label]
        if count > 0:
            class_weights[i] = total / (num_classes * count)
        else:
            class_weights[i] = 1.0
    
    logger.info(f"Label counts: {label_counts}")
    logger.info(f"Class weights: {class_weights}")
    return class_weights

def create_dataset(directory, is_training=True):
    """データセットを作成（Task A のみ）"""
    label_to_idx = {label: i for i, label in enumerate(TASK_A_LABELS)}
    
    def parse_path(path):
        # フォルダ名の1文字目をラベルとして使用
        parts = tf.strings.split(path, os.sep)
        folder_name = parts[-2]
        first_char = tf.strings.substr(folder_name, 0, 1)
        
        # ラベルをインデックスに変換
        label = tf.case([
            (tf.equal(first_char, 'a'), lambda: tf.constant(0)),
            (tf.equal(first_char, 'b'), lambda: tf.constant(1)),
            (tf.equal(first_char, 'c'), lambda: tf.constant(2)),
        ], default=lambda: tf.constant(-1))
        
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image, label
    
    AUTOTUNE = tf.data.AUTOTUNE
    list_ds = tf.data.Dataset.list_files(f'{directory}/*/*.jpg', shuffle=is_training, seed=SEED)
    
    # -1（無効ラベル）をフィルタ
    ds = list_ds.map(parse_path, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda img, label: label >= 0)
    
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
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(head_dropout)(x)

    for _ in range(num_dense_layers):
        x = layers.Dense(dense_units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout)(x)

    # 単タスク出力（3クラス分類）
    x_out = layers.Dense(len(TASK_A_LABELS), name='logits')(x)
    outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x_out)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=False
    )
    return model, base_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='EfficientNetV2B0')
    parser.add_argument('--num_dense_layers', type=int, default=1)
    parser.add_argument('--dense_units', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--head_dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--rotation_range', type=float, default=0.0)
    parser.add_argument('--width_shift_range', type=float, default=0.0)
    parser.add_argument('--height_shift_range', type=float, default=0.0)
    parser.add_argument('--zoom_range', type=float, default=0.0)
    parser.add_argument('--horizontal_flip', type=str, default='False')
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
    
    logger.info(f"Starting Task A single-task trial with params: {args}")

    # クラス重みを計算
    class_weights = calculate_class_weights(PREPROCESSED_TRAIN_DIR)

    # データセット作成
    train_ds = create_dataset(PREPROCESSED_TRAIN_DIR, is_training=True)\
        .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = create_dataset(PREPROCESSED_VALIDATION_DIR, is_training=False)\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model, base_model = create_model(
        args.model_name, 
        args.num_dense_layers, 
        args.dense_units, 
        args.dropout, 
        args.head_dropout,
        args.learning_rate,
        augment_params
    )

    def create_callbacks():
        return [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        ]

    # --- Phase 1: 初期学習 (Headのみ) ---
    phase1_epochs = 10
    if args.fine_tune.lower() == 'true':
        logger.info(f"--- Phase 1: Warmup Training (Head only, {phase1_epochs} epochs) ---")
    else:
        logger.info(f"--- Training (Head only, {phase1_epochs} epochs) ---")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=create_callbacks(),
        class_weight=class_weights,
        verbose=2
    )
    
    warmup_best_score = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0.0
    
    # Warmupモデルを保存（フルモデル）
    temp_model_path = 'temp_warmup_model_task_a.keras'
    model.save(temp_model_path)

    # --- Phase 2: Fine-tuning ---
    if args.fine_tune.lower() == 'true':
        logger.info(f"--- Phase 2: Fine-tuning ({args.epochs} epochs) ---")
        
        base_model.trainable = True
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate / 100),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            jit_compile=False
        )
        
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=create_callbacks(),
            class_weight=class_weights,
            verbose=2
        )
        
        ft_best_score = max(history_ft.history['val_accuracy']) if 'val_accuracy' in history_ft.history else 0.0
        
        logger.info(f"Warmup Best: {warmup_best_score:.4f}, FT Best: {ft_best_score:.4f}")
        
        if ft_best_score < warmup_best_score:
            logger.warning("Fine-tuning degraded performance. Reverting to Warmup model.")
            # Warmupモデルをリロード
            model = models.load_model(temp_model_path, compile=False)
            final_val_acc = warmup_best_score
        else:
            final_val_acc = ft_best_score
    else:
        final_val_acc = warmup_best_score

    # クリーンアップ
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    # モデル保存
    if args.fine_tune.lower() == 'true':
        save_path = 'best_sequential_model_task_a.keras'
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    # Balanced Accuracy を計算
    logger.info("Calculating Balanced Accuracy on validation set...")
    balanced_acc, per_class_acc = calculate_balanced_accuracy(model, val_ds, len(TASK_A_LABELS))
    
    logger.info(f"Per-class accuracy: {dict(zip(TASK_A_LABELS, [f'{a:.2%}' for a in per_class_acc]))}")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")

    # 最終結果出力
    print(f"FINAL_VAL_ACCURACY: {final_val_acc}")
    print(f"FINAL_BALANCED_ACCURACY: {balanced_acc}")

if __name__ == "__main__":
    main()
