import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import mixed_precision

from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2S, ResNet50V2, Xception, DenseNet121
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

    def to_one_hot(image, labels_dict, weights=None):
        # Convert sparse dict labels to one-hot dict
        new_labels = {}
        for k, v in labels_dict.items():
             # Find matching task index to get num_classes
             # Key format: 'task_a_output'
             char_code = k.split('_')[1] # 'a'
             idx = ord(char_code) - ord('a')
             num_classes = len(task_labels[idx])
             new_labels[k] = tf.one_hot(tf.cast(v, tf.int32), num_classes)
        if weights is not None:
            return image, new_labels, weights
        return image, new_labels

    def mixup(entry1, entry2):
        # unpack
        img1, lab1 = entry1[:2]
        img2, lab2 = entry2[:2]
        w1 = entry1[2] if len(entry1) > 2 else None
        w2 = entry2[2] if len(entry2) > 2 else None
        
        alpha = augment_params['mixup_alpha']
        l = tf.random.gamma([], alpha, 1.0)
        l = tf.math.reduce_max([l, 1-l]) # force > 0.5 to keep dominant label if we were using it, but for soft mix it's symmetric
        # Beta distribution is better: np.random.beta(alpha, alpha)
        # But tf doesn't have beta easily accessible without tfp. 
        # Approx with uniform if alpha=1, or just 1.0 if placeholder.
        # Let's use simple linear blend with random ratio
        ratio = tf.random.uniform([], 0, 1)

        img_mix = ratio * img1 + (1 - ratio) * img2
        
        lab_mix = {}
        for k in lab1:
            lab_mix[k] = ratio * lab1[k] + (1 - ratio) * lab2[k]
            
        if w1 is not None and w2 is not None:
             w_mix = tuple(ratio * w1[i] + (1 - ratio) * w2[i] for i in range(len(w1)))
             return img_mix, lab_mix, w_mix
        return img_mix, lab_mix

    AUTOTUNE = tf.data.AUTOTUNE
    list_ds = tf.data.Dataset.list_files(f'{directory}/*/*.jpg', shuffle=True, seed=42)
    ds = list_ds.map(parse_path, num_parallel_calls=AUTOTUNE)
    ds = ds.cache() # Cache raw images before weights/mixup
    ds = ds.map(apply_weights, num_parallel_calls=AUTOTUNE)

    # Mixup Logic
    if augment_params and augment_params.get('mixup_alpha', 0.0) > 0.0:
        # Convert to one-hot first
        ds = ds.map(to_one_hot, num_parallel_calls=AUTOTUNE)
        
        # Create shuffle pair
        ds_shuffled = ds.shuffle(1000)
        ds = tf.data.Dataset.zip((ds, ds_shuffled))
        ds = ds.map(mixup, num_parallel_calls=AUTOTUNE)

    return ds

def get_preprocessing_function(model_name):
    preprocess_map = {
        'EfficientNetV2B0': efficientnet_preprocess,
        'ResNet50V2': resnet_preprocess,
        'Xception': xception_preprocess,
        'DenseNet121': densenet_preprocess
    }
    return preprocess_map[model_name]

class BalancedSparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='balanced_accuracy', **kwargs):
        super(BalancedSparseCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.total_count = self.add_weight(name='tc', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle One-hot inputs (Mixup)
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
            
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)
        
        # Flatten
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        if sample_weight is not None:
             sample_weight = tf.cast(sample_weight, tf.float32)
             sample_weight = tf.reshape(sample_weight, [-1])

        for i in range(self.num_classes):
             is_class = tf.equal(y_true, i)
             is_class = tf.cast(is_class, tf.float32)
             
             if sample_weight is not None:
                 is_class = is_class * sample_weight

             self.total_count[i].assign_add(tf.reduce_sum(is_class))
             
             is_correct = tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i))
             is_correct = tf.cast(is_correct, tf.float32)
             
             if sample_weight is not None:
                 is_correct = is_correct * sample_weight
                 
             self.true_positives[i].assign_add(tf.reduce_sum(is_correct))

    def result(self):
        per_class_acc = tf.math.divide_no_nan(self.true_positives, self.total_count)
        return tf.reduce_mean(per_class_acc)

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.total_count.assign(tf.zeros(self.num_classes))

def create_model(model_name, num_dense_layers, dense_units, dropout, head_dropout, learning_rate, augment_params):
    model_map = {
        'EfficientNetV2B0': EfficientNetV2B0,
        'EfficientNetV2S': EfficientNetV2S,
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
    
    # Loss switching for Mixup
    loss_fn = 'sparse_categorical_crossentropy'
    if augment_params.get('mixup_alpha', 0.0) > 0.0:
        loss_fn = 'categorical_crossentropy'

    loss_dict = {name: loss_fn for name in output_names}
    loss_weights_dict = {name: 1.0 / len(ALL_TASK_LABELS) for name in output_names}
    
    # メトリクス定義 (Balanced Accuracyを追加)
    metrics_dict = {}
    for name, labels in zip(output_names, ALL_TASK_LABELS):
        metrics_dict[name] = [
            'accuracy', 
            BalancedSparseCategoricalAccuracy(len(labels), name='balanced_accuracy')
        ]

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
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    augment_params = {
        'rotation_range': args.rotation_range, 
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'zoom_range': args.zoom_range,
        'horizontal_flip': args.horizontal_flip.lower() == 'true',
        'mixup_alpha': 0.0 # Placeholder for now, to be implemented properly next step
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

    # コールバック生成関数 (Balanced Accuracyを監視 + Cosine Decay)
    def create_callbacks(total_epochs, initial_lr):
        def cosine_decay(epoch):
            if total_epochs == 0: return initial_lr
            # Cosine Decay: 0.5 * (1 + cos(pi * epoch / total_epochs)) * initial_lr
            return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

        return [
            EarlyStopping(monitor='val_task_a_output_balanced_accuracy', patience=5, restore_best_weights=True, verbose=1, mode='max'),
            LearningRateScheduler(cosine_decay, verbose=1)
        ]

    # --- Phase 1: 初期学習 (Headのみ) ---
    # 探索時と条件を合わせるため、Warmupは常に10エポック確保する
    phase1_epochs = 10
    if args.fine_tune.lower() == 'true':
        logger.info(f"--- Phase 1: Warmup Training (Head only, {phase1_epochs} epochs) ---")
    else:
        logger.info(f"--- Training (Head only, {phase1_epochs} epochs) ---")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=create_callbacks(phase1_epochs, args.learning_rate),
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
                callbacks=create_callbacks(args.epochs, args.learning_rate / 100),
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
