import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision

from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2S, ResNet50V2, Xception, DenseNet121
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import logging
import sys

# デフォルトシード（--seed引数で上書き可能）
DEFAULT_SEED = 42

# 混合精度演算
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- 設定 ---
IMG_SIZE = 224
# EPOCHS = 10 # 引数に変更
BATCH_SIZE = 32
PREPROCESSED_TRAIN_DIR = 'preprocessed_multitask/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed_multitask/validation'

# タスク定義（動的生成のため初期値は空）
ALL_TASK_LABELS = []

def get_all_task_labels(directory, single_task_mode=False):
    """
    ディレクトリ内のフォルダ名からタスク構造を解析する
    """
    if not os.path.exists(directory):
        # 存在しない場合はデフォルト（エラー回避のためだが、実行時には必ずディレクトリが必要）
        # フォールバックとして4タスクを返す（既存互換）
        return [['a', 'b'], ['d', 'e'], ['f', 'g'], ['h', 'i']]
        
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if not subdirs:
        return [['a', 'b'], ['d', 'e'], ['f', 'g'], ['h', 'i']]
    
    if single_task_mode:
        # シングルタスクモード: ディレクトリ名をそのままクラス名として扱う
        # タスク数は1、クラス名はディレクトリ名のリスト
        sorted_labels = [sorted(subdirs)]
        logger.info(f"Single Task Mode: Detected 1 task with {len(sorted_labels[0])} classes")
        logger.info(f"  Classes: {sorted_labels[0]}")
        return sorted_labels

    # 文字数が揃っているか確認し、揃っていればそれをタスク数とする
    first_len = len(subdirs[0])
    task_labels = [set() for _ in range(first_len)]
    
    for d in subdirs:
        if len(d) != first_len:
            continue # 文字数が違うフォルダは無視（エラーにするのもありだが）
        for i, char in enumerate(d):
            task_labels[i].add(char)
            
    # ソートしてリスト化
    sorted_labels = [sorted(list(chars)) for chars in task_labels]
    
    # ログ出力
    logger.info(f"Detected Task Structure: {len(sorted_labels)} tasks")
    for i, labels in enumerate(sorted_labels):
        logger.info(f"  Task {chr(65+i)}: {labels}")

    # 全てのタスクが少なくとも1つのクラスを持っているか確認
    # (持っていない場合は空リストになるが、その場合はタスクとして機能しない)
    return sorted_labels

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_class_weights_as_tables(directory, task_labels, single_task_mode=False):
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

    if single_task_mode:
        # シングルタスクモード: ディレクトリ名がそのままクラス
        # task_labels[0] に全クラスが入っている
        per_task_counts = [multi_label_counts] # そのまま辞書を使う
    else:
        per_task_counts = [{label: 0 for label in t_labels} for t_labels in task_labels]
        for multi_label, count in multi_label_counts.items():
            for i, char_label in enumerate(multi_label):
                if i < len(per_task_counts) and char_label in per_task_counts[i]:
                    per_task_counts[i][char_label] += count

    weight_tables = []
    for i, labels in enumerate(task_labels):
        counts = per_task_counts[i]
        num_classes = len(labels)
        class_indices = []
        class_weight_values = []
        for j, label in enumerate(labels):
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

def create_dataset(directory, task_labels, weight_tables=None, augment_params=None, single_task_mode=False):
    label_tables = [
        tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=tf.constant(labels), values=tf.constant(list(range(len(labels))))),
            -1
        ) for labels in task_labels
    ]

    def parse_path(path):
        parts = tf.strings.split(path, os.sep)
        folder_name = parts[-2]
        
        if single_task_mode:
            # シングルタスクモード: フォルダ名全体をクラス名としてルックアップ
            # タスクは1つだけなので、index 0 のテーブルを使用
            labels = tuple([label_tables[0].lookup(folder_name)])
        else:
            # マルチタスクモード: 文字分解
            chars = [tf.strings.substr(folder_name, i, 1) for i in range(len(task_labels))]
            labels = tuple(label_tables[i].lookup(chars[i]) for i in range(len(task_labels)))
            
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
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
        # Beta(alpha, alpha) 分布をGamma分布2つから構築
        g1 = tf.random.gamma([], alpha)
        g2 = tf.random.gamma([], alpha)
        ratio = g1 / (g1 + g2 + 1e-8)

        img_mix = ratio * img1 + (1 - ratio) * img2
        
        lab_mix = {}
        for k in lab1:
            lab_mix[k] = ratio * lab1[k] + (1 - ratio) * lab2[k]
            
        if w1 is not None and w2 is not None:
             w_mix = tuple(ratio * w1[i] + (1 - ratio) * w2[i] for i in range(len(w1)))
             return img_mix, lab_mix, w_mix
        return img_mix, lab_mix

    AUTOTUNE = tf.data.AUTOTUNE
    # 複数拡張子に対応
    import glob as glob_module
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob_module.glob(os.path.join(directory, '*', ext)))
    if not image_files:
        raise ValueError(f"No image files found in {directory}")
    list_ds = tf.data.Dataset.from_tensor_slices(image_files)
    if True:  # Always shuffle for training consistency
        list_ds = list_ds.shuffle(buffer_size=len(image_files), seed=42)
    ds = list_ds.map(parse_path, num_parallel_calls=AUTOTUNE)
    ds = ds.cache() # Cache raw images before weights/mixup
    ds = ds.map(apply_weights, num_parallel_calls=AUTOTUNE)

    # Mixup / Smoothing Logic
    if augment_params and (augment_params.get('mixup_alpha', 0.0) > 0.0 or augment_params.get('label_smoothing', 0.0) > 0.0):
        # Convert to one-hot first (Required for both Mixup and Label Smoothing)
        ds = ds.map(to_one_hot, num_parallel_calls=AUTOTUNE)
        
        # Mixup (Only if alpha > 0)
        if augment_params.get('mixup_alpha', 0.0) > 0.0:
            # Create shuffle pair
            ds_shuffled = ds.shuffle(1000)
            ds = tf.data.Dataset.zip((ds, ds_shuffled))
            ds = ds.map(mixup, num_parallel_calls=AUTOTUNE)

    return ds

def get_preprocessing_function(model_name):
    preprocess_map = {
        'EfficientNetV2B0': efficientnet_preprocess,
        'EfficientNetV2S': efficientnet_preprocess,
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

        # Vectorized Update
        y_true_onehot = tf.one_hot(y_true, self.num_classes) # [Batch, NumClasses]
        
        if sample_weight is not None:
             # Ensure shape broadcasting [Batch, 1] * [Batch, NumClasses]
             sample_weight = tf.expand_dims(sample_weight, -1)
             y_true_onehot = y_true_onehot * sample_weight

        # Update Total Count (per class presence)
        self.total_count.assign_add(tf.reduce_sum(y_true_onehot, axis=0))

        # Update True Positives
        correct_mask = tf.cast(tf.equal(y_true, y_pred), tf.float32) # [Batch]
        # Mask the one-hot vectors to only keep correct ones
        correct_onehot = y_true_onehot * tf.expand_dims(correct_mask, -1)
        self.true_positives.assign_add(tf.reduce_sum(correct_onehot, axis=0))

    def result(self):
        per_class_acc = tf.math.divide_no_nan(self.true_positives, self.total_count)
        return tf.reduce_mean(per_class_acc)

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.total_count.assign(tf.zeros(self.num_classes))

class MinClassAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='min_class_accuracy', **kwargs):
        super(MinClassAccuracy, self).__init__(name=name, **kwargs)
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

        # Vectorized Update
        y_true_onehot = tf.one_hot(y_true, self.num_classes) # [Batch, NumClasses]
        
        if sample_weight is not None:
             # Ensure shape broadcasting [Batch, 1] * [Batch, NumClasses]
             sample_weight = tf.expand_dims(sample_weight, -1)
             y_true_onehot = y_true_onehot * sample_weight

        # Update Total Count (per class presence)
        self.total_count.assign_add(tf.reduce_sum(y_true_onehot, axis=0))

        # Update True Positives
        correct_mask = tf.cast(tf.equal(y_true, y_pred), tf.float32) # [Batch]
        # Mask the one-hot vectors to only keep correct ones
        correct_onehot = y_true_onehot * tf.expand_dims(correct_mask, -1)
        self.true_positives.assign_add(tf.reduce_sum(correct_onehot, axis=0))

    def result(self):
        per_class_acc = tf.math.divide_no_nan(self.true_positives, self.total_count)
        return tf.reduce_min(per_class_acc)

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.total_count.assign(tf.zeros(self.num_classes))



def create_model(model_name, num_dense_layers, dense_units, dropout, head_dropout, learning_rate, augment_params, task_labels):
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

    # Weight Decay: L2正則化で実装（AdamWが使えない環境向け）
    wd = augment_params.get('weight_decay', 0.0)
    kernel_reg = tf.keras.regularizers.l2(wd) if wd > 0 else None

    for _ in range(num_dense_layers):
        x = layers.Dense(dense_units, kernel_regularizer=kernel_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout)(x)

    output_names = [f'task_{chr(97+i)}_output' for i in range(len(task_labels))]
    outputs = []
    for name, labels in zip(output_names, task_labels):
        # Mixed Precision安定化: 出力層はfloat32で行う
        x_out = layers.Dense(len(labels), name=name+'_logits')(x)
        outputs.append(layers.Activation('softmax', dtype='float32', name=name)(x_out))

    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Loss switching for Mixup/Smoothing
    label_smoothing = augment_params.get('label_smoothing', 0.0)
    use_categorical = augment_params.get('mixup_alpha', 0.0) > 0.0 or label_smoothing > 0.0
    
    if use_categorical:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    else:
        loss_fn = 'sparse_categorical_crossentropy'

    loss_dict = {name: loss_fn for name in output_names}
    loss_weights_dict = {name: 1.0 / len(task_labels) for name in output_names}
    
    # メトリクス定義 (Balanced Accuracyを追加)
    metrics_dict = {}
    for name, labels in zip(output_names, task_labels):
        metrics_list = ['accuracy'] 
        if use_categorical:
            metrics_list = [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
        
        metrics_list.append(BalancedSparseCategoricalAccuracy(len(labels), name='balanced_accuracy'))
        metrics_list.append(MinClassAccuracy(len(labels), name='min_class_accuracy'))
        metrics_dict[name] = metrics_list

    # Optimizer: Weight DecayはDense層のkernel_regularizerで適用済み
    try:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=metrics_dict,
        jit_compile=False
    )
    return model


class ConditionalLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    MinClassAccuracy > 0.5 を達成した次のエポックから減衰を開始する... 
    -> 変更(2026-02-17): 常に最初(Epoch 0)から減衰を開始する仕様に変更。
    """
    def __init__(self, initial_lr, total_epochs, task_labels, decay_exponent=1.0, verbose=0):
        super(ConditionalLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.task_labels = task_labels # For metric name resolution
        self.decay_exponent = decay_exponent
        self.verbose = verbose
        self.decay_start_epoch = 0 # 常に0から開始
        self.metric_history = []

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            return
            
        lr = self.initial_lr
        
        # 減衰開始条件を満たしてからの経過エポック数で減衰を計算
        if self.decay_start_epoch is not None:
             # 残りエポック数で減衰させる
             remaining_epochs = max(1.0, self.total_epochs - self.decay_start_epoch)
             
             # progress: 0.0 (start) -> 1.0 (end)
             current_step = epoch - self.decay_start_epoch
             progress = current_step / remaining_epochs
             progress = min(1.0, max(0.0, progress))
             
             min_lr = self.initial_lr * 0.05
             # Polynomial Decay: (1 - progress) ^ exponent
             linear_decay = 1.0 - progress
             decay = linear_decay ** self.decay_exponent
             lr = min_lr + (self.initial_lr - min_lr) * decay
        
        # Set LR
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1}: LearningRateScheduler setting learning rate to {lr:.8f}. (Decay Started: {self.decay_start_epoch})')

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return

        # Calculate Average Min Class Accuracy
        task_mins = []
        if len(self.task_labels) == 1:
            val = logs.get('val_min_class_accuracy')
            if val is not None: task_mins.append(val)
        else:
            for i in range(len(self.task_labels)):
                key = f"val_task_{chr(ord('a')+i)}_output_min_class_accuracy"
                val = logs.get(key)
                if val is not None: task_mins.append(val)
        
        current_score = 0.0
        if task_mins:
            current_score = sum(task_mins) / len(task_mins)
        
        self.metric_history.append(current_score)

        # Condition Check: Removed (Always Active)
        # self.decay_start_epoch is already 0.

def main():
    parser = argparse.ArgumentParser()
    # Model Params
    parser.add_argument('--model_name', type=str, default='EfficientNetV2B0')
    parser.add_argument('--num_dense_layers', type=int, default=1)
    parser.add_argument('--dense_units', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--head_dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    
    # Augmentation Params
    parser.add_argument('--rotation_range', type=float, default=0.0) # 0.0-1.0 (fraction of 2pi)
    parser.add_argument('--width_shift_range', type=float, default=0.0)
    parser.add_argument('--height_shift_range', type=float, default=0.0)
    parser.add_argument('--zoom_range', type=float, default=0.0)
    parser.add_argument('--horizontal_flip', type=str, default='False')
    parser.add_argument('--mixup_alpha', type=float, default=0.0)
    
    # Mode
    parser.add_argument('--fine_tune', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--unfreeze_layers', type=int, default=40)
    parser.add_argument('--single_task_mode', type=str, default='False') # "True" or "False"
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    
    # 学習率自動調整: 指定したepochでベストになるようLRをデータ枚数に基づき自動算出
    # 0 = 無効（--learning_rate をそのまま使用）、>0 = target epoch
    parser.add_argument('--auto_lr_target_epoch', type=int, default=0)
    parser.add_argument('--enable_early_stopping', type=str, default='True')
    parser.add_argument('--decay_exponent', type=float, default=1.0) # Decay curve exponent (1.0=Linear, 0.5=Sqrt, 2.0=Poly)

    args = parser.parse_args()
    
    # シード設定（--seedで上書き可能）
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    augment_params = {
        'rotation_range': args.rotation_range, 
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'zoom_range': args.zoom_range,
        'horizontal_flip': args.horizontal_flip.lower() == 'true',
        'mixup_alpha': args.mixup_alpha,
        'label_smoothing': args.label_smoothing,
        'weight_decay': args.weight_decay
    }
    
    single_task_mode = args.single_task_mode.lower() == 'true'
    # single_task_mode = True # Hardcoded as requested - Reverted
    
    logger.info(f"Starting trial with params: {args}")
    logger.info(f"Single Task Mode: {single_task_mode}")
    
    # 動的タスクラベル取得
    task_labels = get_all_task_labels(PREPROCESSED_TRAIN_DIR, single_task_mode=single_task_mode)
    logger.info(f"Detected {len(task_labels)} tasks from {PREPROCESSED_TRAIN_DIR}")

    weight_tables = calculate_class_weights_as_tables(PREPROCESSED_TRAIN_DIR, task_labels, single_task_mode=single_task_mode)
    val_weight_tables = calculate_class_weights_as_tables(PREPROCESSED_VALIDATION_DIR, task_labels, single_task_mode=single_task_mode)

    train_ds = create_dataset(PREPROCESSED_TRAIN_DIR, task_labels, weight_tables=weight_tables, augment_params=augment_params, single_task_mode=single_task_mode)\
        .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # 検証データにはMixup/Label Smoothingを適用しない（生データで正しく評価）
    val_ds = create_dataset(PREPROCESSED_VALIDATION_DIR, task_labels, weight_tables=val_weight_tables, augment_params=None, single_task_mode=single_task_mode)\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- 学習率の自動算出 (auto_lr_target_epoch > 0 の場合) ---
    effective_lr = args.learning_rate
    if args.auto_lr_target_epoch > 0:
        # 訓練画像の枚数をカウント
        import glob as glob_count
        train_image_count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            train_image_count += len(glob_count.glob(os.path.join(PREPROCESSED_TRAIN_DIR, '*', ext)))
        
        steps_per_epoch = max(train_image_count / BATCH_SIZE, 1)
        
        # 学習率スケーリング: sqrt（平方根）ベース
        # 
        # 「1 epochあたりの総学習量 ∝ steps_per_epoch × learning_rate」
        # を一定に保つなら線形（LR × reference/actual）だが、
        # 実際のNNの学習ではSGDノイズの蓄積やモーメンタムの効果があるため
        # 線形だと小データ時にLRが大きくなりすぎて不安定化する。
        # 
        # sqrt(reference / actual) を使うことで:
        #   - データ640枚(steps≈20) に対し reference=20 → scale=1.0（変化なし）
        #   - データ320枚(steps≈10) に対し → scale=sqrt(2)≈1.41（穏やかな増加）
        #   - データ3200枚(steps≈100) に対し → scale=sqrt(0.2)≈0.45（穏やかな減少）
        #
        # 基準: steps_per_epoch=20（画像約640枚/batch32）で base_lr がそのまま適用
        REFERENCE_STEPS_PER_EPOCH = 20.0
        lr_scale = np.sqrt(REFERENCE_STEPS_PER_EPOCH / steps_per_epoch)
        
        # 極端なスケーリングを防止 (0.3倍〜3.0倍の範囲に制限)
        lr_scale = max(0.3, min(lr_scale, 3.0))
        
        effective_lr = args.learning_rate * lr_scale
        
        logger.info(f"[Auto LR] target_best_epoch={args.auto_lr_target_epoch}")
        logger.info(f"[Auto LR] train_images={train_image_count}, batch_size={BATCH_SIZE}, steps_per_epoch={steps_per_epoch:.1f}")
        logger.info(f"[Auto LR] reference_steps={REFERENCE_STEPS_PER_EPOCH}, scale={lr_scale:.4f} (sqrt-based)")
        logger.info(f"[Auto LR] base_lr={args.learning_rate} -> effective_lr={effective_lr:.8f}")

    # LRの決定
    if args.fine_tune.lower() == 'true':
        current_lr = args.learning_rate
    else:
        current_lr = effective_lr

    model = create_model(
        args.model_name, 
        args.num_dense_layers, 
        args.dense_units, 
        args.dropout, 
        args.head_dropout,
        current_lr,
        augment_params,
        task_labels
    )

    # コールバック生成関数 (Balanced Accuracyを監視 + Conditional Cosine Decay)
    def create_callbacks(total_epochs, initial_lr, target_epoch=0, enable_early_stopping=True):

        # 精度ベースEarlyStopping: 全タスク平均MinClassAccuracyを監視
        class AccuracyEarlyStopping(tf.keras.callbacks.Callback):
            """
            全タスクの平均val_min_class_accuracyが低下したらpatience回猶予後に停止。
            損失関数ではなく精度を基準にする。
            """
            def __init__(self, task_labels, patience=5, verbose=1):
                super().__init__()
                self.task_labels = task_labels
                self.patience = patience
                self.verbose = verbose
                self.best_score = -1.0
                self.wait = 0
                self.best_weights = None
                self.best_epoch = 0

            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    return
                # 全タスクの平均MinClassAccuracyを計算
                task_mins = []
                if len(self.task_labels) == 1:
                    val = logs.get('val_min_class_accuracy')
                    if val is not None:
                        task_mins.append(float(val))
                else:
                    for i in range(len(self.task_labels)):
                        key = f"val_task_{chr(ord('a')+i)}_output_min_class_accuracy"
                        val = logs.get(key)
                        if val is not None:
                            task_mins.append(float(val))

                if not task_mins:
                    return

                current_score = sum(task_mins) / len(task_mins)

                if current_score > self.best_score:
                    self.best_score = current_score
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                    self.best_epoch = epoch
                else:
                    self.wait += 1
                    if self.verbose > 0:
                        logger.info(f"AccuracyEarlyStopping: 精度改善なし ({self.wait}/{self.patience}), best={self.best_score:.4f} at epoch {self.best_epoch+1}")
                    if self.wait >= self.patience:
                        if self.verbose > 0:
                            logger.info(f"AccuracyEarlyStopping: Patience超過。Epoch {self.best_epoch+1}のベスト重みを復元")
                        self.model.stop_training = True
                        if self.best_weights is not None:
                            self.model.set_weights(self.best_weights)

        # エポックごとの精度サマリー出力
        class EpochSummaryCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    return
                parts = [f"Epoch {epoch+1}/{total_epochs}"]
                # 各タスクのMinClassAccuracy
                if len(task_labels) == 1:
                    val_min = logs.get('val_min_class_accuracy', None)
                    if val_min is not None:
                        parts.append(f"MinClassAcc={val_min:.4f}")
                else:
                    task_mins = []
                    for i in range(len(task_labels)):
                        key = f"val_task_{chr(97+i)}_output_min_class_accuracy"
                        val = logs.get(key, None)
                        if val is not None:
                            parts.append(f"Task{chr(65+i)}={val:.4f}")
                            task_mins.append(val)
                    if task_mins:
                        avg = sum(task_mins) / len(task_mins)
                        parts.append(f"Avg={avg:.4f}")
                # Loss
                val_loss = logs.get('val_loss', None)
                if val_loss is not None:
                    parts.append(f"Loss={val_loss:.4f}")
                logger.info(" | ".join(parts))

        scheduler = ConditionalLearningRateScheduler(initial_lr, total_epochs, task_labels, decay_exponent=args.decay_exponent, verbose=1)
        
        callbacks_list = [
            scheduler,
            EpochSummaryCallback()
        ]

        if enable_early_stopping:
            # patience: target_epochの半分を目安にする（ただし最低3）
            if target_epoch > 0:
                patience = max(3, target_epoch // 2)
            else:
                patience = 5
            # 精度ベースEarlyStopping（全タスク平均MinClassAccuracy監視）
            callbacks_list.insert(0, AccuracyEarlyStopping(task_labels=task_labels, patience=patience, verbose=1))
        
        return callbacks_list

    # --- 構成と学習方針の決定 ---
    if args.fine_tune.lower() == 'true':
        logger.info(f"--- Fine-tuning Mode ({args.epochs} epochs) ---")
        training_epochs = args.epochs
        
        # ベースモデル解凍
        base_model_layer = None
        for layer in model.layers:
            if isinstance(layer, models.Model):
                 base_model_layer = layer
                 break
        
        if base_model_layer:
            base_model_layer.trainable = True
            # 下位層は再固定 (unfreeze_layers で制御)
            if args.unfreeze_layers < len(base_model_layer.layers):
                for layer in base_model_layer.layers[:-args.unfreeze_layers]:
                    layer.trainable = False
            logger.info(f"Unfreezing top {args.unfreeze_layers} layers of {len(base_model_layer.layers)} total")
    else:
        training_epochs = args.auto_lr_target_epoch * 2 if args.auto_lr_target_epoch > 0 else args.epochs
        logger.info(f"--- Training Mode (Head only, {training_epochs} epochs) ---")

    # 再コンパイル (Fine-tuning時は解凍したため、またはPhase 1でも明示的に再コンパイルを統一で行う)
    output_names = [f'task_{chr(97+i)}_output' for i in range(len(task_labels))]
    
    label_smoothing = augment_params.get('label_smoothing', 0.0)
    use_categorical = augment_params.get('mixup_alpha', 0.0) > 0.0 or label_smoothing > 0.0
    
    if use_categorical:
        loss_dict = {name: tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing) for name in output_names}
    else:
        loss_dict = {name: 'sparse_categorical_crossentropy' for name in output_names}

    loss_weights_dict = {name: 1.0 / len(task_labels) for name in output_names}
    
    metrics_dict = {}
    for name, labels in zip(output_names, task_labels):
        metrics_list = ['accuracy'] 
        if use_categorical:
            metrics_list = [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
        
        metrics_list.append(BalancedSparseCategoricalAccuracy(len(labels), name='balanced_accuracy'))
        metrics_list.append(MinClassAccuracy(len(labels), name='min_class_accuracy'))
        metrics_dict[name] = metrics_list
    
    logger.info(f"Training LR: {current_lr:.8f} (clipnorm=1.0)")
    try:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=current_lr, clipnorm=1.0)
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=loss_weights_dict,
        metrics=metrics_dict,
        jit_compile=False
    )
    
    # --- 学習実行 (単一フェーズ) ---
    enable_early_stopping = args.enable_early_stopping.lower() == 'true'
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_epochs,
        callbacks=create_callbacks(training_epochs, current_lr, target_epoch=args.auto_lr_target_epoch, enable_early_stopping=enable_early_stopping),
        verbose=2
    )
    
    # ベストスコア記録
    final_val_acc = 0.0
    best_epoch = 1
    last_epoch_score = 0.0  # 最終エポックのスコア（延長判定用）
    if hasattr(history, 'history'):
        task_acc_keys = []
        if len(task_labels) == 1:
            single_keys = ['val_min_class_accuracy', 'val_balanced_accuracy', 'val_accuracy']
            for k in single_keys:
                if k in history.history:
                    task_acc_keys.append(k)
                    break 
        else:
            for i in range(len(task_labels)):
                char_code = chr(ord('a') + i)
                key_min = f"val_task_{char_code}_output_min_class_accuracy"
                key_bal = f"val_task_{char_code}_output_balanced_accuracy"
                key_acc = f"val_task_{char_code}_output_accuracy"
                
                if key_min in history.history:
                    task_acc_keys.append(key_min)
                elif key_bal in history.history:
                    task_acc_keys.append(key_bal)
                elif key_acc in history.history:
                    task_acc_keys.append(key_acc)
        
        if task_acc_keys:
            num_epochs = len(history.history[task_acc_keys[0]])
            avg_scores = []
            for epoch in range(num_epochs):
                epoch_sum = 0
                count = 0
                for char_code in range(ord('a'), ord('a') + len(task_labels)):
                    key_min = f"val_task_{chr(char_code)}_output_min_class_accuracy"
                    if key_min in history.history:
                         epoch_sum += history.history[key_min][epoch]
                         count += 1
                    elif 'val_min_class_accuracy' in history.history:
                         epoch_sum += history.history['val_min_class_accuracy'][epoch]
                         count += 1
                         break 
                
                if count > 0:
                    avg_scores.append(epoch_sum / count)
                else:
                    epoch_sum_fallback = sum(history.history[k][epoch] for k in task_acc_keys)
                    avg_scores.append(epoch_sum_fallback / len(task_acc_keys))

            final_val_acc = max(avg_scores)
            best_epoch_idx = avg_scores.index(final_val_acc)
            best_epoch = best_epoch_idx + 1  # 1-indexed
            last_epoch_score = avg_scores[-1]  # 最終エポックのスコアを記録
            
            if args.fine_tune.lower() == 'true':
                print(f"FT_BEST_EPOCH: {best_epoch}")
                logger.info(f"FT Best Score: {final_val_acc:.4f} (at epoch {best_epoch}/{num_epochs})")
            else:
                print(f"BEST_EPOCH: {best_epoch}")
                logger.info(f"Best Target Score: {final_val_acc:.4f} (at epoch {best_epoch}/{num_epochs})")
        else:
            logger.warning(f"No validation accuracy keys found in history. Available keys: {history.history.keys()}")
            
    # --- 延長学習の判定 (Conditional Extension) ---
    # best_epochが最終エポック、または最終エポックのスコアがベストと同等の場合に延長
    # (avg_scores.index()は最初の出現を返すため、同スコアが複数ある場合の対策)
    is_best_at_last = (best_epoch == training_epochs) or (abs(last_epoch_score - final_val_acc) < 1e-8)
    
    temp_weights_path = 'temp_training_weights.weights.h5'
    
    if final_val_acc < 0.5 or is_best_at_last:
        reason = f"Score {final_val_acc:.4f} < 0.5" if final_val_acc < 0.5 else f"Best Epoch reached at last epoch ({training_epochs})"
        logger.info(f"--- Extension Training ({reason}) ---")
        
        # Save current best weights before extension
        model.save_weights(temp_weights_path)
        
        # LR固定 (Initial * 0.05)
        extension_lr = args.learning_rate * 0.05
        if hasattr(model.optimizer, 'lr'):
            tf.keras.backend.set_value(model.optimizer.lr, extension_lr)
        logger.info(f"Extension LR set to {extension_lr:.8f}")
        
        max_ext_epochs = 20
        best_ext_score = final_val_acc
        
        for i in range(max_ext_epochs):
            logger.info(f"Extension Epoch {i+1}/{max_ext_epochs}...")
            
            # 1 Epoch training (epochs=1 means run 1 epoch from scratch in this call)
            hist_ext = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1,
                verbose=2
            )
            
            # Score Calculation
            val_scores = []
            if hasattr(hist_ext, 'history'):
                history_keys = hist_ext.history
                # Iterate phases to find metric keys
                if len(task_labels) == 1:
                    if 'val_min_class_accuracy' in history_keys:
                        val_scores.append(history_keys['val_min_class_accuracy'][0])
                    elif 'val_balanced_accuracy' in history_keys:
                        val_scores.append(history_keys['val_balanced_accuracy'][0])
                    elif 'val_accuracy' in history_keys:
                        val_scores.append(history_keys['val_accuracy'][0])
                else:
                    for t_idx in range(len(task_labels)):
                        char_code = chr(ord('a') + t_idx)
                        key_min = f"val_task_{char_code}_output_min_class_accuracy"
                        key_bal = f"val_task_{char_code}_output_balanced_accuracy"
                        key_acc = f"val_task_{char_code}_output_accuracy"
                        
                        if key_min in history_keys:
                            val_scores.append(history_keys[key_min][0])
                        elif key_bal in history_keys:
                            val_scores.append(history_keys[key_bal][0])
                        elif key_acc in history_keys:
                            val_scores.append(history_keys[key_acc][0])
            
            current_ext_score = sum(val_scores) / len(val_scores) if val_scores else 0.0
            logger.info(f"Extension Score: {current_ext_score:.4f} (Best: {best_ext_score:.4f})")
            
            if current_ext_score > best_ext_score:
                best_ext_score = current_ext_score
                final_val_acc = best_ext_score
                model.save_weights(temp_weights_path) # Update best weights
            else:
                logger.info("Accuracy dropped. Stopping extension.")
                break
        
        # Restore best weights from extension phase
        if os.path.exists(temp_weights_path):
            model.load_weights(temp_weights_path)
            logger.info(f"Restored best extension weights (Final Score: {final_val_acc:.4f})")

    # クリーンアップ
    if os.path.exists(temp_weights_path):
        os.remove(temp_weights_path)

    # モデル保存 (Fine-tuning時のみ、seed付きファイル名で保存)
    if args.fine_tune.lower() == 'true':
        save_path = f'outputs/models/model_seed{args.seed}.keras'
        os.makedirs('outputs/models', exist_ok=True)
        model.save(save_path)
        logger.info(f"Fine-tuned model saved to {save_path}")

    # 最終結果出力
    # 上記ロジックで final_val_acc が計算されている場合がある
    logger.info(f"Final Score calculated as average across {len(task_labels)} tasks.")
    
    # 最終結果出力 (全タスク)
    # print(f"FINAL_VAL_ACCURACY: {final_val_acc}") # 計算方法を変更するため一旦コメントアウト

    # 全タスクのスコアを表示
    if 'history' in locals() and hasattr(history, 'history'):
        task_names = [chr(65+i) for i in range(len(task_labels))] # A, B, C...
        for char_code, task_label in zip(range(ord('a'), ord('a') + len(task_labels)), task_names):
            # Try Balanced Accuracy first
            task_key_min = f"val_task_{chr(char_code)}_output_min_class_accuracy"
            task_key_balanced = f"val_task_{chr(char_code)}_output_balanced_accuracy"
            task_key_acc = f"val_task_{chr(char_code)}_output_accuracy"
            
            if task_key_min in history.history:
                best_task_score = max(history.history[task_key_min])
                print(f"TASK_{task_label}_ACCURACY: {best_task_score:.4f} (MinClass)")
            elif task_key_balanced in history.history:
                best_task_score = max(history.history[task_key_balanced])
                print(f"TASK_{task_label}_ACCURACY: {best_task_score:.4f} (Balanced)")
            elif task_key_acc in history.history:
                best_task_score = max(history.history[task_key_acc])
                print(f"TASK_{task_label}_ACCURACY: {best_task_score:.4f} (Normal)")

    # --- 詳細なクラス別精度の出力 ---
    logger.info("Computing detailed per-class accuracy...")
    
    # バリデーションデータの予測
    val_preds = model.predict(val_ds, verbose=0)
    
    # Datasetから正解ラベルを回収
    val_labels_dict = {}
    for _, y_batch, *_ in val_ds: # y_batch is dict
        for k, v in y_batch.items():
            if k not in val_labels_dict: val_labels_dict[k] = []
            val_labels_dict[k].append(v.numpy())
            
    # Concatenate
    for k in val_labels_dict:
        val_labels_dict[k] = np.concatenate(val_labels_dict[k], axis=0)
        
    # 各タスクごとに計算
    # model.outputs の順序と shape を考慮
    output_names = [f'task_{chr(97+i)}_output' for i in range(len(task_labels))]
    
    # validation予測値の整形 (単一タスクの場合はリストではない可能性がある)
    if isinstance(val_preds, np.ndarray):
        val_preds = [val_preds] # リスト化
        
    for i, (task_name, class_names) in enumerate(zip(output_names, task_labels)):
        # 予測ラベル
        pred_probs = val_preds[i] # [N, NumClasses]
        pred_labels = np.argmax(pred_probs, axis=-1)
        
        # 正解ラベル
        true_labels_raw = val_labels_dict[task_name] # [N, ] or [N, NumClasses] (OneHot)
        
        # One-hotならインデックスに戻す
        if len(true_labels_raw.shape) > 1 and true_labels_raw.shape[-1] > 1:
            true_labels = np.argmax(true_labels_raw, axis=-1)
        else:
            true_labels = true_labels_raw.astype(int)
            
        print(f"\n--- Task {chr(65+i)} Details ({len(class_names)} classes) ---")
        
        # クラスごとの精度計算
        # Confusion Matrix的なものを手計算
        for cls_idx, cls_name in enumerate(class_names):
            # このクラスが正解であるインデックス
            mask = (true_labels == cls_idx)
            count = np.sum(mask)
            
            if count > 0:
                correct = np.sum(pred_labels[mask] == cls_idx)
                accuracy = correct / count
                print(f"  Class '{cls_name}': {accuracy:.4f} ({correct}/{count})")
            else:
                print(f"  Class '{cls_name}': N/A (0 samples)")

    logger.info("Detailed metrics complete.")
    
    # --- 最終スコア出力 ---
    # final_val_acc は history から計算した全エポック中のベスト（Phase1/Phase2含む）
    # model.predict による再計算は最終エポックの重みを使うため、
    # restore_best_weightsが効かないケースではベストと乖離する。
    # よって history ベースの final_val_acc を優先する。
    
    # 参考: 現在のモデル重みでの再計算スコアもログに出す
    task_min_accuracies = []
    for i, (task_name, class_names) in enumerate(zip(output_names, task_labels)):
        pred_probs = val_preds[i]
        pred_labels = np.argmax(pred_probs, axis=-1)
        
        true_labels_raw = val_labels_dict[task_name]
        if len(true_labels_raw.shape) > 1 and true_labels_raw.shape[-1] > 1:
            true_labels = np.argmax(true_labels_raw, axis=-1)
        else:
            true_labels = true_labels_raw.astype(int)
            
        class_accuracies = []
        for cls_idx, cls_name in enumerate(class_names):
            mask = (true_labels == cls_idx)
            count = np.sum(mask)
            if count > 0:
                correct = np.sum(pred_labels[mask] == cls_idx)
                accuracy = correct / count
                class_accuracies.append(accuracy)
                
        if class_accuracies:
            task_min_acc = min(class_accuracies)
            task_min_accuracies.append(task_min_acc)
            
    if task_min_accuracies:
        current_model_score = sum(task_min_accuracies) / len(task_min_accuracies)
        logger.info(f"Current model weights score (MinClass): {current_model_score:.8f}")
        logger.info(f"History best score (MinClass): {final_val_acc:.8f}")
        # historyのベストと現在のモデル重みスコアの大きい方を使用
        best_score = max(final_val_acc, current_model_score)
        print(f"FINAL_VAL_ACCURACY: {best_score:.8f}")
        logger.info(f"FINAL_VAL_ACCURACY (max of history/current): {best_score:.8f}")
    else:
        print(f"FINAL_VAL_ACCURACY: {final_val_acc}")

if __name__ == "__main__":
    main()
