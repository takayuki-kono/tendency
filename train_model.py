import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import logging
import kerastuner as kt # KerasTunerをインポート

# --- モデルのインポート ---
# 使用したいモデルに応じて、対応する行のコメントを解除してください
from tensorflow.keras.applications import EfficientNetV2B0, ResNet50V2, Xception, DenseNet121

# --- 前処理関数のインポート ---
# 使用したいモデルに応じて、対応する行のコメントを解除してください
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess


# ===== モデル選択 =====
# ここで使用したいモデル名を設定してください
# 利用可能なモデル: 'EfficientNetV2B0', 'ResNet50V2', 'Xception', 'DenseNet121'0.40
MODEL_TO_USE = 'DenseNet121'
# =====================


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'train_log_{MODEL_TO_USE}.txt', mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

# ディレクトリのパス
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed/validation'

# 設定
img_size = 224

# データ拡張設定
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = [0.9, 1.1]

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

def create_transfer_model(model_name, input_shape=(224, 224, 3), num_classes=4):
    model_map = {
        'EfficientNetV2B0': EfficientNetV2B0,
        'ResNet50V2': ResNet50V2,
        'Xception': Xception,
        'DenseNet121': DenseNet121
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unsupported model name: {model_name}. Available models: {list(model_map.keys())}")

    BaseCnnModel = model_map[model_name]
    base_model = BaseCnnModel(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def get_preprocessing_function(model_name):
    preprocess_map = {
        'EfficientNetV2B0': efficientnet_preprocess,
        'ResNet50V2': resnet_preprocess,
        'Xception': xception_preprocess,
        'DenseNet121': densenet_preprocess
    }
    if model_name not in preprocess_map:
        raise ValueError(f"Unsupported model name: {model_name}. Available models: {list(preprocess_map.keys())}")
    return preprocess_map[model_name]

def build_model_for_tuning(hp):
    """KerasTunerが呼び出すモデル構築関数"""
    # --- 1. データ拡張の探索範囲を定義 ---
    rotation_range = hp.Int('rotation_range', min_value=0, max_value=20, step=5)
    width_shift_range = hp.Float('width_shift_range', min_value=0.0, max_value=0.2, step=0.05)
    height_shift_range = hp.Float('height_shift_range', min_value=0.0, max_value=0.2, step=0.05)
    zoom_range = hp.Float('zoom_range', min_value=0.0, max_value=0.2, step=0.05)
    horizontal_flip = hp.Boolean('horizontal_flip')
    
    # --- 2. 学習率の探索範囲を定義 ---
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    # --- 3. モデルを構築 ---
    num_classes = get_num_classes(PREPROCESSED_TRAIN_DIR)
    model = create_transfer_model(MODEL_TO_USE, input_shape=(img_size, img_size, 3), num_classes=num_classes)
    
    train_class_weights = compute_class_weights(PREPROCESSED_TRAIN_DIR)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=weighted_sparse_categorical_crossentropy(train_class_weights),
        metrics=['accuracy']
    )
    return model

def main():
    try:
        logger.info(f"Starting hyperparameter tuning for model: {MODEL_TO_USE}")
        
        num_classes = get_num_classes(PREPROCESSED_TRAIN_DIR)
        if num_classes < 2:
            logger.error(f"Not enough classes found. Aborting.")
            return

        # クラス重みをmain関数のスコープで計算
        train_class_weights = compute_class_weights(PREPROCESSED_TRAIN_DIR)

        # --- データジェネレータの準備 (前処理のみ) ---
        # データ拡張の設定はチューナーが試行するため、ここでは設定しない
        preprocessing_function = get_preprocessing_function(MODEL_TO_USE)
        train_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        validation_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        
        BATCH_SIZE = 32 # チューニング中は固定
        train_generator = train_datagen.flow_from_directory(
            PREPROCESSED_TRAIN_DIR, target_size=(img_size, img_size), batch_size=BATCH_SIZE, class_mode='sparse', color_mode='rgb'
        )
        validation_generator = validation_datagen.flow_from_directory(
            PREPROCESSED_VALIDATION_DIR, target_size=(img_size, img_size), batch_size=BATCH_SIZE, class_mode='sparse', color_mode='rgb'
        )

        # --- KerasTunerのセットアップ ---
        tuner = kt.Hyperband(
            build_model_for_tuning,
            objective='val_accuracy',
            max_epochs=20,  # 1回の試行での最大エポック数
            factor=3,
            directory='kerastuner_dir',
            project_name=f'tuning_{MODEL_TO_USE}'
        )

        stop_early = EarlyStopping(monitor='val_loss', patience=5)
        
        logger.info("--- Starting Hyperparameter Search ---")
        tuner.search(train_generator, validation_data=validation_generator, callbacks=[stop_early])

        # --- 最適なハイパーパラメータを取得 ---
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info("--- Hyperparameter Search Complete ---")
        logger.info(f"Optimal rotation_range: {best_hps.get('rotation_range')}")
        logger.info(f"Optimal width_shift_range: {best_hps.get('width_shift_range')}")
        logger.info(f"Optimal height_shift_range: {best_hps.get('height_shift_range')}")
        logger.info(f"Optimal zoom_range: {best_hps.get('zoom_range')}")
        logger.info(f"Optimal horizontal_flip: {best_hps.get('horizontal_flip')}")
        logger.info(f"Optimal learning_rate: {best_hps.get('learning_rate')}")

        # --- 最適なパラメータで最終的なモデルを学習 ---
        logger.info("--- Training final model with best hyperparameters ---")
        
        # 最適なデータ拡張でジェネレータを再作成
        final_train_datagen = ImageDataGenerator(
            rotation_range=best_hps.get('rotation_range'),
            width_shift_range=best_hps.get('width_shift_range'),
            height_shift_range=best_hps.get('height_shift_range'),
            zoom_range=best_hps.get('zoom_range'),
            horizontal_flip=best_hps.get('horizontal_flip'),
            brightness_range=[0.9, 1.1],
            fill_mode='nearest',
            preprocessing_function=preprocessing_function
        )
        final_train_generator = final_train_datagen.flow_from_directory(
            PREPROCESSED_TRAIN_DIR, target_size=(img_size, img_size), batch_size=BATCH_SIZE, class_mode='sparse', color_mode='rgb'
        )

        model = tuner.hypermodel.build(best_hps)
        
        model_filename = f'best_model_{MODEL_TO_USE}.keras'
        tflite_filename = f'model_{MODEL_TO_USE}.tflite'
        
        model_checkpoint = ModelCheckpoint(model_filename, monitor='val_accuracy', save_best_only=True, verbose=1)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=6, min_lr=1e-6, verbose=1)

        history = model.fit(
            final_train_generator,
            validation_data=validation_generator,
            epochs=100, # 最終学習は多めのエポックで
            callbacks=[model_checkpoint, early_stopping, reduce_lr]
        )
        
        # 学習結果を一度変数に保存
        final_train_acc = history.history['accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])

        # ログファイルにはこれまで通り記録
        logger.info(f"Final Training accuracy: {final_train_acc}")
        logger.info(f"Best Validation accuracy: {best_val_acc}")

        # --- ここからファインチューニング処理を追加 ---
        logger.info("--- Starting Fine-tuning ---")
        
        # 最良のモデルをロード
        best_model = tf.keras.models.load_model(model_filename, custom_objects={'loss': weighted_sparse_categorical_crossentropy(train_class_weights)})
        
        # ベースモデルを再学習可能に設定
        best_model.trainable = True
        
        # ファインチューニング用の非常に低い学習率で再コンパイル
        best_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 低い学習率
            loss=weighted_sparse_categorical_crossentropy(train_class_weights),
            metrics=['accuracy']
        )
        
        logger.info("Fine-tuning with base model unfrozen.")
        best_model.summary(print_fn=logger.info)
        
        # 追加で数エポック学習
        history_fine_tune = best_model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=history.epoch[-1] + 15, # 現在のエポックからさらに15エポック
            initial_epoch=history.epoch[-1] + 1,
            callbacks=[model_checkpoint, early_stopping, reduce_lr] # コールバックを再利用
        )
        
        # ファインチューニング後の最良スコアを取得
        best_val_acc = max(history_fine_tune.history['val_accuracy'])
        final_train_acc = history_fine_tune.history['accuracy'][-1]
        # --- ファインチューニング処理ここまで ---

        best_model = tf.keras.models.load_model(model_filename, custom_objects={'loss': weighted_sparse_categorical_crossentropy(train_class_weights)})
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(tflite_filename, 'wb') as f:
            f.write(tflite_model)

        logger.info(f"Model converted to TensorFlow Lite format and saved as '{tflite_filename}'.")

        # 全ての処理が終わった後、ターミナルに最終結果を出力
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print("="*50)


    except Exception as e:
        logger.error(f"Error processing training: {e}", exc_info=True)

if __name__ == "__main__":
    main()