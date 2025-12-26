# --- 実行コマンド例 ---
# python predict.py model_DenseNet121.tflite predict
# --------------------

import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
import logging
from glob import glob

# --- モデルの前処理関数をインポート ---
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- 設定 ---
IMG_SIZE = 224
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'

def get_preprocessing_function(model_name):
    """モデル名から適切な前処理関数を返す"""
    preprocess_map = {
        'EfficientNetV2B0': efficientnet_preprocess,
        'ResNet50V2': resnet_preprocess,
        'Xception': xception_preprocess,
        'DenseNet121': densenet_preprocess
    }
    for name, func in preprocess_map.items():
        if name.lower() in model_name.lower():
            logging.info(f"Using preprocessing function for: {name}")
            return func
    raise ValueError(f"Could not find a preprocessing function for model name: {model_name}. Supported models: {list(preprocess_map.keys())}")

def main(args):
    # 1. クラスラベルの読み込み
    if not os.path.exists(PREPROCESSED_TRAIN_DIR):
        logging.error(f"Training directory not found at: {PREPROCESSED_TRAIN_DIR}")
        logging.error("Please make sure the 'preprocessed/train' directory exists and contains the class subdirectories.")
        return

    class_names = sorted([name for name in os.listdir(PREPROCESSED_TRAIN_DIR) if os.path.isdir(os.path.join(PREPROCESSED_TRAIN_DIR, name))])
    if not class_names:
        logging.error(f"No class subdirectories found in {PREPROCESSED_TRAIN_DIR}")
        return
    logging.info(f"Found class labels: {class_names}")

    # 2. TensorFlow Liteモデルの読み込み
    try:
        interpreter = tf.lite.Interpreter(model_path=args.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        logging.error(f"Failed to load TFLite model: {e}")
        return

    # 3. 指定ディレクトリ内の画像ファイルを取得
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(args.input_dir, '**', ext), recursive=True))
    
    if not image_paths:
        logging.error(f"No images found in directory: {args.input_dir}")
        return
    
    logging.info(f"Found {len(image_paths)} images to process.")

    # 4. 各画像を処理し、予測結果を蓄積
    all_predictions = []
    preprocessing_function = get_preprocessing_function(args.model_path)

    for image_path in image_paths:
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.warning(f"Could not read image, skipping: {image_path}")
                continue
            
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_expanded = np.expand_dims(img_rgb, axis=0)
            img_preprocessed = preprocessing_function(img_expanded)
            
            # 判定の実行
            interpreter.set_tensor(input_details[0]['index'], img_preprocessed.astype(np.float32))
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            all_predictions.append(predictions[0])

        except Exception as e:
            logging.error(f"Failed to process image {image_path}: {e}")
            continue

    if not all_predictions:
        logging.error("No images were successfully processed.")
        return

    # 5. 予測結果の平均を計算
    avg_predictions = np.mean(all_predictions, axis=0)

    # 6. 平均結果の表示
    results = sorted(zip(class_names, avg_predictions), key=lambda x: x[1], reverse=True)

    print("\n" + "="*40)
    print(f"AVERAGE PREDICTION RESULT ({len(all_predictions)} images)")
    print("="*40)
    
    for class_name, confidence in results:
        print(f"{class_name:<15}: {confidence:.2%}")

    print("="*40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify all images in a directory and output the average result.')
    parser.add_argument('model_path', type=str, help='Path to the .tflite model file.')
    parser.add_argument('input_dir', type=str, help='Path to the directory containing images to classify.')
    
    args = parser.parse_args()
    main(args)
