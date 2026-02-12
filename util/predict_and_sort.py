import os
import sys
import shutil
import argparse
import numpy as np
import tensorflow as tf
import cv2
from glob import glob
from tqdm import tqdm

# 親ディレクトリをパスに追加して components からインポート可能にする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# カスタムメトリクスを読み込むためにインポート
try:
    from components.train_multitask_trial import BalancedSparseCategoricalAccuracy
except ImportError:
    # 読み込めない場合はダミーを定義 (予測時はメトリクス計算しないので)
    print("Warning: Could not import BalancedSparseCategoricalAccuracy. Defining dummy.")
    class BalancedSparseCategoricalAccuracy(tf.keras.metrics.Metric):
        def __init__(self, num_classes, name='balanced_accuracy', **kwargs):
            super().__init__(name=name, **kwargs)
        def update_state(self, y_true, y_pred, sample_weight=None): pass
        def result(self): return 0.0
        def reset_state(self): pass

# --- 設定 ---
IMG_SIZE = 224
BATCH_SIZE = 32

# ラベル定義 (train_multitask_trial.py と合わせる)
TASK_A_LABELS = ['a', 'b', 'c']
TASK_B_LABELS = ['d', 'e']
TASK_C_LABELS = ['f', 'g']
TASK_D_LABELS = ['h', 'i']
ALL_TASK_LABELS = [TASK_A_LABELS, TASK_B_LABELS, TASK_C_LABELS, TASK_D_LABELS]
TASK_NAMES = ['Task A', 'Task B', 'Task C', 'Task D']

def load_prediction_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model: {model_path}")
    try:
        # custom_objects に必要なクラスを指定
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'BalancedSparseCategoricalAccuracy': BalancedSparseCategoricalAccuracy}
        )
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        # compile=False で読み込み試行（学習しないならこれでOKな場合も）
        try:
            print("Trying to load with compile=False...")
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded with compile=False.")
            return model
        except Exception as e2:
            print(f"Still failed: {e2}")
            sys.exit(1)

def preprocess_image(path):
    try:
        # 日本語パス対応
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            
        if img is None: return None
        
        # Resize & Convert
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Model expects RGB usually? Check preprocess.
        
        # EfficientNetV2などは内部またはpreprocess_inputで処理するが、
        # ここではモデルにRescalingが含まれているか、preprocess_inputが必要か確認が必要。
        # analyze_errors.py では tf.io.read_file だけで preprocess_input 呼んでない？
        # train_multitask_trial.py を見ると、モデル内に `layers.Lambda(preprocess_func)` があるので
        # Raw RGB (0-255) を渡せばOK
        
        return img
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Classify images in a folder using the latest model.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images to classify')
    parser.add_argument('--output_dir', type=str, default='outputs/sorted_predictions', help='Directory to save sorted images')
    parser.add_argument('--model', type=str, default='best_sequential_model.keras', help='Path to the model file')
    parser.add_argument('--action', type=str, choices=['copy', 'move', 'print'], default='copy', help='Action: copy files, move files, or just print results')
    parser.add_argument('--task_index', type=int, default=0, help='Task index to use for sorting (0=Task A, 1=Task B, etc.). Default 0.')
    parser.add_argument('--confidence_thresh', type=float, default=0.0, help='Minimum confidence to sort (below this goes to "uncertain")')
    
    args = parser.parse_args()
    
    # モデル読み込み
    model = load_prediction_model(args.model)
    
    # 画像リスト取得
    print(f"Scanning {args.input_dir}...")
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob(os.path.join(args.input_dir, '**', ext), recursive=True))
    
    if not image_paths:
        print("No images found.")
        return
        
    print(f"Found {len(image_paths)} images.")
    
    # ターゲットタスクの情報
    if args.task_index >= len(ALL_TASK_LABELS):
        print(f"Error: Invalid task_index {args.task_index}. Max is {len(ALL_TASK_LABELS)-1}.")
        return
        
    target_labels = ALL_TASK_LABELS[args.task_index]
    task_name = TASK_NAMES[args.task_index]
    print(f"Sorting based on: {task_name} (Labels: {target_labels})")
    
    
    # 統計用
    from collections import defaultdict
    stats = defaultdict(int)
    total_processed = 0
    confidences = defaultdict(list)

    # バッチ処理で予測
    num_images = len(image_paths)
    for i in range(0, num_images, BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = []
        valid_paths = []
        
        for path in batch_paths:
            img = preprocess_image(path)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(path)
        
        if not batch_images: continue
        
        batch_tensor = np.array(batch_images)
        preds = model.predict(batch_tensor, verbose=0)
        
        # preds は [task_a_out, task_b_out, ...] のリストになっているはず
        # 特定タスクの出力を取得
        target_preds = preds[args.task_index]
        
        for j, pred_probs in enumerate(target_preds):
            current_path = valid_paths[j]
            filename = os.path.basename(current_path)
            
            # クラス判定
            class_idx = np.argmax(pred_probs)
            confidence = pred_probs[class_idx]
            label_name = target_labels[class_idx]
            
            if confidence < args.confidence_thresh:
                label_name = "uncertain"
            
            # 統計更新
            stats[label_name] += 1
            total_processed += 1
            confidences[label_name].append(confidence)

            if args.action == 'print':
                print(f"{filename} -> {label_name} ({confidence:.2%})")
            else:
                # 出力先: output_dir/task_name/label_name/filename
                dest_dir = os.path.join(args.output_dir, task_name.replace(" ", "_"), label_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, filename)
                
                try:
                    if args.action == 'copy':
                        shutil.copy2(current_path, dest_path)
                    elif args.action == 'move':
                        shutil.move(current_path, dest_path)
                except Exception as e:
                    print(f"Error {args.action}ing {filename}: {e}")

        # 進捗表示
        print(f"Processed {min(i+BATCH_SIZE, num_images)}/{num_images}...", end='\r')
        
    print(f"\n\nProcessing complete. Results in {args.output_dir}")
    print("="*50)
    print(f"PREDICTION SUMMARY (Task: {task_name})")
    print("="*50)
    print(f"{'Label':<15} | {'Count':<8} | {'Ratio':<8} | {'Avg Conf':<8}")
    print("-" * 50)
    
    # Sort by label index if possible, otherwise alphabetical
    sorted_labels = sorted(stats.keys())
    
    # Try to sort using the original label order, "uncertain" last
    def sort_key(k):
        if k == "uncertain": return 999
        if k in target_labels: return target_labels.index(k)
        return 0
    sorted_labels = sorted(stats.keys(), key=sort_key)

    for label in sorted_labels:
        count = stats[label]
        ratio = (count / total_processed * 100) if total_processed > 0 else 0
        avg_conf = np.mean(confidences[label]) if confidences[label] else 0.0
        print(f"{label:<15} | {count:<8} | {ratio:>6.1f}% | {avg_conf:.1%}")
    
    print("-" * 50)
    print(f"{'TOTAL':<15} | {total_processed:<8} | {'100.0%':<8} |")
    print("="*50)

if __name__ == "__main__":
    main()
