"""
Person (7-Class Label) エラー分析スクリプト
"""
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import json
from tensorflow.keras.utils import custom_object_scope

# --- 設定 ---
DEFAULT_MODEL_PATH = 'best_person_model.keras'
VALIDATION_DIR = 'preprocessed_person/validation'
OUTPUT_DIR = 'error_analysis_person'

IMG_SIZE = 224
BATCH_SIZE = 32

# クラス名を動的に取得するためにプレースホルダー
CLASS_NAMES = [] 

def load_validation_data(class_names):
    """検証データを読み込み"""
    image_paths = []
    true_labels = []
    
    # クラス順序は class_names に従う (sort順)
    for folder_name in class_names:
        folder_path = os.path.join(VALIDATION_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        label_idx = class_names.index(folder_name)
        
        # Walk recursively to find images in person subdirectories
        for root, dirs, files in os.walk(folder_path):
            for img_file in files:
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, img_file))
                    true_labels.append(label_idx)
    
    return image_paths, true_labels


def analyze_data_distribution(image_paths, class_names):
    """検証データのクラス分布を分析"""
    folder_counts = defaultdict(int)
    for path in image_paths:
        # parent dir name is the class
        folder_name = os.path.basename(os.path.dirname(path))
        folder_counts[folder_name] += 1
    
    print("\n" + "=" * 60)
    print("検証データのクラス分布 (Person/7-Class)")
    print("=" * 60)
    
    total = sum(folder_counts.values())
    for label in class_names:
        cnt = folder_counts.get(label, 0)
        pct = (cnt / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {label:<10}: {cnt:>5} ({pct:>5.1f}%) {bar}")


def predict_batch(model, image_paths):
    """バッチで予測"""
    predictions = []
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = []
        
        for path in batch_paths:
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            batch_images.append(img)
        
        batch_tensor = tf.stack(batch_images)
        # Assuming model outputs probabilities
        preds = model.predict(batch_tensor, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        predictions.extend(pred_classes)
        
        if (i + BATCH_SIZE) % 100 == 0:
            print(f"Processed {min(i+BATCH_SIZE, len(image_paths))}/{len(image_paths)} images")
    
    return predictions


def create_confusion_matrix(true_labels, predictions, class_names):
    """混同行列を作成"""
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for t, p in zip(true_labels, predictions):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    
    # 割合計算
    cm_percent = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_percent[i] = cm[i] / row_sum * 100
    
    # コンソール出力
    print(f"\n  [Confusion Details]")
    for i, true_label in enumerate(class_names):
        row_sum = cm[i].sum()
        print(f"    正解={true_label} ({row_sum}件):")
        for j, pred_label in enumerate(class_names):
            cnt = cm[i][j]
            pct = cm_percent[i][j]
            if cnt > 0:
                marker = "✓" if i == j else "✗"
                print(f"      → {pred_label:<10}: {cnt:>4} ({pct:>5.1f}%) {marker}")
    
    # 可視化
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title('Person/7-Class Confusion Matrix (%)')
    plt.colorbar(label='%')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = 50
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_percent[i][j]
            text = f"{pct:.1f}%\n({cm[i][j]})"
            plt.text(j, i, text,
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=9,
                     color="white" if pct > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_person.png'))
    plt.close()
    
    # 精度計算
    correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    accuracy = correct / len(true_labels) if true_labels else 0
    
    # Balanced Accuracy
    per_class_acc = []
    for i in range(num_classes):
        row_sum = cm[i].sum()
        if row_sum > 0:
            per_class_acc.append(cm[i][i] / row_sum)
        else:
            per_class_acc.append(0.0) # No samples for this class
            
    balanced_accuracy = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0
    
    return cm, accuracy, balanced_accuracy, per_class_acc


def collect_errors(image_paths, true_labels, predictions, class_names):
    """エラー画像を収集"""
    errors = defaultdict(list)
    
    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        pred_label = predictions[i]
        
        if true_label != pred_label:
            key = f"true_{class_names[true_label]}_pred_{class_names[pred_label]}"
            errors[key].append(path)
    
    # エラー画像をコピー
    for error_type, paths in errors.items():
        error_type_dir = os.path.join(OUTPUT_DIR, error_type)
        os.makedirs(error_type_dir, exist_ok=True)
        
        for path in paths[:20]: # Limit to 20 per error type to save space
            dst = os.path.join(error_type_dir, os.path.basename(path))
            try:
                shutil.copy2(path, dst)
            except: pass
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="Person (7-Class) Error Analysis")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    
    MODEL_PATH = args.model
    
    print("=" * 60)
    print("Person (7-Class) Error Analysis Script")
    print("=" * 60)
    
    # 出力ディレクトリをクリーンアップ
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up old output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # クラス名の自動検出
    if not os.path.exists(VALIDATION_DIR):
        print(f"Error: Validation dir not found: {VALIDATION_DIR}")
        return
        
    global CLASS_NAMES
    CLASS_NAMES = sorted([d for d in os.listdir(VALIDATION_DIR) if os.path.isdir(os.path.join(VALIDATION_DIR, d))])
    print(f"Detected Classes: {CLASS_NAMES}")
    
    if not CLASS_NAMES:
        print("No classes found.")
        return
        
    # モデル読み込み (Custom objects対策)
    print(f"\nLoading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    # common.pyで定義したカスタムメトリクスが必要かもしれないのでダミー定義またはimport
    # ここではシンプルに custom_object_scope を使うか、compile=Falseで読み込む
    # compile=Falseならメトリクス定義は不要
    try:
        model = models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 検証データ読み込み
    print(f"\nLoading validation data from: {VALIDATION_DIR}")
    image_paths, true_labels = load_validation_data(CLASS_NAMES)
    print(f"Found {len(image_paths)} validation images.")
    
    if len(image_paths) == 0:
        print("Error: No validation images found.")
        return
    
    # データ分布を表示
    analyze_data_distribution(image_paths, CLASS_NAMES)
    
    # 予測実行
    print("\nRunning predictions...")
    predictions = predict_batch(model, image_paths)
    
    # 分析
    print("\n" + "=" * 60)
    print("--- Analysis Results ---")
    
    cm, accuracy, balanced_accuracy, per_class_acc = create_confusion_matrix(true_labels, predictions, CLASS_NAMES)
    
    print(f"\n  Accuracy:          {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f} ({balanced_accuracy*100:.1f}%)")
    print(f"\n  Per-class Accuracy:")
    for label, acc in zip(CLASS_NAMES, per_class_acc):
        print(f"    {label:<10}: {acc:.2%}")
    
    # エラー収集
    errors = collect_errors(image_paths, true_labels, predictions, CLASS_NAMES)
    
    error_summary = {k: len(v) for k, v in errors.items()}
    print(f"\nErrors: {sum(error_summary.values())} total")
    for error_type, count in sorted(error_summary.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")
    
    # レポート保存
    report = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'per_class_accuracy': dict(zip(CLASS_NAMES, per_class_acc)),
        'errors': error_summary
    }
    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
