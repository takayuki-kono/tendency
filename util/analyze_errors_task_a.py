"""
Task A 単タスクモデル用エラー分析スクリプト
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

# --- 設定 ---
DEFAULT_MODEL_PATH = 'best_sequential_model_task_a.keras'
VALIDATION_DIR = 'preprocessed_multitask/validation'
OUTPUT_DIR = 'error_analysis_task_a'

IMG_SIZE = 224
BATCH_SIZE = 32

# Task A ラベル
TASK_A_LABELS = ['a', 'b', 'c']


def load_validation_data():
    """検証データを読み込み"""
    image_paths = []
    true_labels = []
    
    for folder_name in os.listdir(VALIDATION_DIR):
        folder_path = os.path.join(VALIDATION_DIR, folder_name)
        if not os.path.isdir(folder_path) or len(folder_name) == 0:
            continue
        
        first_char = folder_name[0]
        if first_char not in TASK_A_LABELS:
            continue
        
        label_idx = TASK_A_LABELS.index(first_char)
        
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(folder_path, img_file))
                true_labels.append(label_idx)
    
    return image_paths, true_labels


def analyze_data_distribution(image_paths):
    """検証データのクラス分布を分析"""
    folder_counts = defaultdict(int)
    for path in image_paths:
        folder_name = os.path.basename(os.path.dirname(path))
        first_char = folder_name[0] if folder_name else '?'
        folder_counts[first_char] += 1
    
    print("\n" + "=" * 60)
    print("検証データのクラス分布 (Task A)")
    print("=" * 60)
    
    total = sum(folder_counts.values())
    for label in TASK_A_LABELS:
        cnt = folder_counts.get(label, 0)
        pct = (cnt / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {label}: {cnt:>5} ({pct:>5.1f}%) {bar}")


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
        preds = model.predict(batch_tensor, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        predictions.extend(pred_classes)
        
        print(f"Processed {min(i+BATCH_SIZE, len(image_paths))}/{len(image_paths)} images")
    
    return predictions


def create_confusion_matrix(true_labels, predictions):
    """混同行列を作成"""
    num_classes = len(TASK_A_LABELS)
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
    for i, true_label in enumerate(TASK_A_LABELS):
        row_sum = cm[i].sum()
        print(f"    正解={true_label} ({row_sum}件):")
        for j, pred_label in enumerate(TASK_A_LABELS):
            cnt = cm[i][j]
            pct = cm_percent[i][j]
            if cnt > 0:
                marker = "✓" if i == j else "✗"
                print(f"      → {pred_label}: {cnt:>4} ({pct:>5.1f}%) {marker}")
    
    # 可視化
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title('Task A Confusion Matrix (%)')
    plt.colorbar(label='%')
    tick_marks = np.arange(len(TASK_A_LABELS))
    plt.xticks(tick_marks, TASK_A_LABELS)
    plt.yticks(tick_marks, TASK_A_LABELS)

    thresh = 50
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_percent[i][j]
            text = f"{pct:.1f}%\n({cm[i][j]})"
            plt.text(j, i, text,
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=10,
                     color="white" if pct > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_task_a.png'))
    plt.close()
    
    # 精度計算
    correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    accuracy = correct / len(true_labels) if true_labels else 0
    
    # Balanced Accuracy（各クラスの正解率の平均）
    per_class_acc = []
    for i in range(num_classes):
        row_sum = cm[i].sum()
        if row_sum > 0:
            per_class_acc.append(cm[i][i] / row_sum)
    balanced_accuracy = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0
    
    return cm, accuracy, balanced_accuracy, per_class_acc


def collect_errors(image_paths, true_labels, predictions):
    """エラー画像を収集"""
    errors = defaultdict(list)
    
    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        pred_label = predictions[i]
        
        if true_label != pred_label:
            key = f"true_{TASK_A_LABELS[true_label]}_pred_{TASK_A_LABELS[pred_label]}"
            errors[key].append(path)
    
    # エラー画像をコピー
    for error_type, paths in errors.items():
        error_type_dir = os.path.join(OUTPUT_DIR, error_type)
        os.makedirs(error_type_dir, exist_ok=True)
        
        for path in paths[:50]:
            dst = os.path.join(error_type_dir, os.path.basename(path))
            shutil.copy2(path, dst)
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="Task A Error Analysis")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    
    MODEL_PATH = args.model
    
    print("=" * 60)
    print("Task A Error Analysis Script")
    print("=" * 60)
    
    # 出力ディレクトリをクリーンアップ
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up old output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # モデル読み込み
    print(f"\nLoading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Available .keras files:")
        for f in os.listdir('.'):
            if f.endswith('.keras'):
                print(f"  - {f}")
        return
    
    model = models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
    
    # 検証データ読み込み
    print(f"\nLoading validation data from: {VALIDATION_DIR}")
    image_paths, true_labels = load_validation_data()
    print(f"Found {len(image_paths)} validation images.")
    
    if len(image_paths) == 0:
        print("Error: No validation images found.")
        return
    
    # データ分布を表示
    analyze_data_distribution(image_paths)
    
    # 予測実行
    print("\nRunning predictions...")
    predictions = predict_batch(model, image_paths)
    
    # 分析
    print("\n" + "=" * 60)
    print("--- Task A Analysis Results ---")
    
    cm, accuracy, balanced_accuracy, per_class_acc = create_confusion_matrix(true_labels, predictions)
    
    print(f"\n  Accuracy:          {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f} ({balanced_accuracy*100:.1f}%)")
    print(f"\n  Per-class Accuracy:")
    for label, acc in zip(TASK_A_LABELS, per_class_acc):
        print(f"    {label}: {acc:.2%}")
    
    # エラー収集
    errors = collect_errors(image_paths, true_labels, predictions)
    
    error_summary = {k: len(v) for k, v in errors.items()}
    print(f"\nErrors: {sum(error_summary.values())} total")
    for error_type, count in sorted(error_summary.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")
    
    # レポート保存
    report = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'per_class_accuracy': dict(zip(TASK_A_LABELS, per_class_acc)),
        'errors': error_summary
    }
    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
