"""
エラー分析スクリプト
学習済みモデルの予測ミスを可視化・分析する
"""
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
# import seaborn as sns # 削除
from collections import defaultdict
import argparse
import json

# --- 設定 ---
# 分析対象のモデル
DEFAULT_MODEL_PATH = 'best_sequential_model.keras'
# 検証データのディレクトリ
VALIDATION_DIR = 'preprocessed_multitask/validation'
# 出力ディレクトリ
OUTPUT_DIR = 'error_analysis'

# 画像設定
IMG_SIZE = 224
BATCH_SIZE = 32

# ラベル定義（train_for_filter_search.py と同じ）
TASK_A_LABELS = ['a', 'b', 'c']
TASK_B_LABELS = ['d', 'e']
TASK_C_LABELS = ['f', 'g']
TASK_D_LABELS = ['h', 'i']
ALL_TASK_LABELS = [TASK_A_LABELS, TASK_B_LABELS, TASK_C_LABELS, TASK_D_LABELS]
TASK_NAMES = ['Task A', 'Task B', 'Task C', 'Task D']


def analyze_data_distribution(image_paths):
    """検証データのタスクごとのクラス分布を分析"""
    from collections import defaultdict
    
    # フォルダ名からカウントを取得
    folder_counts = defaultdict(int)
    for path in image_paths:
        folder_name = os.path.basename(os.path.dirname(path))
        folder_counts[folder_name] += 1
    
    print("\n" + "=" * 60)
    print("検証データのクラス分布 (Data Distribution)")
    print("=" * 60)
    
    for task_idx, (task_labels, task_name) in enumerate(zip(ALL_TASK_LABELS, TASK_NAMES)):
        label_counts = {label: 0 for label in task_labels}
        
        for folder_name, count in folder_counts.items():
            if len(folder_name) > task_idx:
                char = folder_name[task_idx]
                if char in label_counts:
                    label_counts[char] += count
        
        task_total = sum(label_counts.values())
        
        print(f"\n--- {task_name} ---")
        for label in task_labels:
            cnt = label_counts[label]
            pct = (cnt / task_total * 100) if task_total > 0 else 0
            bar = "█" * int(pct / 5)  # 5%ごとに1ブロック
            print(f"  {label}: {cnt:>5} ({pct:>5.1f}%) {bar}")

def load_validation_data():
    """検証データを読み込み、画像パスとラベルを取得"""
    image_paths = []
    true_labels = []  # [(task_a, task_b, task_c, task_d), ...]
    
    for multi_label in os.listdir(VALIDATION_DIR):
        label_dir = os.path.join(VALIDATION_DIR, multi_label)
        if not os.path.isdir(label_dir):
            continue
        
        # マルチラベルから各タスクのラベルを抽出
        if len(multi_label) < len(ALL_TASK_LABELS):
            continue
            
        task_labels = []
        for i, task_label_list in enumerate(ALL_TASK_LABELS):
            char = multi_label[i]
            if char in task_label_list:
                task_labels.append(task_label_list.index(char))
            else:
                task_labels.append(-1)  # 不明
        
        for img_file in os.listdir(label_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(label_dir, img_file))
                true_labels.append(tuple(task_labels))
    
    return image_paths, true_labels


def predict_batch(model, image_paths):
    """バッチで予測を実行"""
    predictions = [[] for _ in range(len(ALL_TASK_LABELS))]
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = []
        
        for path in batch_paths:
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            batch_images.append(img)
        
        batch_tensor = tf.stack(batch_images)
        
        # 予測
        preds = model.predict(batch_tensor, verbose=0)
        
        # 各タスクの予測を保存
        for task_idx in range(len(ALL_TASK_LABELS)):
            pred_classes = np.argmax(preds[task_idx], axis=1)
            predictions[task_idx].extend(pred_classes)
        
        print(f"Processed {min(i+BATCH_SIZE, len(image_paths))}/{len(image_paths)} images")
    
    return predictions


def create_confusion_matrix(true_labels, predictions, task_idx, task_labels, task_name):
    """混同行列を作成・保存"""
    # from sklearn.metrics import confusion_matrix # 削除
    
    true_task = [t[task_idx] for t in true_labels]
    pred_task = predictions[task_idx]
    
    # 有効なラベルのみフィルタ
    valid_indices = [i for i, t in enumerate(true_task) if t >= 0]
    true_filtered = [true_task[i] for i in valid_indices]
    pred_filtered = [pred_task[i] for i in valid_indices]
    
    # 混同行列を自前で計算
    num_classes = len(task_labels)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_filtered, pred_filtered):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    
    # 行ごとの割合を計算（各正解ラベルに対する予測分布）
    cm_percent = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_percent[i] = cm[i] / row_sum * 100
    
    # コンソールに詳細表示
    print(f"\n  [Confusion Details]")
    for i, true_label in enumerate(task_labels):
        row_sum = cm[i].sum()
        print(f"    正解={true_label} ({row_sum}件):")
        for j, pred_label in enumerate(task_labels):
            cnt = cm[i][j]
            pct = cm_percent[i][j]
            if cnt > 0:
                marker = "✓" if i == j else "✗"
                print(f"      → {pred_label}: {cnt:>4} ({pct:>5.1f}%) {marker}")
    
    # 可視化 (matplotlibのみ使用) - 割合表示バージョン
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title(f'{task_name} Confusion Matrix (%)')
    plt.colorbar(label='%')
    tick_marks = np.arange(len(task_labels))
    plt.xticks(tick_marks, task_labels)
    plt.yticks(tick_marks, task_labels)

    # 数字をプロット（割合%）
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
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{task_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # 精度計算
    correct = sum(1 for t, p in zip(true_filtered, pred_filtered) if t == p)
    accuracy = correct / len(true_filtered) if true_filtered else 0
    
    return cm, accuracy


def collect_errors(image_paths, true_labels, predictions, task_idx, task_labels, task_name):
    """エラー画像を収集・整理"""
    errors = defaultdict(list)
    
    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        true_class = true_label[task_idx]
        pred_class = predictions[task_idx][i]
        
        if true_class >= 0 and true_class != pred_class:
            key = f"true_{task_labels[true_class]}_pred_{task_labels[pred_class]}"
            errors[key].append(path)
    
    # エラー画像をコピー
    task_error_dir = os.path.join(OUTPUT_DIR, task_name.lower().replace(" ", "_"))
    os.makedirs(task_error_dir, exist_ok=True)
    
    for error_type, paths in errors.items():
        error_type_dir = os.path.join(task_error_dir, error_type)
        os.makedirs(error_type_dir, exist_ok=True)
        
        for path in paths[:50]:  # 最大50枚
            dst = os.path.join(error_type_dir, os.path.basename(path))
            shutil.copy2(path, dst)
    
    return errors


def analyze_per_combination(image_paths, true_labels, predictions):
    """マルチラベル組み合わせ単位での精度を分析"""
    from collections import defaultdict
    
    # フォルダ名（組み合わせ）を抽出
    combination_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        # フォルダ名を取得
        folder_name = os.path.basename(os.path.dirname(path))
        
        # 4タスク全てが正解か判定
        all_correct = True
        for task_idx in range(len(ALL_TASK_LABELS)):
            if true_label[task_idx] != predictions[task_idx][i]:
                all_correct = False
                break
        
        combination_stats[folder_name]['total'] += 1
        if all_correct:
            combination_stats[folder_name]['correct'] += 1
    
    # 精度を計算してソート
    results = {}
    for combo, stats in combination_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        results[combo] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total']
        }
    
    # 精度順にソート（昇順）
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['accuracy']))
    return sorted_results

def main():
    parser = argparse.ArgumentParser(description="Error Analysis Script")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to the model file')
    args = parser.parse_args()
    
    MODEL_PATH = args.model

    print("=" * 60)
    print("Error Analysis Script")
    print("=" * 60)
    
    # 出力ディレクトリをクリーンアップ
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up old output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    # 出力ディレクトリ作成
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
    
    # 各タスクの分析
    report = {}
    print("\n" + "=" * 60)
    print("Analysis Results")
    print("=" * 60)
    
    for task_idx, (task_labels, task_name) in enumerate(zip(ALL_TASK_LABELS, TASK_NAMES)):
        print(f"\n--- {task_name} ---")
        
        # 混同行列
        cm, accuracy = create_confusion_matrix(
            true_labels, predictions, task_idx, task_labels, task_name
        )
        print(f"Accuracy: {accuracy:.4f}")
        
        # エラー収集
        errors = collect_errors(
            image_paths, true_labels, predictions, task_idx, task_labels, task_name
        )
        
        error_summary = {k: len(v) for k, v in errors.items()}
        print(f"Errors: {sum(error_summary.values())} total")
        for error_type, count in sorted(error_summary.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
        
        report[task_name] = {
            'accuracy': accuracy,
            'errors': error_summary
        }
    
    # マルチラベル組み合わせ単位の分析
    print("\n--- Per-Combination Accuracy (全タスク正解率) ---")
    combo_results = analyze_per_combination(image_paths, true_labels, predictions)
    report['combination_accuracy'] = combo_results
    
    for combo, stats in combo_results.items():
        acc_pct = stats['accuracy'] * 100
        print(f"  {combo}: {acc_pct:.1f}% ({stats['correct']}/{stats['total']})")
    
    # レポート保存
    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}/")
    print("  - confusion_matrix_*.png: 混同行列の可視化")
    print("  - task_*/: エラー画像（正解_予測でフォルダ分け）")
    print("  - report.json: 分析レポート")
    print("=" * 60)


if __name__ == "__main__":
    main()
