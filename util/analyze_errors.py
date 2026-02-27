"""
エラー分析スクリプト
学習済みモデルの予測ミスを可視化・分析する。
デフォルト: 前処理前の train と validation の両方で実行（out_dir/train/, out_dir/val/ に出力）。
単一ディレクトリのみ分析する場合は --data_dir を指定する。
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
import cv2
import pickle
import json
# from insightface.app import FaceAnalysis # 削除: Protobuf Error回避のためキャッシュ利用

# --- 設定 ---
# 分析対象のモデル
DEFAULT_MODEL_PATH = 'outputs/models/best_sequential_model.keras'
# デフォルト: 前処理前の train / validation の両方で実行
VALIDATION_DIR = 'preprocessed_multitask/validation'
DEFAULT_TRAIN_DIR = 'train'
DEFAULT_VAL_DIR = 'validation'
# 出力ディレクトリ
OUTPUT_DIR = 'error_analysis'
CACHE_DIR = 'outputs/cache'

# 画像設定
IMG_SIZE = 224
BATCH_SIZE = 32

# 指標計算用 (InsightFace定数は不要になったが、辞書キーとして使うので名前だけ残す意味はない)
# FACE_APP = None # 削除

# ラベル定義（動的に更新されるため初期値は空）
ALL_TASK_LABELS = []
TASK_NAMES = []


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

def load_validation_data(data_dir):
    """検証データを読み込み、画像パスとラベルを取得 (シングル/マルチタスク対応)"""
    global ALL_TASK_LABELS, TASK_NAMES
    
    image_paths = []
    true_labels = []  # [(task_0_label, ...), ...]
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return [], []

    # ディレクトリ構造を解析してタスク定義を更新
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs:
        return [], []
        
    first_dir = subdirs[0]
    
    # シングルタスク判定: ディレクトリ名がそのままクラス名の場合 (例: 'a', 'b', ...)
    # またはマルチタスク: ディレクトリ名が文字連結 (例: 'adfh', 'aefh', ...)
    # 決定ロジック:
    # 全てのサブディレクトリの長さが同じで、かつ文字連結構造とみなせるか？
    # data_dir直下がクラスフォルダ（a, b, ...）ならシングルタスク
    # data_dir直下が組み合わせフォルダ（abcd）ならマルチタスク
    
    # 今回はシンプルに、train_multitask_trial.py と同様のロジックで判定
    # シングルタスクモードを優先的に判定したいが、外部フラグがないので推測する。
    # 既存の仕様では、マルチタスクの組み合わせフォルダはすべて同じ長さ(4文字など)になる。
    # シングルタスクの場合は、'a', 'z' など長さ1もありうるし、'classA' などもありうる。
    
    # ここでは、ディレクトリ内の画像有無で判定する
    # data_dir/subdir/image.jpg exists? -> Single Task Struct
    # data_dir/subdir/subdir_person/image.jpg exists? -> Multi Task Struct (Preprocessed format)
    # data_dir/subdir/image.jpg exists -> Single Task Struct (Preprocessed format)
    
    # 実際の前処理済みフォルダ構造:
    # Multi: preprocessed/val/adfh/person/image.jpg OR preprocessed/val/adfh/image.jpg
    # Single: preprocessed/val/a/image.jpg
    
    # 判定: フォルダ名を解析してタスク構造を推定
    # 1. 全部長さが同じ > 1 かつ 構成文字種が限定的 -> Multi?
    # 2. それ以外 -> Single
    
    # より確実な方法: モデルのOutput数と合わせるべきだが、モデルロード前なのでデータから推測
    
    # Try to treat as Multi-Task first if lengths match and > 1
    is_multitask_struct = True
    dir_len = len(first_dir)
    if dir_len <= 1: 
        is_multitask_struct = False
    else:
        for d in subdirs:
            if len(d) != dir_len:
                is_multitask_struct = False
                break
    
    # しかし、ユーザーが指定した Single Task Mode (フォルダ名=クラス) の場合、長さがバラバラなこともあろうが、
    # 今回の文脈では single_task_mode=True の場合、フォルダ名=クラス名として扱う。
    # Analyze_errors.py は汎用的にしたい。
    
    # 簡易実装:
    # フォルダ名を収集し、文字位置ごとにユニークな文字を集めてタスク定義を作る (Multi)
    # または、フォルダ名そのものをクラスリストとする (Single)
    
    # ここでは、「フォルダ名そのものを1つのタスクのクラス」として扱うモード (Single Task / Flat Class) をデフォルトとし、
    # 特定の条件（文字数一定など）でマルチタスク分解を試みるスイッチ、あるいは引数が欲しいところだが...
    
    # 以前のコードとの互換性のため、ALL_TASK_LABELSが空なら推論する
    
    inferred_tasks = []
    
    # 1. マルチタスク解釈を試行
    if is_multitask_struct:
        temp_tasks = [set() for _ in range(dir_len)]
        valid = True
        for d in subdirs:
            for i, c in enumerate(d):
                temp_tasks[i].add(c)
        
        # タスクごとのクラス数が妥當か？ (例えば全て1クラスだけならマルチタスクの意味なし)
        # ここは決め打ちで、「フォルダ名リスト」をそのままシングルタスクとして扱うほうが安全かもしれない
        # が、既存のマルチタスクモデルの評価には分解が必要。
        
        # ユーザーの意図: 今回は Single Task Mode で学習したモデルを評価したい。
        # データセットも Single Task 構造 (a, b...) なので is_multitask_struct = False になるはず。
        # もしデータセットが adfh 等のまま Single Task Mode True で学習したなら？
        # -> train_multitask_trial.py では「ディレクトリ名をそのままクラス名」として扱っている。
        # つまり adfh というクラス、 aefh というクラス... になる。
        
        # 結論: フォルダ名をそのままクラス名として扱う「シングルタスク」として解析し、
        # もしモデルがマルチ出力を持っていたらエラーになる、という形が自然。
        pass

    # 今回はモデルに合わせて動的にしたいが、ロード前。
    # 常に「フォルダ名＝クラス名」の1タスクとして扱うのが、
    # 以前の変更 (Single Task Mode) に追従する形になる。
    
    # ただし、既存のマルチタスク評価も壊したくない。
    # 妥協案: ディレクトリ名が全て len=4 かつ ... という判定は危険。
    
    # シンプルに: フォルダ名をソートしてクラスリストとし、タスク数=1 とする。
    # これで Single Task Mode の評価は可能。
    # Multi Task Mode のデータセット (adfh...) を評価する場合も、それを「1つのクラス」として扱うことになる。
    # 詳細なタスク別評価はできなくなるが、エラー分析としては機能する。
    
    # もし本当にマルチタスク分解が必要なら引数で指定すべき。
    
    # アップデート:
    sorted_subdirs = sorted(subdirs)
    ALL_TASK_LABELS = [sorted_subdirs]
    TASK_NAMES = ['Task A']
    
    classes = sorted_subdirs
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    print(f"Detected 1 task with {len(classes)} classes")
    
    for label_name in subdirs:
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir): continue
        
        idx = class_to_idx[label_name]
        
        # Recursive search or direct
        # Direct files
        files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            image_paths.append(os.path.join(label_dir, f))
            true_labels.append((idx,)) # Single task tuple
            
        # Recursive (person subdirs)
        for item in os.listdir(label_dir):
            item_path = os.path.join(label_dir, item)
            if os.path.isdir(item_path):
                files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for f in files:
                    image_paths.append(os.path.join(item_path, f))
                    true_labels.append((idx,))

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
        
        # モデル出力形式の正規化: シングルタスク(1出力)の場合でもリスト形式にする
        if not isinstance(preds, list):
            preds = [preds]
            
        # 各タスクの予測を保存
        for task_idx in range(len(ALL_TASK_LABELS)):
            if task_idx < len(preds):
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
    task_error_dir = os.path.join(OUTPUT_DIR, "errors", task_name.lower().replace(" ", "_"))
    os.makedirs(task_error_dir, exist_ok=True)
    
    for error_type, paths in errors.items():
        error_type_dir = os.path.join(task_error_dir, error_type)
        os.makedirs(error_type_dir, exist_ok=True)
        
        for path in paths:  # 無制限
            dst = os.path.join(error_type_dir, os.path.basename(path))
            shutil.copy2(path, dst)
    
    return errors


def collect_correct(image_paths, true_labels, predictions, task_idx, task_labels, task_name):
    """正解画像を収集・整理"""
    correct = defaultdict(list)
    
    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        true_class = true_label[task_idx]
        pred_class = predictions[task_idx][i]
        
        if true_class >= 0 and true_class == pred_class:
            key = task_labels[true_class]
            correct[key].append(path)
    
    # 正解画像をコピー
    task_correct_dir = os.path.join(OUTPUT_DIR, "correct", task_name.lower().replace(" ", "_"))
    os.makedirs(task_correct_dir, exist_ok=True)
    
    for label, paths in correct.items():
        label_dir = os.path.join(task_correct_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        for path in paths:  # 無制限
            try:
                dst = os.path.join(label_dir, os.path.basename(path))
                shutil.copy2(path, dst)
            except: pass
    
    return correct


def analyze_per_combination(image_paths, true_labels, predictions):
    """マルチラベル組み合わせ単位での精度を分析"""
    from collections import defaultdict
    
    # フォルダ名（組み合わせ）を抽出
    combination_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        # フォルダ名を取得
        folder_name = os.path.basename(os.path.dirname(path))
        
        # 全タスク正解判定
        all_correct = True
        for task_idx in range(len(ALL_TASK_LABELS)):
            if task_idx < len(true_label) and task_idx < len(predictions):
               if true_label[task_idx] != predictions[task_idx][i]:
                   all_correct = False
                   break
            else:
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


GLOBAL_METRICS_CACHE = {}

def load_metrics_cache():
    """outputs/cache/*.pkl からメトリクスを読み込む"""
    global GLOBAL_METRICS_CACHE
    if GLOBAL_METRICS_CACHE: return

    print(f"Loading metrics cache from {CACHE_DIR}...")
    if not os.path.exists(CACHE_DIR):
        print("Cache dir not found.")
        return

    loaded_files = 0
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".pkl") and fname.startswith("metrics_"):
            path = os.path.join(CACHE_DIR, fname)
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    # data is list of dicts: {'path':..., 'metrics':...}
                    for item in data:
                        if 'metrics' in item and item['metrics']:
                            # Key by filename
                            basename = os.path.basename(item['path'])
                            GLOBAL_METRICS_CACHE[basename] = item['metrics']
                    loaded_files += 1
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
    
    print(f"Loaded {loaded_files} cache files.")
    print(f"Total metrics in cache: {len(GLOBAL_METRICS_CACHE)} images.")
    # Debug: Print first 5 keys
    if GLOBAL_METRICS_CACHE:
        print("Sample cache keys:", list(GLOBAL_METRICS_CACHE.keys())[:5])

def get_metrics(img_path):
    """画像から指標を取得（キャッシュ優先）"""
    load_metrics_cache()
    
    basename = os.path.basename(img_path)
    
    # face_size をファイル名から抽出（キャッシュにない場合の保険）
    import re
    face_size = 0
    sz_match = re.search(r'_sz(\d+)', basename)
    if sz_match:
        face_size = int(sz_match.group(1))
    
    # 直接マッチを試行
    if basename in GLOBAL_METRICS_CACHE:
        result = GLOBAL_METRICS_CACHE[basename].copy()
        if 'face_size' not in result:
            result['face_size'] = face_size
        return result
    
    # 人物名プレフィックス除去を試行
    # 例: 倉科カナ_BaiduImageClient_xxx.jpg → BaiduImageClient_xxx.jpg
    parts = basename.split('_')
    if len(parts) >= 4:
        # 最初のパート（人物名）を除去
        stripped_key = '_'.join(parts[1:])
        if stripped_key in GLOBAL_METRICS_CACHE:
            result = GLOBAL_METRICS_CACHE[stripped_key].copy()
            if 'face_size' not in result:
                result['face_size'] = face_size
            return result
    
    # Cache Miss: 簡易計算のみ
    res = {
        'pitch': 0.0, 'symmetry': 0.0, 'y_diff': 0.0,
        'mouth_open': 0.0, 'eb_eye_dist': 0.0, 'sharpness': 0.0,
        'face_size': face_size  # 既に上で抽出済み
    }
    
    try:
        # 日本語パス対策
        with open(img_path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            
        if img is None: return res
        
        # Sharpness (InsightFace不要)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
        
    except Exception as e:
        print(f"Error analyzing {img_path}: {e}")
        
    return res

def analyze_metrics_distribution(image_paths, label_prefix=""):
    """画像リストの指標分布を集計"""
    metrics_list = defaultdict(list)
    
    count = 0
    total = len(image_paths)
    print(f"  Calculating metrics for {total} images ({label_prefix})...")
    
    for path in image_paths:
        m = get_metrics(path)
        for k, v in m.items():
            if v is not None:
                metrics_list[k].append(v)
        
        count += 1
        if count % 100 == 0:
            print(f"    {count}/{total}")
            
    # 平均値を計算 (JSON serializable に float 変換)
    summary = {}
    for k, v in metrics_list.items():
        if v:
            summary[k] = float(np.mean(v))
        else:
            summary[k] = 0.0
    return summary

def run_analysis_for_dataset(model, data_dir, output_dir, dataset_name="data"):
    """
    指定データディレクトリに対してエラー分析を実行し、結果を output_dir に保存する。
    VALIDATION_DIR と OUTPUT_DIR を一時的に更新してから実行する。
    """
    global VALIDATION_DIR, OUTPUT_DIR
    VALIDATION_DIR = data_dir
    OUTPUT_DIR = output_dir

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading data from: {data_dir} ({dataset_name})")
    image_paths, true_labels = load_validation_data(data_dir)
    print(f"Found {len(image_paths)} images.")

    if len(image_paths) == 0:
        print(f"Warning: No images in {data_dir}. Skipping.")
        return

    analyze_data_distribution(image_paths)

    print("\nRunning predictions...")
    predictions = predict_batch(model, image_paths)

    report = {}
    print("\n" + "=" * 60)
    print(f"Analysis Results ({dataset_name})")
    print("=" * 60)

    for task_idx, (task_labels, task_name) in enumerate(zip(ALL_TASK_LABELS, TASK_NAMES)):
        print(f"\n--- {task_name} ---")

        cm, accuracy = create_confusion_matrix(
            true_labels, predictions, task_idx, task_labels, task_name
        )
        print(f"Accuracy: {accuracy:.4f}")

        errors = collect_errors(
            image_paths, true_labels, predictions, task_idx, task_labels, task_name
        )
        error_summary = {k: len(v) for k, v in errors.items()}
        print(f"Errors: {sum(error_summary.values())} total")
        for error_type, count in sorted(error_summary.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")

        report[task_name] = {'accuracy': accuracy, 'errors': error_summary}

        correct = collect_correct(
            image_paths, true_labels, predictions, task_idx, task_labels, task_name
        )
        correct_summary = {k: len(v) for k, v in correct.items()}
        print(f"Correct: {sum(correct_summary.values())} total")
        report[task_name]['correct'] = correct_summary

        print(f"\n  [Preprocess Metrics Analysis]")
        error_paths = []
        for paths in errors.values():
            error_paths.extend(paths)
        correct_paths = []
        for paths in correct.values():
            correct_paths.extend(paths)

        import random
        random.seed(42)
        if len(correct_paths) > len(error_paths) and len(error_paths) > 0:
            sample_size = max(50, len(error_paths))
            correct_paths_sample = random.sample(correct_paths, sample_size) if len(correct_paths) > sample_size else correct_paths
        else:
            correct_paths_sample = correct_paths

        error_metrics = analyze_metrics_distribution(error_paths, "Errors")
        correct_metrics = analyze_metrics_distribution(correct_paths_sample, "Correct (Sampled)")

        print(f"\n    {'Metric':<15} | {'Errors (Avg)':<15} | {'Correct (Avg)':<15} | {'Diff':<10}")
        print(f"    {'-'*15}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")
        for k in ['pitch', 'symmetry', 'y_diff', 'mouth_open', 'eb_eye_dist', 'sharpness', 'face_size']:
            e_val = error_metrics.get(k, 0.0)
            c_val = correct_metrics.get(k, 0.0)
            print(f"    {k:<15} | {e_val:>15.4f} | {c_val:>15.4f} | {e_val - c_val:>+10.4f}")

        report[task_name]['metrics_comparison'] = {'errors': error_metrics, 'correct': correct_metrics}

    print("\n--- Per-Combination Accuracy (全タスク正解率) ---")
    combo_results = analyze_per_combination(image_paths, true_labels, predictions)
    report['combination_accuracy'] = combo_results
    for combo, stats in combo_results.items():
        print(f"  {combo}: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']})")

    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_DIR}/")


def main():
    global VALIDATION_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Error Analysis Script (default: train+val 前処理前)")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to the model file')
    parser.add_argument('--data_dir', type=str, default=None, help='Single run only: 指定時はこのディレクトリのみ分析（未指定時は train+val 両方）')
    parser.add_argument('--train_dir', type=str, default=None, help='Train データ（未指定時は train）')
    parser.add_argument('--val_dir', type=str, default=None, help='Val データ（未指定時は validation）')
    parser.add_argument('--out_dir', type=str, default=OUTPUT_DIR, help='Path to output directory')
    args = parser.parse_args()

    MODEL_PATH = args.model
    base_out = args.out_dir

    print("=" * 60)
    print("Error Analysis Script")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        for f in os.listdir('.'):
            if f.endswith('.keras'):
                print(f"  - {f}")
        return

    model = models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")

    # デフォルト: 前処理前の train と validation の両方で実行
    if args.data_dir is not None:
        print(f"Mode: single (--data_dir)")
        print(f"Target Data: {args.data_dir}")
        print(f"Output Dir : {base_out}")
        run_analysis_for_dataset(model, args.data_dir, base_out, "data")
        print("\n" + "=" * 60)
        print(f"Analysis complete! Results saved to: {base_out}/")
        print("  - confusion_matrix_*.png, errors/, correct/, report.json")
        print("=" * 60)
    else:
        train_dir = args.train_dir if args.train_dir is not None else DEFAULT_TRAIN_DIR
        val_dir = args.val_dir if args.val_dir is not None else DEFAULT_VAL_DIR
        print(f"Mode: train + val (前処理前: {train_dir}, {val_dir})")
        print(f"  Train: {train_dir} -> {os.path.join(base_out, 'train')}")
        print(f"  Val:   {val_dir} -> {os.path.join(base_out, 'val')}")
        os.makedirs(base_out, exist_ok=True)
        run_analysis_for_dataset(model, train_dir, os.path.join(base_out, 'train'), "train")
        run_analysis_for_dataset(model, val_dir, os.path.join(base_out, 'val'), "val")
        print("\n" + "=" * 60)
        print(f"Analysis complete! Results: {base_out}/train/ and {base_out}/val/")
        print("  - confusion_matrix_*.png, errors/, correct/, report.json in each")
        print("=" * 60)


if __name__ == "__main__":
    main()
