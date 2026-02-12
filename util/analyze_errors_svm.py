import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import cv2
import pickle
import json
import logging
import io
import sys
from insightface.app import FaceAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# コンソール文字化け対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- 設定 ---
DEFAULT_VALIDATION_DIR = 'preprocessed_multitask_svm/validation'
DEFAULT_TRAIN_DIR = 'preprocessed_multitask_svm/train'
DEFAULT_OUTPUT_DIR = 'svm_error_analysis'

# InsightFaceの初期化
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def get_embedding(img_path):
    try:
        # 日本語パス対応
        with open(img_path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

        if img is None: return None
        faces = app.get(img)
        if len(faces) == 0: return None
        # 最大の顔を選択
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        return face.embedding
    except Exception as e:
        logger.error(f"Error extracting embedding for {img_path}: {e}")
        return None

def train_svm(train_dir, svm_params):
    logger.info("Extracting embeddings from training data...")
    X_train = []
    y_train = []
    
    if not os.path.exists(train_dir):
        logger.error(f"Training directory not found: {train_dir}")
        return None, None, None

    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    for folder_name in classes:
        # フォルダ名の1文字目をラベルとする (a, bなど)
        if len(folder_name) < 1: continue
        label = folder_name[0]
        
        class_dir = os.path.join(train_dir, folder_name)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')): continue
            path = os.path.join(class_dir, fname)
            emb = get_embedding(path)
            if emb is not None:
                X_train.append(emb)
                y_train.append(label)

    if len(X_train) == 0:
        logger.error("No training data found.")
        return None, None, None

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Preprocessing
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # SVM Training
    logger.info(f"Training SVM with params: {svm_params}")
    clf = SVC(
        kernel=svm_params.get('kernel', 'rbf'),
        C=svm_params.get('C', 1.0),
        gamma=svm_params.get('gamma', 'scale'),
        class_weight='balanced',
        probability=True
    )
    clf.fit(X_train_scaled, y_train_enc)
    logger.info("Training complete.")
    
    return clf, scaler, le

def load_validation_data(val_dir):
    logger.info(f"Loading validation data from {val_dir}...")
    X_val_paths = []
    y_val_true = []
    
    if not os.path.exists(val_dir):
        logger.error(f"Validation directory not found: {val_dir}")
        return [], []

    classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    
    for folder_name in classes:
        if len(folder_name) < 1: continue
        label = folder_name[0] # 先頭1文字をクラスラベルとする
        
        class_dir = os.path.join(val_dir, folder_name)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')): continue
            path = os.path.join(class_dir, fname)
            X_val_paths.append(path)
            y_val_true.append(label)
    
    logger.info(f"Found {len(X_val_paths)} validation images.")
    return X_val_paths, y_val_true

def analyze_errors(clf, scaler, le, X_val_paths, y_val_true, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred_labels = []
    
    logger.info("Predicting validation set...")
    valid_indices = [] # 顔が検出できた画像のインデックスと結果格納用
    pred_results = [] # (true, pred, path)
    errors = defaultdict(list)
    
    cm_labels = le.classes_
    label_to_idx = {l: i for i, l in enumerate(cm_labels)}
    cm = np.zeros((len(cm_labels), len(cm_labels)), dtype=float)

    face_detected_count = 0
    no_face_count = 0

    for i, (path, true_label) in enumerate(zip(X_val_paths, y_val_true)):
        emb = get_embedding(path)
        if emb is None:
            # 顔検出失敗
            no_face_count += 1
            errors["no_face"].append(path)
            continue
            
        face_detected_count += 1
        emb_scaled = scaler.transform([emb])
        probs = clf.predict_proba(emb_scaled)[0]
        pred_idx = np.argmax(probs)
        pred_label = le.classes_[pred_idx]
        
        if true_label not in label_to_idx:
            continue

        true_idx = label_to_idx[true_label]
        # pred_idx is already index
        
        cm[true_idx][pred_idx] += 1
        
        if true_label != pred_label:
            key = f"True_{true_label}_Pred_{pred_label}"
            errors[key].append(path)
            # エラー画像の保存
            err_dir = os.path.join(output_dir, "errors", key)
            os.makedirs(err_dir, exist_ok=True)
            try:
                shutil.copy2(path, os.path.join(err_dir, os.path.basename(path)))
            except Exception as e:
                logger.error(f"Failed to copy error image {path}: {e}")
        else:
             # 正解画像の保存 (オプション)
             corr_dir = os.path.join(output_dir, "correct", true_label)
             os.makedirs(corr_dir, exist_ok=True)
             try:
                shutil.copy2(path, os.path.join(corr_dir, os.path.basename(path)))
             except: pass

    # Metrics
    total_samples = np.sum(cm)
    correct_predictions = np.trace(cm)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    logger.info(f"Validation Accuracy (on face-detected images): {accuracy:.4f}")
    logger.info(f"Total Images: {len(X_val_paths)}, Face Detected: {face_detected_count}, No Face: {no_face_count}")
    
    # Save Confusion Matrix
    plt.figure(figsize=(10, 8))
    
    # 混同行列（件数ベース）
    cm_counts = cm.astype(int)
    
    # 混同行列（正規化 %）
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    # 0除算回避
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = cm / row_sums * 100
    cm_percent = np.nan_to_num(cm_percent)
    
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title(f'SVM Confusion Matrix (Acc={accuracy:.2%})')
    plt.colorbar()
    tick_marks = np.arange(len(cm_labels))
    plt.xticks(tick_marks, cm_labels, rotation=45)
    plt.yticks(tick_marks, cm_labels)
    
    thresh = 50.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_percent[i, j]
            count = cm_counts[i, j]
            txt = f"{val:.1f}%\n({count})"
            plt.text(j, i, txt,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if val > thresh else "black")
                     
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svm_confusion_matrix.png'))
    plt.close()
    
    # Save Report
    report = {
        'accuracy': accuracy,
        'total_images': len(X_val_paths),
        'face_detected': face_detected_count,
        'no_face': no_face_count,
        'correct_predictions': int(correct_predictions),
        'errors_summary': {k: len(v) for k, v in errors.items()}
    }
    with open(os.path.join(output_dir, 'svm_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Analysis complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze SVM errors on validation set.")
    parser.add_argument('--train_dir', type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument('--val_dir', type=str, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    
    # Load Best Params
    best_params_file = "outputs/best_svm_train_params.json"
    svm_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
    
    if os.path.exists(best_params_file):
        try:
            with open(best_params_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                svm_params.update(loaded)
            logger.info(f"Loaded best params: {svm_params}")
        except:
            logger.warning("Using default params.")
            
    clf, scaler, le = train_svm(args.train_dir, svm_params)
    if clf is None: return
    
    if not os.path.exists(args.val_dir):
        logger.error(f"Validation directory not found: {args.val_dir}")
        return

    X_val_paths, y_val_true = load_validation_data(args.val_dir)
    if not X_val_paths:
        logger.error("No validation data found.")
        return
        
    analyze_errors(clf, scaler, le, X_val_paths, y_val_true, args.out_dir)

if __name__ == "__main__":
    main()
