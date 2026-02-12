import os
import argparse
import sys
import shutil
import numpy as np
import cv2
from glob import glob
from insightface.app import FaceAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging
import io

# 文字化け対策 (Windowsコンソール向け)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

def train_and_predict(input_dir, output_dir, train_dir, svm_params):
    # 1. 学習データの読み込みとEmbedding抽出
    logger.info("Extracting embeddings from training data...")
    X_train = []
    y_train = []
    
    if not os.path.exists(train_dir):
        logger.error(f"Training directory not found: {train_dir}")
        return

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
        return

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # 2. 前処理（Encoding & Scaling）
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. SVMの学習
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

    # 4. 予測対象データの処理
    logger.info(f"Processing images in {input_dir}...")
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    logger.info(f"Found {len(image_paths)} images to classify.")
    
    count_dict = {label: 0 for label in le.classes_}
    count_dict['unknown'] = 0
    count_dict['no_face'] = 0

    for path in image_paths:
        emb = get_embedding(path)
        if emb is None:
            logger.warning(f"No face detected in {path}")
            count_dict['no_face'] += 1
            # 顔が見つからない場合の処理（オプションで移動など）
            dest_dir = os.path.join(output_dir, "no_face")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))
            continue

        # 前処理と予測
        emb_scaled = scaler.transform([emb])
        probs = clf.predict_proba(emb_scaled)[0]
        pred_idx = np.argmax(probs)
        pred_label = le.classes_[pred_idx]
        confidence = probs[pred_idx]
        
        # 結果の保存（フォルダ移動/コピー）
        dest_dir = os.path.join(output_dir, pred_label)
        os.makedirs(dest_dir, exist_ok=True)
        try:
            shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))
            count_dict[pred_label] += 1
            logger.info(f"{os.path.basename(path)} -> {pred_label} ({confidence:.2%})")
        except Exception as e:
            logger.error(f"Failed to copy {path}: {e}")

    # 5. 結果サマリ
    logger.info("\n=== Classification Summary ===")
    total_valid = sum(count_dict.values()) - count_dict['no_face']
    
    # 全画像の確率ベクトルの平均を計算
    all_probs = []

    for path in image_paths:
        emb = get_embedding(path)
        if emb is None: continue

        emb_scaled = scaler.transform([emb])
        probs = clf.predict_proba(emb_scaled)[0]
        all_probs.append(probs)

    if all_probs:
        avg_probs = np.mean(all_probs, axis=0)
        logger.info("\n=== Overall Classification Tendency (Average Probabilities) ===")
        for i, label_name in enumerate(le.classes_):
            logger.info(f"{label_name}: {avg_probs[i]:.2%}")
        
        # 最も高い平均確率を持つクラスを特定
        top_idx = np.argmax(avg_probs)
        logger.info(f"\nDominant Class: {le.classes_[top_idx]} ({avg_probs[top_idx]:.2%})")

    for label, count in count_dict.items():
        ratio = (count / total_valid * 100) if total_valid > 0 and label != 'no_face' else 0
        logger.info(f"{label}: {count} ({ratio:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Classify images using SVM trained on the fly (or loaded).")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images to classify')
    parser.add_argument('--output_dir', type=str, default='outputs/svm_predictions', help='Directory to output classified images')
    parser.add_argument('--train_dir', type=str, default='preprocessed_multitask_svm/train', help='Directory containing training data (to train SVM)')
    
    # SVM Params (Optional override)
    parser.add_argument('--C', type=float, help='SVM C parameter')
    parser.add_argument('--kernel', type=str, help='SVM kernel')
    parser.add_argument('--gamma', type=str, help='SVM gamma')

    args = parser.parse_args()

    # Load best params from file (optimized by optimize_svm_sequential.py)
    import json
    best_params_file = "outputs/best_svm_train_params.json"
    svm_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
    
    if os.path.exists(best_params_file):
        try:
            with open(best_params_file, 'r') as f:
                loaded_params = json.load(f)
                svm_params.update(loaded_params)
            logger.info(f"Loaded best SVM params: {svm_params}")
        except:
            logger.warning("Failed to load best SVM params, using defaults.")

    # Override with CLI args if provided
    if args.C is not None: svm_params['C'] = args.C
    if args.kernel is not None: svm_params['kernel'] = args.kernel
    if args.gamma is not None: svm_params['gamma'] = args.gamma # Handle float conversion if needed inside logic

    # Handle numeric gamma from file or CLI
    if 'gamma' in svm_params:
        try:
             svm_params['gamma'] = float(svm_params['gamma'])
        except ValueError:
             pass # Keep as string ('scale', 'auto')

    train_and_predict(args.input_dir, args.output_dir, args.train_dir, svm_params)

if __name__ == "__main__":
    main()
