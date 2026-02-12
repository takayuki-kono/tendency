import os
import cv2
import numpy as np
import argparse
from insightface.app import FaceAnalysis
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import sys

# Configure Logging to match standard
logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize InsightFace
# Using standard providers. GPU recommended but CPU works for inference.
# Suppress insightface logging slightly
# logger.setLevel(logging.ERROR)
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def read_image(path):
    try:
        # Handle non-ASCII paths on Windows
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

import hashlib
import pickle

CACHE_EMBED_DIR = os.path.join("outputs", "cache", "embeddings")
os.makedirs(CACHE_EMBED_DIR, exist_ok=True)

def get_file_hash(filepath):
    """Calculate MD5 hash of file content for caching key"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def get_embedding(img, filepath):
    """Get embedding from cache or run inference"""
    # Use file hash as key to handle same filename but different content situations (though unlikely here)
    # Also ensures that if file changes, we recompute.
    file_hash = get_file_hash(filepath)
    cache_path = os.path.join(CACHE_EMBED_DIR, f"{file_hash}.npy")
    
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except:
            pass # Load failed, recompute

    # Inference
    faces = app.get(img)
    if len(faces) > 0:
        # Use largest face
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        if face.embedding is not None:
            np.save(cache_path, face.embedding)
            return face.embedding
            
    return None

def extract_embeddings(data_dir):
    """
    Extract embeddings and labels for ALL tasks encoded in folder names.
    Returns:
        embeddings: np.array of shape (N, D)
        task_labels_list: list of N-sized arrays, one for each task's labels
    """
    embeddings = []
    # List of lists to store labels for each task. 
    # Will initialize when we see the first valid folder.
    task_labels_list = [] 
    
    if not os.path.exists(data_dir):
        return np.array([]), []
    
    if not os.listdir(data_dir):
        return np.array([]), []

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    num_tasks = 0
    
    for folder_name in classes:
        if len(folder_name) < 1: continue
        
        # Initialize label lists on first valid folder
        if num_tasks == 0:
            num_tasks = len(folder_name)
            task_labels_list = [[] for _ in range(num_tasks)]
        
        # Check consistency (skip if length differs)
        if len(folder_name) != num_tasks:
            continue
            
        current_labels = list(folder_name) # ['a', 'b', 'c', ...]
        
        class_dir = os.path.join(data_dir, folder_name)
        
        # Collect images in this class folder
        # Optimization: Scan dir once, filtering for images
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for fname in image_files:
            fpath = os.path.join(class_dir, fname)
            
            # Read image is still needed to check validity
            img = read_image(fpath)
            if img is None:
                continue
            
            emb = get_embedding(img, fpath)
            
            if emb is not None:
                embeddings.append(emb)
                # Append labels for EACH task
                for i in range(num_tasks):
                    task_labels_list[i].append(current_labels[i])
    
    if len(embeddings) == 0:
        return np.array([]), []

    print(f"Extracted {len(embeddings)} faces from {data_dir}. Detected {num_tasks} tasks.")
    
    # Convert lists to arrays
    task_labels_arrays = [np.array(labels) for labels in task_labels_list]
    
    return np.array(embeddings), task_labels_arrays

def min_class_accuracy_score(y_true, y_pred):
    """Calculate the minimum accuracy across all classes (MinClassAccuracy)"""
    classes = np.unique(y_true)
    if len(classes) == 0:
        return 0.0
        
    accuracies = []
    for cls in classes:
        mask = (y_true == cls)
        if np.sum(mask) == 0:
            continue
            
        cls_acc = np.mean(y_pred[mask] == y_true[mask])
        accuracies.append(cls_acc)
        
    if not accuracies:
        return 0.0
        
    return np.min(accuracies)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="preprocessed_multitask", help="Root dir of dataset")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter")
    parser.add_argument("--kernel", type=str, default='rbf', help="SVM kernel (linear, poly, rbf, sigmoid)")
    parser.add_argument("--gamma", type=str, default='scale', help="SVM gamma (scale, auto) or float")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "validation")

    # Extract
    print(f"Extracting train from {train_dir}...")
    X_train, y_train_list = extract_embeddings(train_dir)
    print(f"Extracting val from {val_dir}...")
    X_val, y_val_list = extract_embeddings(val_dir)

    if len(X_train) == 0 or len(X_val) == 0:
        print("Error: No data found.")
        print("FINAL_SCORE: 0.0")
        return

    # Check task consistency
    if len(y_train_list) != len(y_val_list):
        print(f"Error: Mismatch in number of tasks (Train: {len(y_train_list)}, Val: {len(y_val_list)})")
        print("FINAL_SCORE: 0.0")
        return
        
    num_tasks = len(y_train_list)
    print(f"Training SVMs for {num_tasks} tasks...")
    
    # Scaling (Common for all tasks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    scores = []
    
    # Handle gamma
    gamma_val = args.gamma
    try:
        gamma_val = float(gamma_val)
    except ValueError:
        pass 

    tasks = [chr(ord('A') + i) for i in range(num_tasks)]
    task_scores = {}

    print("-" * 50)
    for i, task_name in enumerate(tasks):
        y_train_raw = y_train_list[i]
        y_val_raw = y_val_list[i]
        
        # Encoding
        le = LabelEncoder()
        # Fit on combined labels to ensure consistency
        all_labels = np.concatenate([y_train_raw, y_val_raw])
        le.fit(all_labels)
        
        y_train = le.transform(y_train_raw)
        y_val = le.transform(y_val_raw)
        
        # Show distribution
        train_counts = np.bincount(y_train)
        val_counts = np.bincount(y_val)
        class_names = le.classes_
        
        if len(class_names) < 2:
            print(f"Task {task_name}: Skipped (Only 1 class found: {class_names})")
            continue
            
        print(f"Task {task_name} (Classes: {len(class_names)})")
        for cls_idx, cls_name in enumerate(class_names):
            t_cnt = train_counts[cls_idx] if cls_idx < len(train_counts) else 0
            v_cnt = val_counts[cls_idx] if cls_idx < len(val_counts) else 0
            print(f"  - Class '{cls_name}': Train={t_cnt}, Val={v_cnt}")

        if len(np.unique(y_train)) < 2:
            print(f"  -> Skipped: Training data has only 1 class.")
            continue
            
        if len(np.unique(y_val)) < 2:
            print(f"  -> Warning: Validation data has only 1 class. Score might be biased.")

        print(f"  Training SVM...")
        clf = SVC(kernel=args.kernel, C=args.C, gamma=gamma_val, class_weight='balanced', probability=True)
        
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            
            # Predict only gives hard labels, but let's check unique predictions
            unique_pred = np.unique(y_pred)
            if len(unique_pred) == 1:
                print(f"  -> Warning: SVM predicted only class {le.inverse_transform(unique_pred)[0]} for all validation samples.")
            
            # Use MinClassAccuracy instead of Balanced Accuracy
            task_score = min_class_accuracy_score(y_val, y_pred)
            scores.append(task_score)
            task_scores[task_name] = task_score
            print(f"  -> Task {task_name} Score (MinClassAcc): {task_score:.5f}")
            
        except Exception as e:
            print(f"  -> Error training Task {task_name}: {e}")
            scores.append(0.0)
            task_scores[task_name] = 0.0
        print("-" * 50)

    if scores:
        avg_score = sum(scores) / len(scores)
        print("=" * 50)
        print("TASK SCORES SUMMARY (MinClassAccuracy):")
        for t, s in task_scores.items():
            print(f"  Task {t}: {s:.5f}")
        print("=" * 50)
        print(f"FINAL_SCORE: {avg_score:.5f}")
    else:
        print("FINAL_SCORE: 0.0")

if __name__ == "__main__":
    main()
