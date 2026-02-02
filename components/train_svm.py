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
    embeddings = []
    labels = []
    
    if not os.path.exists(data_dir):
        return np.array([]), np.array([])
    
    if not os.listdir(data_dir):
        return np.array([]), np.array([])

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for folder_name in classes:
        # According to train_for_filter_search.py, folder name chars map to tasks.
        # Task A is the 1st character.
        if len(folder_name) < 1: continue
        task_a_label = folder_name[0]
        
        class_dir = os.path.join(data_dir, folder_name)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            fpath = os.path.join(class_dir, fname)
            
            # Read image is still needed to check validity, but maybe we can skip if cache exists?
            # We need image for inference if cache miss.
            # Optimization: check cache FIRST by file hash (requires reading file), then read image ONLY if needed.
            
            # However, logic below reads image first. Let's keep it simple for now.
            img = read_image(fpath)
            if img is None:
                continue
            
            emb = get_embedding(img, fpath)
            
            if emb is not None:
                embeddings.append(emb)
                labels.append(task_a_label)
    
    print(f"Extracted {len(embeddings)} faces from {data_dir}. Found {len(np.unique(labels))} unique Task A classes.")
    return np.array(embeddings), np.array(labels)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="preprocessed_multitask", help="Root dir of dataset")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "validation")

    # Extract
    print(f"Extracting train from {train_dir}...")
    X_train, y_train_txt = extract_embeddings(train_dir)
    print(f"Extracting val from {val_dir}...")
    X_val, y_val_txt = extract_embeddings(val_dir)

    # Encoding
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_txt, y_val_txt])) # Fit on all potential labels
    
    y_train = le.transform(y_train_txt)
    y_val = le.transform(y_val_txt)
    
    unique_train = np.unique(y_train)
    if len(unique_train) < 2:
         print("Error: Need at least 2 classes in training data.")
         print("FINAL_SCORE: 0.0")
         return

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True)
    
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        score = balanced_accuracy_score(y_val, y_pred)
        print(f"FINAL_SCORE: {score:.5f}")

    except Exception as e:
        print(f"Error during training: {e}")
        print("FINAL_SCORE: 0.0")

if __name__ == "__main__":
    main()
