import os
import sys
import shutil
import cv2
import numpy as np
import logging
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
import random

# --- Config ---
TARGET_DIR = r"D:\tendency\master_data\奈緒"
OUTPUT_ROOT = r"D:\tendency\test_filter_viz"
THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55] # Test range
MIN_SAMPLES = 2 # Keeping consistent with part2b default, or try 3?

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def imread_safe(path):
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        return img
    except:
        return None

def get_embeddings(input_dir):
    logger.info("Initializing InsightFace...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"Scanning {len(files)} files in {input_dir}...")
    
    embeddings = []
    valid_paths = []
    
    for idx, fname in enumerate(files):
        path = os.path.join(input_dir, fname)
        img = imread_safe(path)
        if img is None: continue
        
        try:
            faces = app.get(img)
            if faces:
                # Assume largest face is target
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                embeddings.append(face.embedding)
                valid_paths.append(path)
        except Exception as e:
            pass
            
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(files)}...")

    return np.array(embeddings), valid_paths

def run_test():
    if not os.path.exists(TARGET_DIR):
        logger.error(f"Target directory not found: {TARGET_DIR}")
        return

    logger.info("--- Phase 1: Embedding Extraction ---")
    embeddings, image_paths = get_embeddings(TARGET_DIR)
    
    if len(embeddings) < 2:
        logger.error("Not enough faces found.")
        return
        
    logger.info(f"Extracted {len(embeddings)} valid face embeddings.")

    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    logger.info("\n--- Phase 2: DBSCAN Tuning ---")
    
    for eps in THRESHOLDS:
        logger.info(f"\nTesting EPS (Tolerance) = {eps} ...")
        
        clustering = DBSCAN(metric='cosine', eps=eps, min_samples=MIN_SAMPLES).fit(embeddings)
        labels = clustering.labels_
        
        # Analyze clusters
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        
        if len(unique_labels) == 0:
            logger.warning(f"  [EPS {eps}] No clusters found (All noise). All {len(labels)} images removed.")
            continue
            
        main_cluster_label = unique_labels[np.argmax(counts)]
        param_dir = os.path.join(OUTPUT_ROOT, f"eps_{eps:.2f}")
        dir_kept = os.path.join(param_dir, "kept_sample")
        dir_removed = os.path.join(param_dir, "removed_all")
        os.makedirs(dir_kept, exist_ok=True)
        os.makedirs(dir_removed, exist_ok=True)
        
        kept_paths = []
        removed_paths = []
        
        for i, label in enumerate(labels):
            if label == main_cluster_label:
                kept_paths.append(image_paths[i])
            else:
                removed_paths.append(image_paths[i])
                
        # Stats
        count_kept = len(kept_paths)
        count_removed = len(removed_paths)
        total = count_kept + count_removed
        removal_rate = (count_removed / total) * 100
        
        logger.info(f"  [EPS {eps}] Kept: {count_kept}, Removed: {count_removed} ({removal_rate:.1f}%)")
        
        # Save visualizations
        # Copy ALL removed (to check what we are losing/cleaning)
        for p in removed_paths:
            shutil.copy(p, os.path.join(dir_removed, os.path.basename(p)))
            
        # Copy SAMPLE of kept (to ensure purity) - max 20
        sample_kept = random.sample(kept_paths, min(len(kept_paths), 20))
        for p in sample_kept:
            shutil.copy(p, os.path.join(dir_kept, os.path.basename(p)))
            
    logger.info(f"\nTest finished. Check results in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    run_test()
