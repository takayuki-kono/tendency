import os
import sys
import shutil
import cv2
import numpy as np
import logging
import argparse
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis

# --- Config ---
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "log_standalone_filter.txt"), mode='w', encoding='utf-8')
    ]
)
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

def filter_folder(input_dir, eps, physical_delete=False):
    logger.info(f"Starting standalone filter on: {input_dir}")
    logger.info(f"Parameters: eps={eps}, physical_delete={physical_delete}")

    if not os.path.exists(input_dir):
        logger.error(f"Directory not found: {input_dir}")
        return

    # Init InsightFace
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        logger.error(f"Failed to init InsightFace: {e}")
        return

    # List files (exclude directories)
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"Found {len(files)} image files.")

    if len(files) < 2:
        logger.info("Not enough images to cluster.")
        return

    embeddings = []
    valid_paths = []

    logger.info("Extracting embeddings...")
    for idx, fname in enumerate(files):
        path = os.path.join(input_dir, fname)
        img = imread_safe(path)
        if img is None: continue
        
        try:
            faces = app.get(img)
            if faces:
                # Largest face
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                embeddings.append(face.embedding)
                valid_paths.append(path)
        except:
            pass
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(files)}")

    if not embeddings:
        logger.warning("No embeddings detected.")
        return

    # Clustering
    logger.info(f"Clustering with DBSCAN(eps={eps}, min_samples=2)...")
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=2).fit(embeddings)
    labels = clustering.labels_

    # Main Cluster
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        logger.warning("No main cluster found. All images considered noise.")
        main_cluster_label = -999 # None
        # In this case, ALL are outliers.
    else:
        main_cluster_label = unique_labels[np.argmax(counts)]
        logger.info(f"Main cluster label: {main_cluster_label} (Count: {counts.max()})")

    # Move/Delete Outliers
    deleted_dir = os.path.join(input_dir, "deleted_outliers")
    if not physical_delete and not os.path.exists(deleted_dir):
        os.makedirs(deleted_dir)

    outlier_count = 0
    for i, label in enumerate(labels):
        path = valid_paths[i]
        if label != main_cluster_label:
            # Outlier
            outlier_count += 1
            try:
                if physical_delete:
                    os.remove(path)
                    logger.info(f"Deleted: {os.path.basename(path)}")
                else:
                    dst = os.path.join(deleted_dir, os.path.basename(path))
                    if os.path.exists(dst): os.remove(dst)
                    shutil.move(path, dst)
                    logger.info(f"Moved to outliers: {os.path.basename(path)}")
            except Exception as e:
                logger.error(f"Error handling {path}: {e}")

    logger.info(f"Filter complete. Removed/Moved {outlier_count} images out of {len(valid_paths)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Target directory")
    parser.add_argument("--eps", type=float, default=0.45, help="DBSCAN tolerance (lower is stricter). Default 0.45")
    parser.add_argument("--physical_delete", action="store_true", help="Permanently delete files")
    
    args = parser.parse_args()
    
    filter_folder(args.input_dir, args.eps, args.physical_delete)
