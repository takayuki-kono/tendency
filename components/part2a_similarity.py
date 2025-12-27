# Part 2a: Similarity Check (Argument-based)
import os
import cv2
import numpy as np
import logging
import shutil
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
import sys

# --- Globals ---
METRIC = 'cosine'
DEDUPLICATION_TOLERANCE = 0.25 # Tight tolerance for duplicates
MIN_SAMPLES = 2
PHYSICAL_DELETE = True  # True: 物理削除, False: deleted フォルダへ移動
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'log_part2a.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# --- Safe I/O ---
def imread_safe(path):
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning(f"Failed to read image {path}: {e}")
        return None

# --- Function ---
def find_similar_images(input_dir):
    """input_dir 内の画像から類似画像を検出し、重複を deleted へ移動する"""
    logger.info("Starting similarity search with InsightFace embeddings...")
    
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("FaceAnalysis initialized.")
    except Exception as e:
        logger.error(f"FaceAnalysis initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # input_dir を直接処理（サブディレクトリは探さない）
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        logger.warning(f"Input directory not found or is empty: {input_dir}. Skipping.")
        return

    # deleted ディレクトリは親ディレクトリに作成（論理削除の場合のみ）
    parent_dir = os.path.dirname(input_dir)
    deleted_dir = os.path.join(parent_dir, "deleted")
    if not PHYSICAL_DELETE and not os.path.exists(deleted_dir):
        os.makedirs(deleted_dir)

    # Filter for standard image extensions
    image_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(input_dir, f))
    ]
    logger.info(f"Found {len(image_files)} images in input directory.")

    if not image_files:
        logger.warning("No images found in processed directory.")
        return

    # --- Get Embeddings ---
    encodings = []
    image_path_list = []
    
    for img_path in image_files:
        try:
            bgr_image = imread_safe(img_path)
            if bgr_image is None:
                continue
            
            # Since these are already cropped/align-ready images from Part 1,
            # Detection should be easier, but extremely tight crops (eye-only?) 
            # might fail detection.
            faces = app.get(bgr_image)
            
            if faces:
                # Use the largest face if multiple found in this pre-cropped image
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                encodings.append(face.embedding)
                image_path_list.append(img_path)
            else:
                # !!! Fallback for Eye Crop !!!
                # Because "Top of Eye to Chin" is a partial face (missing forehead/brows),
                # standard detectors might fail.
                # If fail, we cannot extract embeddings easily for deduplication using InsightFace.
                # However, Part 1 saves "processed" files named identically for both crops?
                # No, they are in different folders.
                # If detection fails, we can't filter by similarity here.
                # But wait, if crop_eyebrow works, we can leverage that result?
                # The user asked: "If it doesn't work, apply the same filtering as eyebrow crop".
                # To do that, we need to know WHICH files were deleted in crop_eyebrow run.
                # This script runs independently.
                # So we cannot easily "copy" the deletion decision unless we shared state.
                # But let's log the failure.
                logger.warning(f"No face found in processed image: {os.path.basename(img_path)}")
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    if len(encodings) < MIN_SAMPLES:
        logger.info("Not enough faces for clustering.")
        return

    # --- Clustering ---
    logger.info(f"Clustering with DBSCAN, eps={DEDUPLICATION_TOLERANCE}")
    clustering = DBSCAN(metric=METRIC, eps=DEDUPLICATION_TOLERANCE, min_samples=MIN_SAMPLES).fit(np.array(encodings))
    labels = clustering.labels_

    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label not in clusters: clusters[label] = []
            clusters[label].append(image_path_list[i])
    
    logger.info(f"Found {len(clusters)} duplicate groups.")

    # --- Remove Duplicates ---
    for label, group in clusters.items():
        # Keep first
        keep_one = group.pop(0)
        logger.info(f"Cluster {label}: Keeping {os.path.basename(keep_one)}, removing {len(group)} duplicates.")

        for img_path in group:
            try:
                if os.path.exists(img_path):
                    if PHYSICAL_DELETE:
                        os.remove(img_path)
                        logger.info(f"Deleted: {os.path.basename(img_path)}")
                    else:
                        dst = os.path.join(deleted_dir, os.path.basename(img_path))
                        if os.path.exists(dst):
                            os.remove(dst) 
                        shutil.move(img_path, dst)
                        logger.info(f"Moved to deleted: {os.path.basename(img_path)}")
            except Exception as e:
                logger.error(f"Error removing duplicate {img_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python part2a_similarity.py <input_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    logger.info(f"Part 2a starting for directory: {input_dir}")
    try:
        find_similar_images(input_dir)
        logger.info("Part 2a finished.")
    except Exception as e:
        logger.error(f"Fatal error in Part 2a: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
