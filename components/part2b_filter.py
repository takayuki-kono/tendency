# Part 2b: InsightFace Filtering and Cleanup (Argument-based)
import os
import cv2
import numpy as np
import logging
import shutil
from sklearn.cluster import DBSCAN
import sys

sys.path.append('/mnt/d/tendency/.venv_new/lib/python3.12/site-packages')
import argparse
from insightface.app import FaceAnalysis

# --- Globals ---
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'log_part2b.txt'),
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

# --- Functions ---
def filter_by_main_person_insightface(input_dir, physical_delete):
    logger.info("Starting person filtering with InsightFace...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))

    # input_dir をそのまま処理対象として使用
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        logger.warning(f"Input directory not found or is empty: {input_dir}. Skipping.")
        return

    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    encodings = []
    image_path_list = []

    logger.info("Extracting face embeddings with InsightFace...")
    skipped_aspect = 0
    skipped_resolution = 0

    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        try:
            bgr_image = imread_safe(img_path)
            if bgr_image is None: continue

            img_h, img_w = bgr_image.shape[:2]
            faces = app.get(bgr_image)

            if faces:
                # Assuming one face per cropped image, or take largest
                face = faces[0]
                x1, y1, x2, y2 = face.bbox

                # 縮尺チェック1: アスペクト比（潰れすぎ/伸びすぎ）
                face_width = x2 - x1
                face_height = y2 - y1
                aspect_ratio = face_height / face_width if face_width > 0 else 0

                if aspect_ratio < 0.9 or aspect_ratio > 1.8:
                    logger.info(f"Skipped (abnormal aspect ratio {aspect_ratio:.3f}): {img_path}")
                    skipped_aspect += 1
                    continue

                # 全チェック通過
                embedding = face.embedding
                encodings.append(embedding)
                image_path_list.append(img_path)
            else:
                logger.warning(f"InsightFace did not find a face in: {img_path}")
        except Exception as e:
            logger.error(f"Error during InsightFace processing for {img_path}: {e}", exc_info=True)

    logger.info(f"Filtering summary: skipped {skipped_aspect} (aspect), {skipped_resolution} (resolution)")

    if len(encodings) < 2: 
        logger.info("Fewer than 2 images to cluster. Skipping filtering.")
        return

    # Using the tuned parameters
    METRIC = 'cosine'
    TOLERANCE = 0.5 
    MIN_SAMPLES = 2

    logger.info(f"Clustering with DBSCAN, metric={METRIC}, eps={TOLERANCE}")
    clustering = DBSCAN(metric=METRIC, eps=TOLERANCE, min_samples=MIN_SAMPLES).fit(np.array(encodings))
    labels = clustering.labels_

    if len(labels) == 0: return

    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0: 
        logger.warning("No main cluster found, only noise.")
        return
    
    main_cluster_label = unique_labels[np.argmax(counts)]
    logger.info(f"Main cluster label: {main_cluster_label} with {counts.max()} images.")

    images_to_keep = []
    for i, label in enumerate(labels):
        if label == main_cluster_label:
            images_to_keep.append(image_path_list[i])

    # deleted ディレクトリは親ディレクトリに作成（論理削除の場合のみ）
    parent_dir = os.path.dirname(input_dir)
    deleted_dir = os.path.join(parent_dir, "deleted_outliers")
    if not physical_delete and not os.path.exists(deleted_dir):
        os.makedirs(deleted_dir)

    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for path in all_images:
        if path not in images_to_keep:
            try:
                if physical_delete:
                    os.remove(path)
                    logger.info(f"Deleted non-main person file: {path}")
                else:
                    dst = os.path.join(deleted_dir, os.path.basename(path))
                    if os.path.exists(dst): os.remove(dst)
                    shutil.move(path, dst)
                    logger.info(f"Moved non-main person file: {path}")
            except Exception as e:
                logger.error(f"Error removing {path}: {e}")

def cleanup_directories(input_dir):
    """
    フィルタリング後の整理:
    - input_dir (rotated) 内の残り画像を親ディレクトリに移動
    - 空になった rotated フォルダを削除
    """
    logger.info("Starting final cleanup.")
    parent_dir = os.path.dirname(input_dir)
    
    # rotated 内の画像を親ディレクトリに移動
    for item_name in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item_name)
        if os.path.isfile(item_path):
            try:
                dst = os.path.join(parent_dir, item_name)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(item_path, dst)
                logger.info(f"Moved to parent: {item_name}")
            except Exception as e:
                logger.error(f"Error moving {item_path}: {e}")
    
    # 空になった rotated フォルダを削除
    try:
        if os.path.exists(input_dir) and not os.listdir(input_dir):
            shutil.rmtree(input_dir)
            logger.info(f"Removed empty directory: {input_dir}")
    except Exception as e:
        logger.error(f"Error removing directory {input_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Part 2b: Filtering")
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("--physical_delete", action="store_true", help="Enable physical deletion")
    args = parser.parse_args()

    input_dir = args.input_dir
    logger.info(f"Part 2b starting for directory: {input_dir}, physical_delete: {args.physical_delete}")
    try:
        filter_by_main_person_insightface(input_dir, args.physical_delete)
        cleanup_directories(input_dir)
        logger.info(f"Part 2b finished for directory: {input_dir}")
    except Exception as e:
        logger.error(f"Fatal error in Part 2b: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()