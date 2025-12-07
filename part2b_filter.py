# Part 2b: InsightFace Filtering and Cleanup (Argument-based)
import os
import cv2
import numpy as np
import logging
import shutil
from sklearn.cluster import DBSCAN
import sys

sys.path.append('/mnt/d/tendency/.venv_new/lib/python3.12/site-packages')
from insightface.app import FaceAnalysis

# --- Logging Setup ---
logging.basicConfig(
    filename='log_part2b.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# --- Functions ---
def filter_by_main_person_insightface(input_dir):
    logger.info("Starting person filtering with InsightFace...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))

    rotated_dir = os.path.join(input_dir, "rotated")
    if not os.path.exists(rotated_dir) or not os.listdir(rotated_dir):
        logger.warning(f"'rotated' directory not found or is empty. Skipping.")
        return

    image_files = [f for f in os.listdir(rotated_dir) if os.path.isfile(os.path.join(rotated_dir, f))]
    encodings = []
    image_path_list = []

    logger.info("Extracting face embeddings with InsightFace...")
    skipped_aspect = 0
    skipped_resolution = 0

    for filename in image_files:
        img_path = os.path.join(rotated_dir, filename)
        try:
            bgr_image = cv2.imread(img_path)
            if bgr_image is None: continue

            img_h, img_w = bgr_image.shape[:2]
            faces = app.get(bgr_image)

            if faces:
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

    deleted_dir = os.path.join(input_dir, "deleted")
    if not os.path.exists(deleted_dir):
        os.makedirs(deleted_dir)

    all_rotated_images = [os.path.join(rotated_dir, f) for f in os.listdir(rotated_dir)]
    for path in all_rotated_images:
        if path not in images_to_keep:
            shutil.move(path, os.path.join(deleted_dir, os.path.basename(path)))
            logger.info(f"Moved non-main person file: {path}")

def cleanup_directories(input_dir):
    logger.info("Starting final cleanup.")
    dirs_to_delete = ["processed", "bbox_cropped", "bbox_rotated"]
    for dir_name in dirs_to_delete:
        dir_path = os.path.join(input_dir, dir_name)
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        except Exception as e:
            logger.error(f"Error deleting directory {dir_path}: {e}")
    
    logger.info(f"Cleaning up leftover files in {input_dir}")
    for item_name in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item_name)
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
            except Exception as e:
                logger.error(f"Error deleting file {item_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python part2b_filter.py <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    logger.info(f"Part 2b starting for directory: {input_dir}")
    try:
        filter_by_main_person_insightface(input_dir)
        cleanup_directories(input_dir)
        logger.info(f"Part 2b finished for directory: {input_dir}")
    except Exception as e:
        logger.error(f"Fatal error in Part 2b: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()