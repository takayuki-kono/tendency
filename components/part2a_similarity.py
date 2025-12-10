# Part 2a: Similarity Check (Argument-based)
import os
import cv2
import numpy as np
import logging
import shutil
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
import sys

# Correct the path for the new Windows environment
# sys.path.append('/mnt/d/tendency/.venv_new/lib/python3.12/site-packages')

# --- Globals ---
MAP_FILE = "_map.txt"
METRIC = 'cosine'
# Very strict tolerance for finding near-duplicates.
DEDUPLICATION_TOLERANCE = 0.26 
MIN_SAMPLES = 2

# --- Logging Setup ---
logging.basicConfig(
    filename='log_part2a.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# --- Function ---
def find_similar_images(input_dir, processed_face_to_original_map):
    logger.info("Starting similarity search with InsightFace embeddings...")
    
    # Initialize InsightFace - using the new Windows GPU environment
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("FaceAnalysis initialized with CUDA provider.")
    except Exception as e:
        logger.error(f"FaceAnalysis initialization failed: {e}", exc_info=True)
        logger.error("Please ensure CUDA and cuDNN are installed correctly for the Windows environment.")
        sys.exit(1)

    processed_dir = os.path.join(input_dir, "processed")
    rotated_dir = os.path.join(input_dir, "rotated")
    bbox_cropped_dir = os.path.join(input_dir, "bbox_cropped")
    bbox_rotated_dir = os.path.join(input_dir, "bbox_rotated")
    deleted_dir = os.path.join(input_dir, "deleted")
    if not os.path.exists(deleted_dir):
        os.makedirs(deleted_dir)

    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        logger.warning(f"'processed' directory not found or empty. Skipping similarity check.")
        return

    image_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith('.png')]
    logger.info(f"Found {len(image_files)} images in 'processed' directory for embedding extraction.")

    # --- Get Embeddings ---
    encodings = []
    image_path_list = []
    for img_path in image_files:
        try:
            bgr_image = cv2.imread(img_path)
            if bgr_image is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
            faces = app.get(bgr_image)
            if faces:
                encodings.append(faces[0].embedding)
                image_path_list.append(img_path)
            else:
                logger.warning(f"No face found in: {img_path}")
        except Exception as e:
            logger.error(f"Error processing {img_path} for embedding: {e}", exc_info=True)

    if len(encodings) < MIN_SAMPLES:
        logger.info("Not enough faces to perform duplicate clustering. Skipping.")
        return

    # --- Perform Clustering to find duplicates ---
    logger.info(f"Clustering to find duplicates with DBSCAN, metric={METRIC}, eps={DEDUPLICATION_TOLERANCE}")
    clustering = DBSCAN(metric=METRIC, eps=DEDUPLICATION_TOLERANCE, min_samples=MIN_SAMPLES).fit(np.array(encodings))
    labels = clustering.labels_

    # --- Group images by cluster label ---
    clusters = {}  # {label: [img_path, img_path, ...]}
    for i, label in enumerate(labels):
        if label != -1:  # We only care about clusters (duplicates), not noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(image_path_list[i])
    
    logger.info(f"Found {len(clusters)} groups of similar (duplicate) images.")

    # --- Loop through clusters and remove duplicates ---
    for label, group in clusters.items():
        # Keep the first image, delete the rest of the group
        keep_one = group.pop(0)
        logger.info(f"Cluster {label}: Keeping {os.path.basename(keep_one)}, removing {len(group)} duplicates.")
        
        for img_path in group:
            try:
                base_name, _ = os.path.splitext(os.path.basename(img_path))
                original_filename = processed_face_to_original_map.get(base_name)
                if not original_filename:
                    logger.warning(f"Original filename not found in map for {base_name}, cannot perform full deletion.")
                    continue
                
                original_ext = os.path.splitext(original_filename)[1]

                files_to_move = [
                    img_path,  # the processed png
                    os.path.join(rotated_dir, f"{base_name}{original_ext}"),
                    os.path.join(bbox_cropped_dir, f"{base_name}{original_ext}"),
                    os.path.join(bbox_rotated_dir, f"{base_name}{original_ext}"),
                    os.path.join(input_dir, original_filename)
                ]

                for file_path_to_move in files_to_move:
                    if os.path.exists(file_path_to_move):
                        shutil.move(file_path_to_move, os.path.join(deleted_dir, os.path.basename(file_path_to_move)))
                        logger.info(f"Moved duplicate file: {file_path_to_move}")
            except Exception as e:
                logger.error(f"Error while moving duplicates for {img_path}: {e}", exc_info=True)

def main():
    if len(sys.argv) != 2:
        print("Usage: python part2a_similarity.py <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    logger.info(f"Part 2a starting for directory: {input_dir}")
    try:
        processed_face_to_original_map = {}
        map_path = os.path.join(input_dir, MAP_FILE)
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ',' in line:
                        key, value = line.strip().split(',', 1)
                        processed_face_to_original_map[key] = value
        else:
            logger.error(f"Map file not found: {map_path}")
            sys.exit(1)

        find_similar_images(input_dir, processed_face_to_original_map)
        logger.info(f"Part 2a (similarity check) finished.")

    except Exception as e:
        logger.error(f"Fatal error in Part 2a: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
