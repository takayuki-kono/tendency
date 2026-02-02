#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


# python test_clustering.py person_classification/rotated
import os
import cv2
import numpy as np
import logging
import shutil
from sklearn.cluster import DBSCAN
import sys

sys.path.append('/mnt/d/tendency/.venv_new/lib/python3.12/site-packages')
from insightface.app import FaceAnalysis

# --- Parameters to Tune ---
# METRIC: 'cosine' or 'euclidean'
# Cosine is generally better for face embeddings.
METRIC = 'cosine' 

# TOLERANCE (eps): 
# - For 'cosine', a value between 0.4 and 0.6 is typical. Lower is stricter.
# - For 'euclidean', a value between 20 and 30 is typical. Lower is stricter.
TOLERANCE = 0.5

# MIN_SAMPLES: Minimum number of faces to form a cluster.
MIN_SAMPLES = 2

# --- Script --- 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Print logs directly to the console
)
logger = logging.getLogger(__name__)

def main():
    # The input directory is hardcoded to 'input_images' in the same directory as the script.
    input_dir = "input_images"
    if not os.path.isdir(input_dir):
        logger.error(f"Error: Directory not found at '{input_dir}'")
        sys.exit(1)

    # New output directory naming scheme
    output_base_dir = f"grouped_faces_insightface_{METRIC}"
    logger.info(f"--- Starting Clustering Test ---")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Metric: {METRIC}, Tolerance (eps): {TOLERANCE}, Min Samples: {MIN_SAMPLES}")
    logger.info(f"Results will be copied to: {output_base_dir}")

    # Clean up previous results
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir)

    # --- Initialize InsightFace ---
    logger.info("Initializing FaceAnalysis...")
    try:
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("FaceAnalysis initialized successfully.")
    except Exception as e:
        logger.error(f"FaceAnalysis initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Get Embeddings ---
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if not image_files:
        logger.warning("No image files found in the input directory.")
        return

    encodings = []
    image_path_list = []

    logger.info(f"Extracting embeddings from {len(image_files)} images...")
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        try:
            bgr_image = cv2.imread(img_path)
            if bgr_image is None: continue
            faces = app.get(bgr_image)
            if faces:
                encodings.append(faces[0].embedding)
                image_path_list.append(img_path)
            else:
                logger.warning(f"No face found in: {filename}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)

    if len(encodings) < MIN_SAMPLES:
        logger.warning("Not enough faces to perform clustering.")
        return

    # --- Perform Clustering ---
    logger.info(f"Performing DBSCAN clustering...")
    clustering = DBSCAN(metric=METRIC, eps=TOLERANCE, min_samples=MIN_SAMPLES).fit(np.array(encodings))
    labels = clustering.labels_

    # --- Save Results ---
    logger.info("Copying images to cluster directories...")
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f'Estimated number of clusters: {num_clusters}')

    # Create unclassified dir for noise points
    unclassified_dir = os.path.join(output_base_dir, "unclassified")
    os.makedirs(unclassified_dir, exist_ok=True)

    for i, label in enumerate(labels):
        if label == -1:
            cluster_dir = unclassified_dir
        else:
            cluster_dir = os.path.join(output_base_dir, f"person_{label}")
        
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        shutil.copy(image_path_list[i], cluster_dir)

    logger.info("--- Clustering Test Finished ---")

if __name__ == "__main__":
    main()
