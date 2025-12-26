import os
import cv2
import numpy as np
import logging
import shutil
import argparse
import sys
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[
        logging.FileHandler("apply_filter.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# InsightFace settings
DET_SIZE = (320, 320)
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# DBSCAN settings
METRIC = 'cosine'
TOLERANCE = 0.5 
MIN_SAMPLES = 2

def filter_folder(folder_path, app):
    """
    Filters images in a specific folder using InsightFace and DBSCAN.
    Moves rejected images to a 'deleted' subdirectory.
    """
    logger.info(f"Processing folder: {folder_path}")
    
    # Collect image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(folder_path, f))]
    
    if not image_files:
        logger.info(f"No images found in {folder_path}. Skipping.")
        return

    encodings = []
    image_path_list = []
    files_to_delete = [] # List of (path, reason)

    # 1. Face Detection & Quality Check
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        try:
            # cv2.imread fails on paths with non-ascii chars sometimes on Windows if not handled, 
            # but usually fine with modern python/opencv. If issues arise, use numpy fromfile.
            bgr_image = cv2.imread(img_path)
            if bgr_image is None:
                files_to_delete.append((img_path, "Read Error"))
                continue

            faces = app.get(bgr_image)

            if not faces:
                files_to_delete.append((img_path, "No Face Detected"))
                continue
            
            # Use the largest face if multiple found (or just the first one returned by insightface which is usually sorted by score/size? actually usually by detection order, but let's assume single person focus)
            # Better: find largest face
            face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            
            x1, y1, x2, y2 = face.bbox
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Aspect Ratio Check
            aspect_ratio = face_height / face_width if face_width > 0 else 0
            if aspect_ratio < 0.9 or aspect_ratio > 1.8:
                files_to_delete.append((img_path, f"Bad Aspect Ratio ({aspect_ratio:.2f})"))
                continue

            # Passed quality checks (resolution/aspect)
            
            # Pose Check (Pitch)
            # pose = [pitch, yaw, roll]
            # "Forward leaning" usually means significant pitch.
            # Threshold: 20 degrees (adjustable)
            if face.pose is not None:
                pitch, yaw, roll = face.pose
                # Assuming pitch is the first element. 
                # Positive/Negative direction depends on the model, but usually we care about magnitude for "leaning" 
                # (either up or down, though "forward" is specific).
                # Let's filter if absolute pitch is too high.
                # User reported 20 was not enough. Lowering to 10.
                if abs(pitch) > 15:
                    msg = f"Bad Pose (Pitch={pitch:.1f}, Yaw={yaw:.1f})"
                    files_to_delete.append((img_path, msg))
                    logger.info(f"Marked for deletion: {img_path} ({msg})")
                    continue

            encodings.append(face.embedding)
            image_path_list.append(img_path)

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            files_to_delete.append((img_path, "Processing Error"))

    # 2. Clustering (only if we have enough faces)
    if len(encodings) < 2:
        logger.warning(f"Not enough valid faces in {folder_path} to cluster (Found {len(encodings)}). Skipping clustering.")
        # If 0 or 1 face, we can't really cluster. We just keep them (unless they were already marked for deletion above).
    else:
        try:
            logger.info(f"Clustering {len(encodings)} faces in {folder_path}...")
            clustering = DBSCAN(metric=METRIC, eps=TOLERANCE, min_samples=MIN_SAMPLES).fit(np.array(encodings))
            labels = clustering.labels_
            
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            
            if len(counts) > 0:
                main_cluster_label = unique_labels[np.argmax(counts)]
                logger.info(f"  Main cluster: Label {main_cluster_label} with {counts.max()} images.")
                
                for i, label in enumerate(labels):
                    if label != main_cluster_label:
                        files_to_delete.append((image_path_list[i], f"Not Main Person (Label {label})"))
            else:
                logger.warning("  No main cluster found (all noise). Marking all as noise.")
                # If everything is noise, maybe we shouldn't delete everything? 
                # Or strictly delete everything? 
                # Usually 'noise' means they don't look like each other. 
                # Let's be safe: if NO cluster is found, maybe keep them or delete all?
                # User's original script logic: "No main cluster found, only noise" -> returns, so nothing deleted from clustering phase.
                # But here we want to filter. 
                # Let's follow original script logic: if no main cluster, do nothing for clustering phase.
                pass

        except Exception as e:
            logger.error(f"Error during clustering in {folder_path}: {e}")

    # 3. Move rejected files
    if files_to_delete:
        deleted_dir = os.path.join(folder_path, "deleted")
        os.makedirs(deleted_dir, exist_ok=True)
        
        for file_path, reason in files_to_delete:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(deleted_dir, filename)
            
            # Avoid overwriting if file exists in deleted (e.g. from previous run)
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                dest_path = os.path.join(deleted_dir, f"{base}_dup{ext}")
            
            try:
                shutil.move(file_path, dest_path)
                logger.info(f"Moved to deleted: {filename} Reason: {reason}")
            except Exception as e:
                logger.error(f"Failed to move {filename}: {e}")

def process_root_directory(root_dir, app):
    """
    Recursively finds leaf directories containing images and processes them.
    """
    for root, dirs, files in os.walk(root_dir):
        # Check if this directory contains images
        has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in files)
        
        # We only want to process "leaf" directories or directories that actually contain the dataset images.
        # In the user's structure: train/label/person/image.jpg
        # We should process 'person' directories.
        # If a directory has images, we process it.
        if has_images:
            # Skip 'deleted' folders to avoid re-processing
            if os.path.basename(root) == "deleted":
                continue
                
            filter_folder(root, app)

def main():
    parser = argparse.ArgumentParser(description="Apply InsightFace filtering to existing datasets.")
    parser.add_argument("--train_dir", default="train", help="Path to train directory")
    parser.add_argument("--val_dir", default="validation", help="Path to validation directory")
    args = parser.parse_args()

    # Initialize InsightFace once
    logger.info("Initializing FaceAnalysis...")
    try:
        app = FaceAnalysis(providers=PROVIDERS)
        app.prepare(ctx_id=0, det_size=DET_SIZE)
    except Exception as e:
        logger.error(f"Failed to initialize FaceAnalysis: {e}")
        sys.exit(1)

    # Process Train
    if os.path.exists(args.train_dir):
        logger.info(f"Scanning Train Directory: {args.train_dir}")
        process_root_directory(args.train_dir, app)
    else:
        logger.warning(f"Train directory not found: {args.train_dir}")

    # Process Validation
    if os.path.exists(args.val_dir):
        logger.info(f"Scanning Validation Directory: {args.val_dir}")
        process_root_directory(args.val_dir, app)
    else:
        logger.warning(f"Validation directory not found: {args.val_dir}")

    logger.info("All processing complete.")

if __name__ == "__main__":
    main()
