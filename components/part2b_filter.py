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
# InsightFace 106 Landmarks for face position check
LEFT_INNER_EYE_IDX = 89
RIGHT_INNER_EYE_IDX = 39

# InsightFace 68 Landmarks (3D) for cheek width check
# Index 1: Right cheek contour (upper)
# Index 15: Left cheek contour (upper)
LEFT_CHEEK_3D_IDX = 15
RIGHT_CHEEK_3D_IDX = 1
CHEEK_WIDTH_3D_RATIO_MIN = 0.7  # 3D頬幅/顔高さ の最小比率

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
    skipped_face_position = 0
    skipped_cheek_3d = 0

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

                # 縮尺チェック1: アスペクト比（潰れすぎ/伸びすぎ） - 無効化
                face_width = x2 - x1
                face_height = y2 - y1
                # aspect_ratio = face_height / face_width if face_width > 0 else 0
                #
                # if aspect_ratio < 0.9 or aspect_ratio > 1.8:
                #     logger.info(f"Skipped (abnormal aspect ratio {aspect_ratio:.3f}): {img_path}")
                #     skipped_aspect += 1
                #     continue

                # 顔位置フィルター (Face Position Filter)
                lmk = face.landmark_2d_106
                if lmk is not None:
                    lx, ly = lmk[LEFT_INNER_EYE_IDX]
                    rx, ry = lmk[RIGHT_INNER_EYE_IDX]
                    center_x = img_w / 2.0
                    d_left = lx - center_x
                    d_right = center_x - rx
                    
                    if d_left <= 0 or d_right <= 0:
                        logger.info(f"Skipped (face_pos_invalid d_left={d_left:.1f} d_right={d_right:.1f}): {img_path}")
                        skipped_face_position += 1
                        continue

                # 3D頬幅フィルター (3D Cheek Width Filter) - 無効化
                # lmk3d = face.landmark_3d_68
                # if lmk3d is not None:
                #     left_cheek = lmk3d[LEFT_CHEEK_3D_IDX]
                #     right_cheek = lmk3d[RIGHT_CHEEK_3D_IDX]
                #     
                #     # 3D Euclidean distance
                #     cheek_dist_3d = np.sqrt(
                #         (left_cheek[0] - right_cheek[0])**2 +
                #         (left_cheek[1] - right_cheek[1])**2 +
                #         (left_cheek[2] - right_cheek[2])**2
                #     )
                #     
                #     # Normalize by face height
                #     cheek_ratio_3d = cheek_dist_3d / face_height if face_height > 0 else 0
                #     
                #     if cheek_ratio_3d < CHEEK_WIDTH_3D_RATIO_MIN:
                #         logger.info(f"Skipped (narrow 3D cheek ratio={cheek_ratio_3d:.3f}): {img_path}")
                #         skipped_cheek_3d += 1
                #         continue
                # else:
                #     logger.warning(f"No 3D landmarks for {img_path}, skipping 3D cheek filter.")

                # 全チェック通過
                embedding = face.embedding
                encodings.append(embedding)
                image_path_list.append(img_path)
            else:
                logger.warning(f"InsightFace did not find a face in: {img_path}")
        except Exception as e:
            logger.error(f"Error during InsightFace processing for {img_path}: {e}", exc_info=True)

    logger.info(f"Filtering summary: skipped {skipped_aspect} (aspect), {skipped_resolution} (resolution), {skipped_face_position} (face_position)")

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

    # deleted ディレクトリは親ディレクトリに作成（論理削除の場合のみ）
    parent_dir = os.path.dirname(input_dir)
    
    # Keep top-N clusters (default: 2) to mitigate "second person" dominance.
    keep_top_n = 2
    order = np.argsort(-counts)  # cluster sizes desc
    keep_labels = unique_labels[order[:min(keep_top_n, len(unique_labels))]].tolist()
    keep_counts_sorted = counts[order].tolist()
    logger.info(f"Keeping top-{len(keep_labels)} clusters: {keep_labels} (counts_sorted={keep_counts_sorted})")

    # Create destination folders per kept cluster rank under parent_dir
    clusters_root_dir = os.path.join(parent_dir, "person_clusters")
    os.makedirs(clusters_root_dir, exist_ok=True)
    label_to_rank = {lab: (ri + 1) for ri, lab in enumerate(keep_labels)}
    rank_to_dir = {}
    for rank in label_to_rank.values():
        d = os.path.join(clusters_root_dir, f"person_{rank}")
        os.makedirs(d, exist_ok=True)
        rank_to_dir[rank] = d

    # Move kept images into per-person folders (remove from input_dir)
    for i, label in enumerate(labels):
        if label in label_to_rank:
            src = image_path_list[i]
            rank = label_to_rank[label]
            dst = os.path.join(rank_to_dir[rank], os.path.basename(src))
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
            except Exception as e:
                logger.error(f"Error moving kept file {src} -> {dst}: {e}")

    deleted_dir = os.path.join(parent_dir, "deleted_outliers")
    if not physical_delete and not os.path.exists(deleted_dir):
        os.makedirs(deleted_dir)

    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for path in all_images:
        # At this point, kept images have been moved out of input_dir.
        # Remaining files in input_dir are outliers/noise and should be removed/moved.
        try:
            if physical_delete:
                os.remove(path)
                logger.info(f"Deleted outlier file: {path}")
            else:
                dst = os.path.join(deleted_dir, os.path.basename(path))
                if os.path.exists(dst): os.remove(dst)
                shutil.move(path, dst)
                logger.info(f"Moved outlier file: {path}")
        except Exception as e:
            logger.error(f"Error removing {path}: {e}")

def cleanup_directories(input_dir):
    """
    フィルタリング後の整理:
    - 空になった rotated フォルダを削除
    """
    logger.info("Starting final cleanup.")
    
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