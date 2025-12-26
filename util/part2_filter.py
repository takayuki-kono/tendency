# Part 2: Face Recognition Filtering and Cleanup
import os
import cv2
import numpy as np
import logging
import shutil
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import DBSCAN
from collections import defaultdict
import sys

# Add site-packages to path and import face_recognition first
sys.path.append('/mnt/d/tendency/.venv_new/lib/python3.12/site-packages')
import face_recognition

# --- Globals for Part 2 ---
SIMILARITY_THRESHOLD = 0.7
IMG_SIZE = 224
WORKDIR_FILE = "_workdir.txt"

# --- Logging Setup ---
logging.basicConfig(
    filename='log_part2.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# --- Functions from original script ---
def find_similar_images(input_dir, processed_face_to_original_map):
    processed_dir = os.path.join(input_dir, "processed")
    resized_dir = os.path.join(input_dir, "resized")
    rotated_dir = os.path.join(input_dir, "rotated")
    bbox_cropped_dir = os.path.join(input_dir, "bbox_cropped")
    bbox_rotated_dir = os.path.join(input_dir, "bbox_rotated")
    deleted_dir = os.path.join(input_dir, "deleted")
    if os.path.exists(deleted_dir):
        shutil.rmtree(deleted_dir)
    os.makedirs(deleted_dir, exist_ok=True)
    logger.info(f"{processed_dir} の類似画像検索開始")
    image_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if os.path.isfile(os.path.join(processed_dir, f))]
    logger.info(f"{os.path.basename(processed_dir)} で {len(image_files)} 画像を検出")

    def compare_images(img_path1, img_path2):
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None or img1.shape != img2.shape:
            return 0.0
        score = ssim(img1, img2)
        return score

    groups = []
    used_images = set()
    for i, img1_path in enumerate(image_files):
        if img1_path in used_images:
            continue
        current_group = [img1_path]
        for j, img2_path in enumerate(image_files[i+1:], i+1):
            if img2_path in used_images:
                continue
            score = compare_images(img1_path, img2_path)
            if score >= SIMILARITY_THRESHOLD:
                current_group.append(img2_path)
        if len(current_group) > 1:
            groups.append(current_group)
            used_images.update(current_group)
        else:
            used_images.add(img1_path)

    logger.info(f"{len(groups)} グループを検出")
    for group_idx, group in enumerate(groups, 1):
        if len(group) < 2:
            continue
        keep_img_path = group[0]
        for img_path in group[1:]:
            try:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                original_filename_with_ext = processed_face_to_original_map.get(base_name)
                if not original_filename_with_ext:
                    continue
                _, original_ext = os.path.splitext(original_filename_with_ext)

                files_to_delete = [
                    os.path.join(processed_dir, f"{base_name}.png"),
                    os.path.join(resized_dir, f"{base_name}{original_ext}"),
                    os.path.join(rotated_dir, f"{base_name}{original_ext}"),
                    os.path.join(bbox_cropped_dir, f"{base_name}{original_ext}"),
                    os.path.join(bbox_rotated_dir, f"{base_name}{original_ext}"),
                    os.path.join(input_dir, original_filename_with_ext)
                ]

                for file_path_to_move in files_to_delete:
                    if os.path.exists(file_path_to_move):
                        destination_path = os.path.join(deleted_dir, os.path.basename(file_path_to_move))
                        shutil.move(file_path_to_move, destination_path)
                        logger.info(f"Moved similar file: {file_path_to_move} -> {destination_path}")
            except Exception as e:
                logger.error(f"Error processing similar file {img_path}: {e}")

def filter_by_main_person_cnn_euclidean(input_dir, processed_face_to_original_map):
    logger.info("人物ごとの画像グループ分けとフィルタリングを開始 (face_recognition, HOG, Euclidean, DBSCAN)")
    rotated_dir = os.path.join(input_dir, "rotated")
    if not os.path.exists(rotated_dir) or not os.listdir(rotated_dir):
        logger.warning(f"rotatedディレクトリが見つからないか空です: {rotated_dir}。人物フィルタリングをスキップします。")
        return

    image_files = [f for f in os.listdir(rotated_dir) if os.path.isfile(os.path.join(rotated_dir, f))]
    encodings = []
    image_path_list = []

    logger.info("顔のエンコーディングを抽出中 (HOGモデル)...")
    for filename in image_files:
        img_path = os.path.join(rotated_dir, filename)
        try:
            bgr_image = cv2.imread(img_path)
            if bgr_image is None:
                logger.warning(f"画像の読み込みに失敗しました: {img_path}")
                continue
            
            image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(image_rgb, model="hog") # Using HOG model
            if face_locations:
                face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    image_path_list.append(img_path)
            else:
                logger.warning(f"face_recognition(HOG)が顔を検出できませんでした: {img_path}")
        except Exception as e:
            logger.error(f"face_recognition(HOG)処理中にエラーが発生しました {img_path}: {e}")
            continue

    if len(encodings) < 2:
        logger.info("クラスタリング対象の画像が2枚未満のため、人物フィルタリングを終了します。")
        return

    TOLERANCE = 0.6 # Default for HOG
    encodings_np = np.array(encodings)
    clustering = DBSCAN(metric='euclidean', eps=TOLERANCE, min_samples=1).fit(encodings_np)
    labels = clustering.labels_

    if len(labels) == 0:
        logger.warning("No clusters found.")
        return

    # Find the largest cluster (excluding noise points)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        logger.warning("No main cluster found, only noise.")
        main_cluster_label = -2 # A label that will not match anything
    else:
        main_cluster_label = unique_labels[np.argmax(counts)]

    images_to_keep = []
    for i, label in enumerate(labels):
        if label == main_cluster_label:
            images_to_keep.append(image_path_list[i])

    deleted_dir = os.path.join(input_dir, "deleted")
    if not os.path.exists(deleted_dir):
        os.makedirs(deleted_dir)

    for path in image_path_list:
        if path not in images_to_keep:
            shutil.move(path, os.path.join(deleted_dir, os.path.basename(path)))
            logger.info(f"Moved non-main person file: {path}")

def cleanup_directories(input_dir):
    logger.info("クリーンアップ開始")
    dirs_to_delete = ["processed", "bbox_cropped", "bbox_rotated"]
    for dir_name in dirs_to_delete:
        dir_path = os.path.join(input_dir, dir_name)
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"成功的に削除されたディレクトリ: {dir_path}")
        except Exception as e:
            logger.error(f"ディレクトリ削除エラー {dir_path}: {e}")

def main():
    try:
        if not os.path.exists(WORKDIR_FILE):
            logger.error(f"Work directory file not found: {WORKDIR_FILE}")
            return
        with open(WORKDIR_FILE, "r") as f:
            input_dir = f.read().strip()
        logger.info(f"Processing directory: {input_dir}")

        # Load the map from part 1
        processed_face_to_original_map = {}
        map_file = '_map.txt'
        if os.path.exists(map_file):
            with open(map_file, 'r', encoding='utf-8') as f:
                for line in f:
                    key, value = line.strip().split(',')
                    processed_face_to_original_map[key] = value

        find_similar_images(input_dir, processed_face_to_original_map)
        filter_by_main_person_cnn_euclidean(input_dir, processed_face_to_original_map)
        cleanup_directories(input_dir)
        logger.info(f"Part 2 finished for directory: {input_dir}")

    except Exception as e:
        logger.error(f"Fatal error in Part 2: {e}", exc_info=True)

if __name__ == "__main__":
    main()
