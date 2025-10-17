# Part 1: Setup and InsightFace Processing (Argument-based)
import os
import cv2
import numpy as np
import logging
import shutil
from icrawler.builtin import GoogleImageCrawler
from insightface.app import FaceAnalysis
import sys

sys.path.append('/mnt/d/tendency/.venv_new/lib/python3.12/site-packages')

# --- Globals ---
MAX_NUM = 100
IMG_SIZE = 224
MAP_FILE = "_map.txt"

# --- Logging Setup ---
logging.basicConfig(
    filename='log_part1.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# --- Functions (remain the same) ---
def setup_crawler(storage_dir):
    return GoogleImageCrawler(storage={'root_dir': storage_dir})

def download_images(keyword, max_num):
    search_terms = [
        (keyword, keyword),
        (f"{keyword} 正面", f"{keyword}_正面"),
        (f"{keyword} 顔", f"{keyword}_顔"),
        (f"{keyword} 昔", f"{keyword}_昔"),
        (f"{keyword} 現在", f"{keyword}_現在")
    ]
    for search_keyword, storage_dir in search_terms:
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"Starting download for keyword: {search_keyword}, storage: {storage_dir}")
        crawler = setup_crawler(storage_dir)
        crawler.crawl(keyword=search_keyword, max_num=max_num)
        logger.info(f"Finished download for {search_keyword}")

def rename_files(keyword):
    folders = [keyword, f"{keyword}_昔", f"{keyword}_現在", f"{keyword}_正面", f"{keyword}_顔"]
    for folder in folders:
        if not os.path.exists(folder): continue
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            old_path = os.path.join(folder, file)
            new_filename = f"{folder}_{file}"
            new_path = os.path.join(folder, new_filename)
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                logger.error(f"Error renaming {old_path}: {e}")

def consolidate_files(keyword, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    folders = [keyword, f"{keyword}_昔", f"{keyword}_現在", f"{keyword}_正面", f"{keyword}_顔"]
    for folder in folders:
        if not os.path.exists(folder): continue
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            src_path = os.path.join(folder, file)
            dst_path = os.path.join(output_dir, file)
            try:
                shutil.move(src_path, dst_path)
            except Exception as e:
                logger.error(f"Error moving {src_path}: {e}")
    for folder in folders:
        if os.path.exists(folder) and not os.listdir(folder):
            shutil.rmtree(folder)
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    for i, file in enumerate(files, 1):
        old_path = os.path.join(output_dir, file)
        ext = os.path.splitext(file)[1].lower()
        new_filename = f"{output_dir}_{i:03d}{ext}"
        new_path = os.path.join(output_dir, new_filename)
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            logger.error(f"Error renaming {old_path}: {e}")

def detect_and_crop_faces(input_dir):
    logger.info(f"--- Entering detect_and_crop_faces (Full Logic) for directory: {input_dir} ---")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))

    processed_dir = os.path.join(input_dir, "processed")
    rotated_dir = os.path.join(input_dir, "rotated")
    bbox_cropped_dir = os.path.join(input_dir, "bbox_cropped")
    bbox_rotated_dir = os.path.join(input_dir, "bbox_rotated")
    
    for d in [processed_dir, rotated_dir, bbox_cropped_dir, bbox_rotated_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    logger.info(f"Found {len(files)} files to process.")
    processed_face_to_original_map = {}

    for filename in files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        try:
            faces = app.get(img)
        except Exception as e:
            logger.error(f"InsightFace app.get() failed for {filename}: {e}")
            continue

        if not faces: continue
        
        for face_idx, face in enumerate(faces):
            try:
                original_base_name, ext = os.path.splitext(filename)
                current_face_base_name = f"{original_base_name}_{face_idx}"
                bbox = face.bbox.astype(int)
                x_min, y_min, x_max, y_max = bbox
                x_min = max(0, x_min - int(0.1 * (x_max - x_min)))
                y_min = max(0, y_min - int(0.1 * (y_max - y_min)))
                x_max = min(img.shape[1], x_max + int(0.1 * (x_max - x_min)))
                y_max = min(img.shape[0], y_max + int(0.1 * (y_max - y_min)))
                bbox_img = img[y_min:y_max, x_min:x_max]
                if bbox_img.size == 0: continue
                cv2.imwrite(os.path.join(bbox_cropped_dir, f"{current_face_base_name}{ext}"), bbox_img)

                bbox_faces = app.get(bbox_img)
                if not bbox_faces: continue
                lmk = bbox_faces[0].landmark_2d_106
                dx = lmk[86][0] - lmk[0][0]
                dy = lmk[86][1] - lmk[0][1]
                angle = np.arctan2(dx, -dy) * 180 / np.pi
                
                center = (bbox_img.shape[1] / 2, bbox_img.shape[0] / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                bbox_rotated_img = cv2.warpAffine(bbox_img, M, (bbox_img.shape[1], bbox_img.shape[0]))
                bbox_rotated_path = os.path.join(bbox_rotated_dir, f"{current_face_base_name}{ext}")
                cv2.imwrite(bbox_rotated_path, bbox_rotated_img)

                reloaded_rotated_img = cv2.imread(bbox_rotated_path)
                if reloaded_rotated_img is None: continue
                final_faces = app.get(reloaded_rotated_img)
                if not final_faces: continue
                final_lmk = final_faces[0].landmark_2d_106

                ry_top = min(final_lmk[49][1], final_lmk[104][1])
                ry_bottom = final_lmk[0][1]
                rx_center = final_lmk[86][0]
                r_size = max(ry_bottom - ry_top, reloaded_rotated_img.shape[1] // 2)
                rx_min_crop = int(rx_center - r_size // 2)
                rx_max_crop = int(rx_center + r_size // 2)
                ry_min_crop = int(ry_top)
                ry_max_crop = int(ry_top + r_size)

                rh, rw = reloaded_rotated_img.shape[:2]
                r_pad_top = max(0, -ry_min_crop)
                r_pad_bottom = max(0, ry_max_crop - rh)
                r_pad_left = max(0, -rx_min_crop)
                r_pad_right = max(0, rx_max_crop - rw)
                final_img_padded = cv2.copyMakeBorder(reloaded_rotated_img, r_pad_top, r_pad_bottom, r_pad_left, r_pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
                rx_min_crop += r_pad_left; rx_max_crop += r_pad_left
                ry_min_crop += r_pad_top; ry_max_crop += r_pad_top

                final_cropped_face = final_img_padded[ry_min_crop:ry_max_crop, rx_min_crop:rx_max_crop]
                if final_cropped_face.size == 0: continue

                final_resized_face = cv2.resize(final_cropped_face, (IMG_SIZE, IMG_SIZE))

                cv2.imwrite(os.path.join(rotated_dir, f"{current_face_base_name}{ext}"), final_resized_face)
                cv2.imwrite(os.path.join(processed_dir, f"{current_face_base_name}.png"), cv2.cvtColor(final_resized_face, cv2.COLOR_BGR2GRAY))

                logger.info(f"Saved final processed files for {current_face_base_name}")
                processed_face_to_original_map[current_face_base_name] = filename
            except Exception as e:
                logger.error(f"Error processing face {face_idx} in {filename}: {e}", exc_info=True)

    # --- Cleanup unprocessed files ---
    logger.info("Cleaning up original files where no faces were detected...")
    processed_original_files = set(processed_face_to_original_map.values())
    unprocessed_files = set(files) - processed_original_files
    
    deleted_dir = os.path.join(input_dir, "deleted")
    if not os.path.exists(deleted_dir):
        os.makedirs(deleted_dir)

    for unprocessed_file in unprocessed_files:
        if unprocessed_file == MAP_FILE:
            continue
        try:
            shutil.move(os.path.join(input_dir, unprocessed_file), os.path.join(deleted_dir, unprocessed_file))
            logger.info(f"Moved unprocessed file (no face found): {unprocessed_file}")
        except Exception as e:
            logger.error(f"Error moving unprocessed file {unprocessed_file}: {e}")

    # Save the map for part 2a
    map_path = os.path.join(input_dir, MAP_FILE)
    with open(map_path, 'w', encoding='utf-8') as f:
        for key, value in processed_face_to_original_map.items():
            f.write(f"{key},{value}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python part1_setup.py <keyword> <output_dir>")
        sys.exit(1)
    
    keyword = sys.argv[1]
    output_dir = sys.argv[2]

    logger.info(f"Part 1 starting for keyword: '{keyword}', output_dir: '{output_dir}'")
    try:
        download_images(keyword, MAX_NUM)
        rename_files(keyword)
        consolidate_files(keyword, output_dir)
        detect_and_crop_faces(output_dir)
        logger.info(f"Part 1 finished for keyword: {keyword}")
    except Exception as e:
        logger.error(f"Fatal error in Part 1: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()