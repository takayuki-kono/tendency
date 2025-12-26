import os
from PIL import Image
import numpy as np
import logging

# --- 設定 ---
# この閾値以下の中央値を持つ画像が削除されます
MEDIAN_THRESHOLD = 75
# 対象のディレクトリ
TARGET_DIRS = ['preprocessed_multitask/train', 'preprocessed_multitask/validation']
# ログファイル名
LOG_FILE = 'filter_dark_images_log.txt'
# ---

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

def filter_dark_images():
    deleted_count = 0
    processed_count = 0

    logger.info(f"Starting image filtering process.")
    logger.info(f"Target directories: {TARGET_DIRS}")
    logger.info(f"Median pixel value threshold: {MEDIAN_THRESHOLD}")

    for target_dir in TARGET_DIRS:
        if not os.path.exists(target_dir):
            logger.warning(f"Directory not found, skipping: {target_dir}")
            continue

        logger.info(f"Processing directory: {target_dir}")
        for root, _, files in os.walk(target_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    processed_count += 1
                    img_path = os.path.join(root, filename)
                    try:
                        with Image.open(img_path) as img:
                            grayscale_img = img.convert('L')
                            pixel_values = np.array(grayscale_img)
                            median_value = np.median(pixel_values)

                            if median_value <= MEDIAN_THRESHOLD:
                                deleted_count += 1
                                logger.info(f"DELETING: {img_path} (Median: {median_value})")
                                os.remove(img_path)

                    except Exception as e:
                        logger.error(f"Could not process file {img_path}: {e}")

                    if processed_count % 500 == 0:
                        logger.info(f"Processed {processed_count} images...")

    logger.info("--- Filtering Complete ---")
    logger.info(f"Total images processed: {processed_count}")
    logger.info(f"Total images deleted: {deleted_count}")

if __name__ == "__main__":
    filter_dark_images()
