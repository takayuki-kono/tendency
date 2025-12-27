# Part 1: Setup and InsightFace Processing (Argument-based)
import os
import cv2
import numpy as np
import logging
import shutil
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from insightface.app import FaceAnalysis
import sys

# --- Globals ---
MAX_NUM = 100
IMG_SIZE = 224
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'log_part1.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# --- Safe I/O Functions ---
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

def imwrite_safe(path, img):
    try:
        ext = os.path.splitext(path)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            with open(path, mode='wb') as f:
                n.tofile(f)
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to write image {path}: {e}")
        return False

# --- Functions ---
def setup_crawler(storage_dir):
    return BingImageCrawler(storage={'root_dir': storage_dir}, downloader_threads=4)

# --- Google Scraping Helper ---
import requests
from bs4 import BeautifulSoup
import re
import base64

def scrape_google_images(keyword, max_num, output_dir):
    logger.info(f"Starting Google scrape for: {keyword}, target: {max_num}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36"
    }
    
    url = f"https://www.google.com/search?q={keyword}&tbm=isch"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Regex to find image URLs (thumbnails often encoded in base64 or simpler http links in script tags)
        # Google often stores images in AF_initDataCallback
        # This is a heuristic approach for "fast" scraping of initial results
        
        # Method A: Look for direct image sources (thumbnails)
        img_tags = soup.find_all("img")
        
        count = 0
        for img in img_tags:
            if count >= max_num:
                break
                
            src = img.get('src')
            if not src:
                src = img.get('data-src')
                
            if not src or not src.startswith('http'):
                continue
                
            # Skip google logos
            if 'google' in src and 'logo' in src:
                continue

            try:
                img_data = requests.get(src, timeout=5).content
                file_path = os.path.join(output_dir, f"google_{count:04d}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(img_data)
                count += 1
            except Exception as e:
                pass
                
        # Method B: Regex for larger images in scripts (more fragile but better quality if works)
        # matches = re.findall(r'\"(https?://[^\"]+?\.jpg)\"', response.text)
        # ... skipped for stability per user request for "fast/simple" ...
        
        logger.info(f"Google scrape finished. Downloaded: {count} images.")
        
    except Exception as e:
        logger.error(f"Google scrape failed for {keyword}: {e}")

def download_images(keyword, max_num):
    search_terms = [
        (keyword, keyword),
        (f"{keyword} 正面", f"{keyword}_正面"),
        (f"{keyword} 顔", f"{keyword}_顔"),
        (f"{keyword} 昔", f"{keyword}_昔"),
        (f"{keyword} 現在", f"{keyword}_現在"),
        (f"{keyword} ドラマ", f"{keyword}_ドラマ"),
        (f"{keyword} CM", f"{keyword}_CM"),
        (f"{keyword} インタビュー", f"{keyword}_インタビュー"),
        (f"{keyword} 高画質", f"{keyword}_高画質")
    ]
    
    for search_keyword, storage_dir in search_terms:
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
        os.makedirs(storage_dir, exist_ok=True)
        
        logger.info(f"Starting download for: {search_keyword}, storage: {storage_dir}")
        
        # 1. Bing Download (Original)
        try:
            crawler = setup_crawler(storage_dir)
            crawler.crawl(keyword=search_keyword, max_num=max_num)
            logger.info(f"Bing finished for {search_keyword}")
        except Exception as e:
            logger.error(f"Bing Crawler failed for {search_keyword}: {e}")

        # 2. Google Download (New)
        # Save to the same folder, consolidate_files handles renaming/moving anyway
        try:
            # Create a subfolder to avoid filename collisions temporarily? 
            # consolidate_files flattens subdirs so we can just download here with unique prefix
            # BUT icrawler uses 000001.jpg etc.
            # My google scraper uses google_0000.jpg. Safe to mix.
            scrape_google_images(search_keyword, max_num, storage_dir)
        except Exception as e:
            logger.error(f"Google Scraper failed for {search_keyword}: {e}")

def consolidate_files(keyword, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # keywordで始まる全てのディレクトリを対象にする（再帰的ではなくトップレベルのフォルダのみ）
    # ただし output_dir 自身は除外
    root_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith(keyword) and d != output_dir]
    
    # Move files
    for folder in root_dirs:
        # Avoid processing output directory itself if it matches prefix (unlikely due to earlier check but safe)
        if os.path.abspath(folder) == os.path.abspath(output_dir): continue

        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            src_path = os.path.join(folder, file)
            # engine名などを含むフォルダ名をプレフィックスにしてファイル名衝突回避
            safe_folder = "".join(c for c in folder if c.isalnum() or c in (' ', '_', '-')).strip()
            temp_name = f"{safe_folder}_{file}"
            dst_path = os.path.join(output_dir, temp_name)
            try:
                shutil.move(src_path, dst_path)
            except Exception as e:
                logger.error(f"Error moving {src_path}: {e}")
                
    # Remove empty source folders
    for folder in root_dirs:
        try:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        except Exception as e:
            logger.warning(f"Failed to remove folder {folder}: {e}")
            
    # Final Rename to simple format: img_0001.jpg
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    files.sort()
    for i, file in enumerate(files, 1):
        old_path = os.path.join(output_dir, file)
        ext = os.path.splitext(file)[1].lower()
        if ext == '.jpeg': ext = '.jpg'
        
        new_filename = f"img_{i:04d}{ext}"
        new_path = os.path.join(output_dir, new_filename)
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            logger.error(f"Error renaming {old_path}: {e}")

def detect_and_crop_faces(input_dir):
    """
    顔検出・回転補正・クロップを行い、{input_dir}/rotated/ に最終画像のみ保存する。
    中間ファイル（bbox_cropped, bbox_rotated, processed等）は作成しない。
    """
    logger.info(f"--- detect_and_crop_faces for: {input_dir} ---")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))

    # Output directory: {input_dir}/rotated/
    rotated_dir = os.path.join(input_dir, "rotated")
    if os.path.exists(rotated_dir):
        shutil.rmtree(rotated_dir)
    os.makedirs(rotated_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    logger.info(f"Found {len(files)} files to process.")

    for filename in files:
        img_path = os.path.join(input_dir, filename)
        img = imread_safe(img_path)
        
        if img is None:
            logger.warning(f"Failed to read image: {filename}")
            continue

        try:
            faces = app.get(img)
        except Exception as e:
            logger.error(f"InsightFace failed for {filename}: {e}")
            continue

        if not faces:
            logger.debug(f"No faces found in {filename}")
            continue

        for face_idx, face in enumerate(faces):
            try:
                original_base_name, ext = os.path.splitext(filename)
                current_face_base_name = f"{original_base_name}_{face_idx}"
                
                # BBox Crop
                bbox = face.bbox.astype(int)
                x_min, y_min, x_max, y_max = bbox
                x_min = max(0, x_min - int(0.1 * (x_max - x_min)))
                y_min = max(0, y_min - int(0.1 * (y_max - y_min)))
                x_max = min(img.shape[1], x_max + int(0.1 * (x_max - x_min)))
                y_max = min(img.shape[0], y_max + int(0.1 * (y_max - y_min)))
                bbox_img = img[y_min:y_max, x_min:x_max]
                if bbox_img.size == 0: continue

                bbox_faces = app.get(bbox_img)
                if not bbox_faces: continue
                bbox_face = max(bbox_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                lmk = bbox_face.landmark_2d_106
                if lmk is None: continue

                # Rotation
                dx = lmk[86][0] - lmk[0][0]
                dy = lmk[86][1] - lmk[0][1]
                angle = np.arctan2(dx, -dy) * 180 / np.pi
                
                center = (bbox_img.shape[1] / 2, bbox_img.shape[0] / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                bbox_rotated_img = cv2.warpAffine(bbox_img, M, (bbox_img.shape[1], bbox_img.shape[0]))
                
                final_faces = app.get(bbox_rotated_img)
                if not final_faces: continue
                final_face = max(final_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                final_lmk = final_face.landmark_2d_106
                if final_lmk is None: continue

                # Eyebrow Crop (indices 49, 104 for eyebrows)
                ry_top_brow = min(final_lmk[49][1], final_lmk[104][1])
                ry_bottom = final_lmk[0][1]  # Chin
                rx_center = final_lmk[86][0]
                r_size = max(ry_bottom - ry_top_brow, bbox_rotated_img.shape[1] // 2)
                rx_min = int(rx_center - r_size // 2)
                rx_max = int(rx_center + r_size // 2)
                ry_min = int(ry_top_brow)
                ry_max = int(ry_top_brow + r_size)

                rh, rw = bbox_rotated_img.shape[:2]
                pad_t = max(0, -ry_min)
                pad_b = max(0, ry_max - rh)
                pad_l = max(0, -rx_min)
                pad_r = max(0, rx_max - rw)

                # Check padding ratio
                crop_h = ry_max - ry_min
                crop_w = rx_max - rx_min
                max_pad_ratio = max(
                    pad_t / crop_h if crop_h > 0 else 0,
                    pad_b / crop_h if crop_h > 0 else 0,
                    pad_l / crop_w if crop_w > 0 else 0,
                    pad_r / crop_w if crop_w > 0 else 0
                )

                if max_pad_ratio > 0.05:
                    logger.info(f"Skipped (excessive padding): {current_face_base_name}")
                    continue
                
                final_img_padded = cv2.copyMakeBorder(bbox_rotated_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
                rx_min += pad_l; rx_max += pad_l
                ry_min += pad_t; ry_max += pad_t

                final_cropped = final_img_padded[ry_min:ry_max, rx_min:rx_max]
                if final_cropped.size == 0: continue
                
                final_resized = cv2.resize(final_cropped, (IMG_SIZE, IMG_SIZE))
                
                # Get original height/width for quality tracking
                orig_h, orig_w = final_cropped.shape[:2]
                
                # Save to rotated/ only with original size info
                # Format: {name}_sz{width}.jpg
                output_path = os.path.join(rotated_dir, f"{current_face_base_name}_sz{orig_w}.jpg")
                imwrite_safe(output_path, final_resized)
                logger.info(f"Saved: {output_path} (orig size: {orig_w}x{orig_h})")

            except Exception as e:
                logger.error(f"Error processing face {face_idx} in {filename}: {e}", exc_info=True)

    # Clean up: Remove original files from input_dir (keep only rotated folder)
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")
    
    logger.info(f"Crop processing complete. Output in: {rotated_dir}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python part1_setup.py <keyword> <output_dir>")
        sys.exit(1)
    
    keyword = sys.argv[1]
    output_dir = sys.argv[2]

    logger.info(f"Part 1 starting for keyword: '{keyword}', output_dir: '{output_dir}'")
    try:
        download_images(keyword, MAX_NUM)
        consolidate_files(keyword, output_dir)
        detect_and_crop_faces(output_dir)
        logger.info(f"Part 1 finished for keyword: {keyword}")
    except Exception as e:
        logger.error(f"Fatal error in Part 1: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()