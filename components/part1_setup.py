# Part 1: Setup and InsightFace Processing (imagedl + InsightFace)
import os
import sys
import shutil
import logging
import uuid
import cv2
import numpy as np
from imagedl import imagedl
from insightface.app import FaceAnalysis

# --- Globals ---
TARGET_COUNT_PER_ENGINE = 1000
IMG_SIZE = 224
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 使用するエンジン
ENGINES = [
    'YahooImageClient',
    'BingImageClient',
    'BaiduImageClient',
    'GoogleImageClient'
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "log_part1_v3.txt"), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def imread_safe(path):
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        return img
    except:
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
    except:
        return False

def process_and_save_face(img_path, rotated_dir, face_app, engine_name):
    """画像から顔を検出し、加工して rotated_dir に保存する"""
    img = imread_safe(img_path)
    if img is None: return 0

    try:
        faces = face_app.get(img)
    except:
        return 0

    if not faces: return 0

    saved_count = 0
    for face_idx, face in enumerate(faces):
        try:
            # 1. Rotation Alignment
            lmk = face.landmark_2d_106
            if lmk is None: continue
            
            dx = lmk[86][0] - lmk[0][0]
            dy = lmk[86][1] - lmk[0][1]
            angle = np.arctan2(dx, -dy) * 180 / np.pi
            
            center = (img.shape[1] / 2, img.shape[0] / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            
            # 2. Re-detect on rotated image
            new_faces = face_app.get(rotated_img)
            if not new_faces: continue
            target_face = max(new_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            new_lmk = target_face.landmark_2d_106
            
            # 3. Eyebrow-to-Chin Crop
            ry_top_brow = min(new_lmk[49][1], new_lmk[104][1])
            ry_bottom = new_lmk[0][1] # Chin
            rx_center = new_lmk[86][0]
            r_size = max(ry_bottom - ry_top_brow, rotated_img.shape[1] // 4)
            
            rx_min, rx_max = int(rx_center - r_size // 2), int(rx_center + r_size // 2)
            ry_min, ry_max = int(ry_top_brow), int(ry_top_brow + r_size)

            # Padding (Allow all padding)
            rh, rw = rotated_img.shape[:2]
            pad_t, pad_b = max(0, -ry_min), max(0, ry_max - rh)
            pad_l, pad_r = max(0, -rx_min), max(0, rx_max - rw)
            
            padded_img = cv2.copyMakeBorder(rotated_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
            rx_min += pad_l; rx_max += pad_l
            ry_min += pad_t; ry_max += pad_t

            cropped = padded_img[ry_min:ry_max, rx_min:rx_max]
            if cropped.size == 0: continue
            
            # 4. Resize to 224x224
            final_resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            
            # 5. Save with size info
            orig_w = cropped.shape[1]
            new_filename = f"{engine_name}_{uuid.uuid4().hex[:8]}_sz{orig_w}.jpg"
            dest_path = os.path.join(rotated_dir, new_filename)
            
            if imwrite_safe(dest_path, final_resized):
                saved_count += 1
        except:
            continue
    return saved_count

def main():
    if len(sys.argv) != 3:
        print("Usage: python part1_setup.py <keyword> <output_dir>")
        sys.exit(1)
    
    keyword = sys.argv[1]
    output_dir = sys.argv[2] # e.g. master_data/奈緒
    rotated_dir = os.path.join(output_dir, "rotated")
    os.makedirs(rotated_dir, exist_ok=True)

    # Initialize InsightFace
    logger.info("Initializing InsightFace...")
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Search Variations
    search_keywords = [f"{keyword} 女優"]

    temp_root = os.path.abspath('temp_part1_raw')
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)

    total_faces = 0

    for engine_name in ENGINES:
        logger.info(f"--- Engine: {engine_name} ---")
        for skey in search_keywords:
            try:
                client = imagedl.ImageClient(
                    image_source=engine_name,
                    init_image_client_cfg={'work_dir': temp_root, 'max_retries': 3},
                    search_limits=TARGET_COUNT_PER_ENGINE,
                    num_threadings=15
                )
                
                image_infos = client.search(skey, filters={'size': 'large'})
                if not image_infos: continue
                
                downloaded = client.download(image_infos)
                if downloaded:
                    logger.info(f"Processing {len(downloaded)} images from {engine_name} for '{skey}'")
                    for info in downloaded:
                        src_path = info['file_path']
                        if os.path.exists(src_path):
                            total_faces += process_and_save_face(src_path, rotated_dir, face_app, engine_name)
                            os.remove(src_path)
                
                # Cleanup engine temp folder
                if downloaded:
                    shutil.rmtree(downloaded[0]['work_dir'], ignore_errors=True)

            except Exception as e:
                logger.error(f"Error in engine {engine_name}: {e}")

    if os.path.exists(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)

    logger.info(f"Part 1 finished. Total faces extracted: {total_faces}")

if __name__ == "__main__":
    main()