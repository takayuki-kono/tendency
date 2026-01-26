import os
import sys
import logging
import cv2
import numpy as np
import uuid
import shutil
from insightface.app import FaceAnalysis

# --- Globals ---
IMG_SIZE = 224
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "log_process_local.txt"), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Safe I/O ---
def validate_image_header(data_bytes):
    """Check against common image file headers."""
    if len(data_bytes) < 12: return False
    # JPEG
    if data_bytes.startswith(b'\xff\xd8\xff'): return True
    # PNG
    if data_bytes.startswith(b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a'): return True
    # BMP
    if data_bytes.startswith(b'BM'): return True
    # WEBP
    if data_bytes.startswith(b'RIFF') and data_bytes[8:12] == b'WEBP': return True
    return False

def imread_safe(path):
    try:
        # 1. File Size Check
        file_size = os.path.getsize(path)
        if file_size > 20 * 1024 * 1024:
            logger.warning(f"File too large ({file_size} bytes): {path}")
            return None
        if file_size < 100:
            return None

        with open(path, "rb") as f:
            bytes_data = f.read()

        # 2. Magic Number Check
        if not validate_image_header(bytes_data):
            logger.warning(f"Invalid file header (not an image): {path}")
            # Optional: Delete invalid file immediately? 
            # might be risky if it's user data, but for automation it's safer.
            # user didn't ask to delete here, just protection.
            return None

        numpy_array = np.frombuffer(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
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
        logger.warning(f"Failed to write {path}: {e}")
        return False

def process_single_file(img_path, rotated_dir, face_app):
    img = imread_safe(img_path)
    if img is None: return False

    try:
        faces = face_app.get(img)
    except Exception as e:
        logger.error(f"Detection error for {img_path}: {e}")
        return False

    if not faces:
        return False

    success = False
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for face in faces:
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
            
            # 2. Re-detect
            new_faces = face_app.get(rotated_img)
            if not new_faces: continue
            target_face = max(new_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            new_lmk = target_face.landmark_2d_106
            
            # 3. Crop (Chin at bottom)
            # Reverted to Landmark 0
            ry_chin = new_lmk[0][1]
            
            ry_top_brow = min(new_lmk[49][1], new_lmk[104][1])
            
            r_size = ry_chin - ry_top_brow
            if r_size < 10: r_size = 10
            
            rx_center = new_lmk[86][0]
            
            rx_min, rx_max = int(rx_center - r_size // 2), int(rx_center + r_size // 2)
            # Anchor max Y to chin
            ry_min, ry_max = int(ry_chin - r_size), int(ry_chin)

            rh, rw = rotated_img.shape[:2]
            pad_t, pad_b = max(0, -ry_min), max(0, ry_max - rh)
            pad_l, pad_r = max(0, -rx_min), max(0, rx_max - rw)
            
            padded_img = cv2.copyMakeBorder(rotated_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
            rx_min += pad_l; rx_max += pad_l
            ry_min += pad_t; ry_max += pad_t

            cropped = padded_img[ry_min:ry_max, rx_min:rx_max]
            if cropped.size == 0: continue
            
            # 4. Resize
            final_resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            
            # 5. Save
            orig_w = cropped.shape[1]
            # Keep original name prefix if possible, or just generate new one
            # Using uuid to avoid collision
            new_filename = f"{base_name}_{uuid.uuid4().hex[:4]}_sz{orig_w}.jpg"
            dest_path = os.path.join(rotated_dir, new_filename)
            
            if imwrite_safe(dest_path, final_resized):
                success = True
        except Exception as e:
            logger.error(f"Processing failed for face in {img_path}: {e}")
            continue

    return success

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_local_raw.py <directory_to_scan>")
        sys.exit(1)

    target_dir = sys.argv[1]
    logger.info(f"Scanning directory: {target_dir}")

    if not os.path.exists(target_dir):
        logger.error("Directory does not exist.")
        sys.exit(1)

    rotated_dir = os.path.join(target_dir, "rotated")
    os.makedirs(rotated_dir, exist_ok=True)

    # Initialize InsightFace
    logger.info("Initializing InsightFace...")
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        logger.error(f"Failed to init InsightFace: {e}")
        sys.exit(1)

    # Identify files to process: Images that do NOT have "_sz" + digits in them
    # Simple heuristic: "sz" not in filename
    files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    raw_files = [f for f in files if "_sz" not in f and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]

    logger.info(f"Found {len(raw_files)} raw images to process.")
    
    processed_count = 0
    deleted_count = 0

    for fname in raw_files:
        path = os.path.join(target_dir, fname)
        if process_single_file(path, rotated_dir, app):
            processed_count += 1
            # Delete original raw file
            try:
                os.remove(path)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Could not delete {path}: {e}")
    
    logger.info(f"Processing complete. Processed {processed_count}, Deleted original {deleted_count}.")

if __name__ == "__main__":
    main()
