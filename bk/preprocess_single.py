import os
import cv2
import logging
import shutil
import argparse
import concurrent.futures
import numpy as np
import random
from collections import defaultdict
from insightface.app import FaceAnalysis
import pickle
import hashlib

# Seed fix
random.seed(42)
np.random.seed(42)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", encoding='utf-8', force=True)
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

# Constants
DEFAULT_TRAIN_DIR = "train"
DEFAULT_VALIDATION_DIR = "validation"
DEFAULT_TEST_DIR = "test"
DEFAULT_PREPRO_DIR = "preprocessed_single"
DEFAULT_THRESH_RATIO = 0.03

# Default Filters
PITCH_FILTER_PERCENTILE = 0
SYMMETRY_FILTER_PERCENTILE = 50
Y_DIFF_FILTER_PERCENTILE = 0
MOUTH_OPEN_FILTER_PERCENTILE = 0
EYEBROW_EYE_PERCENTILE_HIGH = 0 
EYEBROW_EYE_PERCENTILE_LOW = 0  
SHARPNESS_PERCENTILE_LOW = 0
SHARPNESS_PERCENTILE_HIGH = 0   
FACE_SIZE_PERCENTILE_LOW = 0
FACE_SIZE_PERCENTILE_HIGH = 0
RETOUCHING_PERCENTILE = 0
MASK_PERCENTILE = 0
GLASSES_PERCENTILE = 0

# Landmarks (InsightFace 106)
LEFT_INNER_EYE_IDX = 89
RIGHT_INNER_EYE_IDX = 39
UPPER_LIP_CENTER_IDX = 62
LOWER_LIP_CENTER_IDX = 60
RIGHT_EYEBROW_IDX = 49
LEFT_EYEBROW_IDX = 104
RIGHT_EYE_IDX = 40
LEFT_EYE_IDX = 94

# Global for Worker
face_app = None

def get_skin_mask(img):
    """
    肌色領域のマスクを取得
    YCrCb色空間で肌色を検出
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # 肌色の範囲（YCrCb）
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    
    # ノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def calculate_skin_smoothness(img):
    """
    肌領域のスムージング度を計算
    加工画像（美肌加工等）は肌の高周波成分が少ないため低い値になる
    値が高いほど自然な肌、低いほど加工されている可能性が高い
    """
    skin_mask = get_skin_mask(img)
    
    if skin_mask.sum() < 100:
        # 肌領域が十分にない場合はフィルタリング対象外（高い値を返す）
        return float('inf')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ソーベルフィルタで高周波成分を抽出
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 肌領域の高周波成分の平均
    magnitude_masked = magnitude[skin_mask > 0]
    if len(magnitude_masked) > 0:
        return np.mean(magnitude_masked)
    
    return float('inf')

def calculate_mask_likehood(img, lmk):
    """
    マスク装着の可能性をスコア化 (高いほどマスクの可能性大)
    上顔面(額)と下顔面(口周辺)の肌色比率を比較
    """
    h, w = img.shape[:2]
    mask = get_skin_mask(img)
    
    # 簡易的に: 目のY座標より上をUpper、鼻先より下をLowerとする
    eye_y = int((lmk[89][1] + lmk[39][1]) / 2)
    nose_y = int(lmk[86][1])
    
    # Upper ROI
    upper_roi = mask[0:eye_y, :]
    upper_skin_ratio = (upper_roi > 0).sum() / (upper_roi.size + 1e-6)
    
    # Lower ROI
    lower_roi = mask[nose_y:h, :]
    lower_skin_ratio = (lower_roi > 0).sum() / (lower_roi.size + 1e-6)
    
    if upper_skin_ratio < 0.1:
        # 上顔面も肌が見えない -> 全体的に暗いか顔検出がおかしい -> マスク判定は危険（判定しない=0）
        return 0.0
        
    # 比率スコア (大きいほどマスク)
    score = 1.0 - (lower_skin_ratio / (upper_skin_ratio + 1e-6))
    return max(0.0, score)

def calculate_glasses_score(img, lmk):
    """
    眼鏡の可能性をスコア化 (高いほど眼鏡の可能性大)
    目周辺のエッジ量を顔全体のエッジ量と比較
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Eye Region Box
    # L-Eye: 89, R-Eye: 39. Expand to cover frames.
    dist = np.linalg.norm(lmk[89] - lmk[39])
    pad = int(dist * 0.4)
    
    eye_min_x = int(min(lmk[39][0], lmk[89][0])) - pad
    eye_max_x = int(max(lmk[39][0], lmk[89][0])) + pad
    eye_min_y = int(min(lmk[39][1], lmk[89][1])) - int(pad * 0.5)
    eye_max_y = int(max(lmk[39][1], lmk[89][1])) + int(pad * 0.5)
    
    # Clip
    h, w = gray.shape
    eye_min_x = max(0, eye_min_x); eye_max_x = min(w, eye_max_x)
    eye_min_y = max(0, eye_min_y); eye_max_y = min(h, eye_max_y)
    
    eye_roi = gray[eye_min_y:eye_max_y, eye_min_x:eye_max_x]
    if eye_roi.size == 0: return 0.0
    
    # Sobel Edge
    sobel_x = cv2.Sobel(eye_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(eye_roi, cv2.CV_64F, 0, 1, ksize=3)
    mag_eye = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    
    # Compare to Forehead (smooth skin usually)
    fh_min_y = max(0, eye_min_y - pad)
    fh_max_y = eye_min_y
    fh_roi = gray[fh_min_y:fh_max_y, eye_min_x:eye_max_x]
    
    if fh_roi.size > 0:
        sobel_x = cv2.Sobel(fh_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(fh_roi, cv2.CV_64F, 0, 1, ksize=3)
        mag_fh = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
        
        return mag_eye / (mag_fh + 1e-6)
    
    return mag_eye 

def init_worker():
    global face_app
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(320, 320))

def analyze_single_image(args):
    path, label = args
    res = {'path': path, 'label': label, 'valid': False, 'metrics': {}, 'reason': ''}
    
    img = imread_safe(path)
    if img is None:
        res['reason'] = 'read_error'
        return res
        
    h, w = img.shape[:2]
    faces = face_app.get(img)
    if not faces:
        res['reason'] = 'no_face'
        return res
        
    face = faces[0]
    lmk = face.landmark_2d_106
    if lmk is None:
        res['reason'] = 'no_landmarks'
        return res
        
    # Metrics
    pitch = abs(face.pose[0]) if face.pose is not None else 0
    
    lx, ly = lmk[LEFT_INNER_EYE_IDX]
    rx, ry = lmk[RIGHT_INNER_EYE_IDX]
    center_x = w / 2.0
    face_center_x = (lx + rx) / 2.0
    symmetry = abs(face_center_x - center_x)
    
    y_diff = abs(ly - ry)
    
    ul_y = lmk[UPPER_LIP_CENTER_IDX][1]
    ll_y = lmk[LOWER_LIP_CENTER_IDX][1]
    mouth_open = abs(ll_y - ul_y)
    
    rb_y = lmk[RIGHT_EYEBROW_IDX][1]
    re_y = lmk[RIGHT_EYE_IDX][1]
    lb_y = lmk[LEFT_EYEBROW_IDX][1]
    le_y = lmk[LEFT_EYE_IDX][1]
    eb_eye_dist = (abs(rb_y - re_y) + abs(lb_y - le_y)) / 2.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    box = face.bbox
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    aspect_ratio = box_h / (box_w + 1e-6)
    
    # Face Size from filename (e.g., xxx_sz150.jpg -> 150)
    import re
    filename = os.path.basename(path)
    sz_match = re.search(r'_sz(\d+)', filename)
    face_size = int(sz_match.group(1)) if sz_match else 0
    
    # Advanced Metrics
    skin_smoothness = calculate_skin_smoothness(img)
    mask_score = calculate_mask_likehood(img, lmk)
    glasses_score = calculate_glasses_score(img, lmk)
    
    res['valid'] = True
    res['metrics'] = {
        'pitch': pitch,
        'symmetry': symmetry,
        'y_diff': y_diff,
        'mouth_open': mouth_open,
        'eb_eye_dist': eb_eye_dist,
        'sharpness': sharpness,
        'aspect_ratio': aspect_ratio,
        'face_size': face_size,
        'skin_smoothness': skin_smoothness,
        'mask_score': mask_score,
        'glasses_score': glasses_score
    }
    return res

def copy_worker(args):
    src, dst, grayscale = args
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if grayscale:
            img = imread_safe(src)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ext = os.path.splitext(dst)[1]
                result, encoded = cv2.imencode(ext, gray)
                if result:
                    encoded.tofile(dst)
                else:
                    return False
            else:
                return False
        else:
            shutil.copyfile(src, dst)
        return True
    except Exception:
        return False

def process_dataset(src_root, dst_root, args, skip_undersampling=False):
    if not os.path.exists(src_root):
        logger.warning(f"Source dir not found: {src_root}")
        return 0, 0, 0

    logger.info(f"Scanning {src_root}...")
    files = []
    # Identify PERSON as the label (sub-subdirectory)
    # Structure: train/label/person_name/image.jpg
    for root, dirs, filenames in os.walk(src_root):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                # Determine Person Name
                rel = os.path.relpath(root, src_root)
                parts = rel.split(os.sep)
                
                # We need at least Label and Person
                if len(parts) >= 2:
                    person_name = parts[-1] 
                    label_key = person_name 
                    files.append((os.path.join(root, f), label_key))
                else:
                    files.append((os.path.join(root, f), "unknown"))
    
    total = len(files)
    
    # Cache Logic
    cache_dir = os.path.join("outputs", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{os.path.basename(src_root)}_person_{total}_multitask_update"
    cache_file = os.path.join(cache_dir, f"metrics_{hashlib.md5(cache_key.encode()).hexdigest()}.pkl")
    
    results = []
    max_workers = max(1, os.cpu_count() // 2)

    if os.path.exists(cache_file):
        logger.info(f"Loading analysis results from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            logger.info("Cache loaded successfully.")
        except Exception as e:
            results = []
            
    if not results:
        logger.info(f"Analyzing {total} images...")
        
        init_worker()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for r in executor.map(analyze_single_image, files):
                results.append(r)
                if len(results) % 100 == 0:
                     logger.debug(f"Analyzed {len(results)}/{total}")

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e: pass

    valid_items = [r for r in results if r['valid']]
    logger.info(f"Valid faces detected: {len(valid_items)}/{total}")
    
    if not valid_items: return total, 0, total

    # --- Global Threshold Calculation ---
    def get_th(key, pct):
        vals = [r['metrics'][key] for r in valid_items]
        if not vals: return 0
        return np.percentile(vals, 100 - pct)
    
    def get_th_low(key, pct):
        vals = [r['metrics'][key] for r in valid_items]
        if not vals: return 0
        return np.percentile(vals, pct)
    
    th_pitch = get_th('pitch', args.pitch_percentile)
    th_sym = get_th('symmetry', args.symmetry_percentile)
    th_y = get_th('y_diff', args.y_diff_percentile)
    th_mouth = get_th('mouth_open', args.mouth_open_percentile)
    
    th_sharpness_low = 0
    if args.sharpness_percentile_low > 0:
        th_sharpness_low = get_th_low('sharpness', args.sharpness_percentile_low)
        
    th_sharpness_high = 999999.0
    if args.sharpness_percentile_high > 0:
        th_sharpness_high = get_th('sharpness', args.sharpness_percentile_high)

    th_face_size_low = 0
    th_face_size_high = 999999
    face_size_vals = [r['metrics'].get('face_size', 0) for r in valid_items if r['metrics'].get('face_size', 0) > 0]
    if face_size_vals:
        if args.face_size_percentile_low > 0:
            th_face_size_low = np.percentile(face_size_vals, args.face_size_percentile_low)
        if args.face_size_percentile_high > 0:
            th_face_size_high = np.percentile(face_size_vals, 100 - args.face_size_percentile_high)
            
    th_retouching = 0
    if args.retouching_percentile > 0:
        retouch_vals = [r['metrics'].get('skin_smoothness', float('inf')) for r in valid_items]
        retouch_vals = [v for v in retouch_vals if v != float('inf')]
        if retouch_vals:
            th_retouching = np.percentile(retouch_vals, args.retouching_percentile)
            
    th_mask = 999.0
    if args.mask_percentile > 0:
        mask_vals = [r['metrics'].get('mask_score', 0) for r in valid_items]
        if mask_vals:
            th_mask = np.percentile(mask_vals, 100 - args.mask_percentile)

    th_glasses = 999.0
    if args.glasses_percentile > 0:
        gl_vals = [r['metrics'].get('glasses_score', 0) for r in valid_items]
        if gl_vals:
             th_glasses = np.percentile(gl_vals, 100 - args.glasses_percentile)

    # --- Grouping by Person ---
    grouped = defaultdict(list)
    for r in valid_items:
        grouped[r['label']].append(r)
        
    final_copy_tasks = []
    skipped_count = 0
    skip_reasons = defaultdict(int)

    # --- Undersampling per Person ---
    counts = [len(items) for items in grouped.values()]
    target_count = int(np.mean(counts)) if counts else 0
    logger.info(f"Undersampling target count per person (mean): {target_count}")
    
    for label, items in grouped.items():
        eb_vals = [r['metrics']['eb_eye_dist'] for r in items]
        th_eb_high = 999.0
        th_eb_low = -1.0
        if eb_vals:
            if args.eyebrow_eye_percentile_high > 0:
                th_eb_high = np.percentile(eb_vals, 100 - args.eyebrow_eye_percentile_high)
            if args.eyebrow_eye_percentile_low > 0:
                th_eb_low = np.percentile(eb_vals, args.eyebrow_eye_percentile_low)
        
        label_valid_tasks = []
        for item in items:
            m = item['metrics']
            reason = None
            
            if args.pitch_percentile > 0 and m['pitch'] > th_pitch: reason = 'pitch_global'
            elif args.symmetry_percentile > 0 and m['symmetry'] > th_sym: reason = 'symmetry_global'
            elif args.y_diff_percentile > 0 and m['y_diff'] > th_y: reason = 'y_diff_global'
            elif args.mouth_open_percentile > 0 and m['mouth_open'] > th_mouth: reason = 'mouth_open_global'
            elif args.sharpness_percentile_low > 0 and m['sharpness'] < th_sharpness_low: reason = 'sharpness_low_global'
            elif args.sharpness_percentile_high > 0 and m['sharpness'] > th_sharpness_high: reason = 'sharpness_high_global'
            elif args.face_size_percentile_low > 0 and m.get('face_size', 0) > 0 and m.get('face_size', 0) < th_face_size_low: reason = 'face_size_low_global'
            elif args.face_size_percentile_high > 0 and m.get('face_size', 0) > 0 and m.get('face_size', 0) > th_face_size_high: reason = 'face_size_high_global'
            elif args.retouching_percentile > 0 and m.get('skin_smoothness', float('inf')) != float('inf') and m.get('skin_smoothness', float('inf')) < th_retouching: reason = 'retouching_global'
            elif args.mask_percentile > 0 and m.get('mask_score', 0) > th_mask: reason = 'mask_global'
            elif args.glasses_percentile > 0 and m.get('glasses_score', 0) > th_glasses: reason = 'glasses_global'
            
            elif (args.eyebrow_eye_percentile_high > 0 and m['eb_eye_dist'] > th_eb_high): reason = 'eb_eye_high_personal'
            elif (args.eyebrow_eye_percentile_low > 0 and m['eb_eye_dist'] < th_eb_low): reason = 'eb_eye_low_personal'
            
            if reason:
                skipped_count += 1
                skip_reasons[reason] += 1
            else:
                label_valid_tasks.append(item)

        import random
        random.shuffle(label_valid_tasks)
        
        count_before_cut = len(label_valid_tasks)
        if not skip_undersampling and count_before_cut > target_count:
            label_valid_tasks = label_valid_tasks[:target_count]
            skipped_count += (count_before_cut - target_count)
            skip_reasons['undersampling'] += (count_before_cut - target_count)

        for item in label_valid_tasks:
            src = item['path']
            # Preserve Structure: copy relpath
            rel = os.path.relpath(src, src_root)
            dst = os.path.join(dst_root, rel)
            final_copy_tasks.append((src, dst, args.grayscale))

    skipped_count += (total - len(valid_items))
    saved_count = len(final_copy_tasks)
    
    logger.info(f"Copying {saved_count} images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers*2) as executor:
        list(executor.map(copy_worker, final_copy_tasks))

    logger.info(f"Processed {src_root}: Total={total}, Saved={saved_count}, Skipped={skipped_count}")
    return total, saved_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description="Person-based Preprocessing with Face Filtering")
    parser.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val_dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--test_dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_PREPRO_DIR)
    
    parser.add_argument("--pitch_percentile", type=int, default=PITCH_FILTER_PERCENTILE)
    parser.add_argument("--symmetry_percentile", type=int, default=SYMMETRY_FILTER_PERCENTILE)
    parser.add_argument("--y_diff_percentile", type=int, default=Y_DIFF_FILTER_PERCENTILE)
    parser.add_argument("--mouth_open_percentile", type=int, default=MOUTH_OPEN_FILTER_PERCENTILE)
    parser.add_argument("--eyebrow_eye_percentile_high", type=int, default=EYEBROW_EYE_PERCENTILE_HIGH)
    parser.add_argument("--eyebrow_eye_percentile_low", type=int, default=EYEBROW_EYE_PERCENTILE_LOW)
    parser.add_argument("--sharpness_percentile_low", type=int, default=SHARPNESS_PERCENTILE_LOW)
    parser.add_argument("--sharpness_percentile_high", type=int, default=SHARPNESS_PERCENTILE_HIGH)
    
    # New filters
    parser.add_argument("--face_size_percentile_low", type=int, default=FACE_SIZE_PERCENTILE_LOW)
    parser.add_argument("--face_size_percentile_high", type=int, default=FACE_SIZE_PERCENTILE_HIGH)
    parser.add_argument("--retouching_percentile", type=int, default=RETOUCHING_PERCENTILE)
    parser.add_argument("--mask_percentile", type=int, default=MASK_PERCENTILE)
    parser.add_argument("--glasses_percentile", type=int, default=GLASSES_PERCENTILE)
    
    parser.add_argument("--grayscale", action="store_true")
    
    args = parser.parse_args()
    
    prepro_dir = args.out_dir
    prepro_train = os.path.join(prepro_dir, "train")
    prepro_valid = os.path.join(prepro_dir, "validation")
    prepro_test = os.path.join(prepro_dir, "test")
    
    try:
        if os.path.exists(prepro_dir):
            shutil.rmtree(prepro_dir)

        # Ensure directories exist (even if empty)
        os.makedirs(prepro_train, exist_ok=True)
        os.makedirs(prepro_valid, exist_ok=True)
        # os.makedirs(prepro_test, exist_ok=True) # Optional if test dir exists

        logger.info("Starting Person Preprocessing (Updated with new filters)...")
        process_dataset(args.train_dir, prepro_train, args)
        process_dataset(args.val_dir, prepro_valid, args, skip_undersampling=True) # Usually don't undersample validation
        
        logger.info("All processing complete.")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
