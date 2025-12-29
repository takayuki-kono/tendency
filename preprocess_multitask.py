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
DEFAULT_PREPRO_DIR = "preprocessed_multitask"
DEFAULT_THRESH_RATIO = 0.03

# Default Filters
PITCH_FILTER_PERCENTILE = 0
SYMMETRY_FILTER_PERCENTILE = 0
Y_DIFF_FILTER_PERCENTILE = 0
MOUTH_OPEN_FILTER_PERCENTILE = 0
EYEBROW_EYE_PERCENTILE_HIGH = 0 
EYEBROW_EYE_PERCENTILE_LOW = 0  
SHARPNESS_PERCENTILE_LOW = 5   # Filter bottom X% by sharpness (Laplacian variance)

FACE_POSITION_FILTER_ENABLED = True

# Landmarks (InsightFace 106)
LEFT_CHEEK_IDX = 28
RIGHT_CHEEK_IDX = 12
UPPER_LIP_CENTER_IDX = 62
LOWER_LIP_CENTER_IDX = 60

# Eyebrow and Eye for distance calc
RIGHT_EYEBROW_IDX = 49
LEFT_EYEBROW_IDX = 104
RIGHT_EYE_IDX = 40
LEFT_EYE_IDX = 94

# Global for Worker
face_app = None
face_pos_enabled = True

def init_worker(fpe):
    global face_app, face_pos_enabled
    face_pos_enabled = fpe
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
        
    # Pitch
    pitch = 0
    if face.pose is not None:
        pitch = abs(face.pose[0])
    
    # Symmetry
    lx, ly = lmk[LEFT_CHEEK_IDX]
    rx, ry = lmk[RIGHT_CHEEK_IDX]
    center_x = w / 2.0
    d_left = lx - center_x
    d_right = center_x - rx
    
    if face_pos_enabled and (d_left <= 0 or d_right <= 0):
        res['reason'] = f'face_pos_invalid'
        return res
        
    symmetry = abs(d_left / d_right - 1)
    
    # Y Diff
    y_diff = abs(ly - ry) / h
    
    # Mouth Open
    ul_y = lmk[UPPER_LIP_CENTER_IDX][1]
    ll_y = lmk[LOWER_LIP_CENTER_IDX][1]
    mouth_open = abs(ll_y - ul_y) / h
    
    # Eyebrow Eye Dist
    # R: 49(brow) - 40(eye)
    # L: 104(brow) - 94(eye)
    rb_y = lmk[RIGHT_EYEBROW_IDX][1]
    re_y = lmk[RIGHT_EYE_IDX][1]
    lb_y = lmk[LEFT_EYEBROW_IDX][1]
    le_y = lmk[LEFT_EYE_IDX][1]
    
    dist_r = abs(rb_y - re_y)
    dist_l = abs(lb_y - le_y)
    eb_eye_dist = (dist_r + dist_l) / 2.0 / h
    
    # Sharpness (Laplacian Variance) - higher = sharper
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    res['valid'] = True
    res['metrics'] = {
        'pitch': pitch,
        'symmetry': symmetry,
        'y_diff': y_diff,
        'mouth_open': mouth_open,
        'eb_eye_dist': eb_eye_dist,
        'sharpness': sharpness
    }
    return res

def copy_worker(args):
    src, dst = args
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return True
    except Exception:
        return False

def calculate_thresholds(*args, **kwargs):
    # Dummy function for compatibility if imported
    logger.warning("calculate_thresholds is deprecated and does nothing.")
    return 0,0,0,0

def process_dataset(src_root, dst_root, args):
    if not os.path.exists(src_root):
        logger.warning(f"Source dir not found: {src_root}")
        return 0, 0, 0

    logger.info(f"Scanning {src_root}...")
    files = []
    for root, dirs, filenames in os.walk(src_root):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                rel = os.path.relpath(root, src_root)
                label = rel.split(os.sep)[0]
                if label == '.': label = 'root'
                files.append((os.path.join(root, f), label))
    
    total = len(files)
    
    # Caching Logic
    import pickle
    import hashlib
    
    # Calculate hash based on file paths and modification times (simple consistency check)
    # Using simple length + first/last file name for speed, or just simple param.
    # To be safe, let's use src_root name and total count.
    # Ideally we should hash all filenames, but that's slow.
    # Let's assume if file count is same, it's the same dataset for optimization loop context.
    cache_dir = os.path.join("outputs", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Include face_pos_filter arg in cache key because it affects analysis result (valid bit)
    cache_key = f"{os.path.basename(src_root)}_{total}_fp={args.face_pos_filter}"
    cache_file = os.path.join(cache_dir, f"metrics_{hashlib.md5(cache_key.encode()).hexdigest()}.pkl")
    
    results = []
    
    # Adjust workers safely
    max_workers = max(1, os.cpu_count() // 2)

    if os.path.exists(cache_file):
        logger.info(f"Loading analysis results from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            logger.info("Cache loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-analyzing...")
            results = []
            
    if not results:
        logger.info(f"Analyzing {total} images...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(args.face_pos_filter,)) as executor:
            for r in executor.map(analyze_single_image, files):
                results.append(r)
                if len(results) % 100 == 0:
                    logger.debug(f"Analyzed {len(results)}/{total}")
        
        # Save cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Saved analysis cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    valid_items = [r for r in results if r['valid']]
    logger.info(f"Valid faces detected: {len(valid_items)}/{total}")
    
    if not valid_items:
        return total, 0, total # All skipped

    # --- Global Threshold Calculation ---
    def get_th(key, pct):
        vals = [r['metrics'][key] for r in valid_items]
        if not vals: return 0
        return np.percentile(vals, 100 - pct)
    
    th_pitch = get_th('pitch', args.pitch_percentile)
    th_sym = get_th('symmetry', args.symmetry_percentile)
    th_y = get_th('y_diff', args.y_diff_percentile)
    th_mouth = get_th('mouth_open', args.mouth_open_percentile)
    
    # Sharpness threshold (lower bound - filter blurry images)
    th_sharpness_low = 0
    if args.sharpness_percentile_low > 0:
        sharp_vals = [r['metrics']['sharpness'] for r in valid_items]
        th_sharpness_low = np.percentile(sharp_vals, args.sharpness_percentile_low)
    
    logger.info(f"Global Thresh: Pitch>={th_pitch:.2f}, Sym>={th_sym:.3f}, YDiff>={th_y:.4f}, Mouth>={th_mouth:.4f}, Sharpness<={th_sharpness_low:.1f}")
    
    # --- Filtering & Grouping ---
    grouped = defaultdict(list)
    for r in valid_items:
        grouped[r['label']].append(r)
        
    final_copy_tasks = []
    skipped_count = 0
    skip_reasons = defaultdict(int)

    # --- Undersampling Logic ---
    # Calculate target count per label (using Mean)
    counts = [len(items) for items in grouped.values()]
    target_count = int(np.mean(counts)) if counts else 0
    logger.info(f"Undersampling target count (mean): {target_count}")
    
    for label, items in grouped.items():
        # Personal Thresholds for Eyebrow-Eye Dist
        eb_vals = [r['metrics']['eb_eye_dist'] for r in items]
        th_eb_high = 999.0
        th_eb_low = -1.0
        
        if eb_vals:
            if args.eyebrow_eye_percentile_high > 0:
                th_eb_high = np.percentile(eb_vals, 100 - args.eyebrow_eye_percentile_high)
            if args.eyebrow_eye_percentile_low > 0:
                th_eb_low = np.percentile(eb_vals, args.eyebrow_eye_percentile_low)
        
        # Filter valid items first
        label_valid_tasks = []
        for item in items:
            m = item['metrics']
            reason = None
            
            # Global Checks
            if args.pitch_percentile > 0 and m['pitch'] > th_pitch: reason = 'pitch_global'
            elif args.symmetry_percentile > 0 and m['symmetry'] > th_sym: reason = 'symmetry_global'
            elif args.y_diff_percentile > 0 and m['y_diff'] > th_y: reason = 'y_diff_global'
            elif args.mouth_open_percentile > 0 and m['mouth_open'] > th_mouth: reason = 'mouth_open_global'
            elif args.sharpness_percentile_low > 0 and m['sharpness'] < th_sharpness_low: reason = 'sharpness_low_global'
            
            # Personal Check
            elif (args.eyebrow_eye_percentile_high > 0 and m['eb_eye_dist'] > th_eb_high): reason = 'eb_eye_high_personal'
            elif (args.eyebrow_eye_percentile_low > 0 and m['eb_eye_dist'] < th_eb_low): reason = 'eb_eye_low_personal'
            
            if reason:
                skipped_count += 1
                skip_reasons[reason] += 1
            else:
                label_valid_tasks.append(item)

        # Apply Undersampling Limitation
        import random
        random.shuffle(label_valid_tasks) # Randomize for bias reduction
        
        count_before_cut = len(label_valid_tasks)
        if count_before_cut > target_count:
            # Cut down to target
            label_valid_tasks = label_valid_tasks[:target_count]
            skipped_count += (count_before_cut - target_count)
            skip_reasons['undersampling'] += (count_before_cut - target_count)

        # Create copy tasks
        for item in label_valid_tasks:
            src = item['path']
            rel = os.path.relpath(src, src_root)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                new_filename = "_".join(parts[1:])
                dst = os.path.join(dst_root, parts[0], new_filename)
            else:
                dst = os.path.join(dst_root, rel)
            
            final_copy_tasks.append((src, dst))

    # Add invalid/read_error counts to skipped
    skipped_count += (total - len(valid_items))
    saved_count = len(final_copy_tasks)
    
    logger.info(f"Copying {saved_count} images...")
    
    # --- Execute Copy ---
    # Using ThreadPool is sufficient for file copy usually, but ProcessPool is fine too.
    # We can reuse ProcessPoolExecutor (lighter init) or just ThreadPool.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers*2) as executor:
        list(executor.map(copy_worker, final_copy_tasks))

    logger.info(f"Processed {src_root}: Total={total}, Saved={saved_count}, Skipped={skipped_count}")
    if skip_reasons:
        logger.info(f"Skip Reasons: {dict(skip_reasons)}")
    return total, saved_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description="Multitask Preprocessing with Face Filtering")
    parser.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR, help="Source training directory")
    parser.add_argument("--val_dir", default=DEFAULT_VALIDATION_DIR, help="Source validation directory")
    parser.add_argument("--test_dir", default=DEFAULT_TEST_DIR, help="Source test directory")
    parser.add_argument("--out_dir", default=DEFAULT_PREPRO_DIR, help="Output directory")
    
    # Original Args
    parser.add_argument("--thresh", type=float, default=DEFAULT_THRESH_RATIO, help="Threshold ratio (unused in new logic but kept for compat)")
    parser.add_argument("--pitch_percentile", type=int, default=PITCH_FILTER_PERCENTILE, help="Pitch filter percentile (0-100)")
    parser.add_argument("--symmetry_percentile", type=int, default=SYMMETRY_FILTER_PERCENTILE, help="Symmetry filter percentile (0-100)")
    parser.add_argument("--y_diff_percentile", type=int, default=Y_DIFF_FILTER_PERCENTILE, help="Y-diff filter percentile (0-100)")
    parser.add_argument("--mouth_open_percentile", type=int, default=MOUTH_OPEN_FILTER_PERCENTILE, help="Mouth open filter percentile (0-100)")
    parser.add_argument("--face_position_filter", type=str, default=str(FACE_POSITION_FILTER_ENABLED), help="Enable face position filter (True/False)")
    
    # New Args
    parser.add_argument("--eyebrow_eye_percentile_high", type=int, default=EYEBROW_EYE_PERCENTILE_HIGH, help="Filter top X% of eyebrow-eye distance")
    parser.add_argument("--eyebrow_eye_percentile_low", type=int, default=EYEBROW_EYE_PERCENTILE_LOW, help="Filter bottom X% of eyebrow-eye distance")
    parser.add_argument("--sharpness_percentile_low", type=int, default=SHARPNESS_PERCENTILE_LOW, help="Filter bottom X% by sharpness (blurry images)")
    
    args = parser.parse_args()
    
    # Parse boolean
    args.face_pos_filter = (args.face_position_filter.lower() == 'true')
    
    prepro_dir = args.out_dir
    prepro_train = os.path.join(prepro_dir, "train")
    prepro_valid = os.path.join(prepro_dir, "validation")
    prepro_test = os.path.join(prepro_dir, "test")

    try:
        if os.path.exists(prepro_dir):
            logger.info(f"Deleting existing '{prepro_dir}' folder...")
            shutil.rmtree(prepro_dir)
            logger.info(f"Deleted '{prepro_dir}' folder.")

        logger.info("=" * 60)
        logger.info("Starting preprocessing (New Logic)...")
        logger.info(f"  Pitch Pct: {args.pitch_percentile}")
        logger.info(f"  Sym Pct: {args.symmetry_percentile}")
        logger.info(f"  Y-Diff Pct: {args.y_diff_percentile}")
        logger.info(f"  Mouth Pct: {args.mouth_open_percentile}")
        logger.info(f"  Eb-Eye Pct (High/Low): {args.eyebrow_eye_percentile_high} / {args.eyebrow_eye_percentile_low} (PER PERSON)")
        logger.info(f"  Sharpness Pct Low: {args.sharpness_percentile_low}")
        logger.info("=" * 60)
        
        process_dataset(args.train_dir, prepro_train, args)
        process_dataset(args.val_dir, prepro_valid, args)
        if os.path.exists(args.test_dir):
            process_dataset(args.test_dir, prepro_test, args)
        
        logger.info("All processing complete.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    # Windows specific fix
    # multiprocessing.freeze_support() 
    main()