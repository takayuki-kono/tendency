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
SHARPNESS_PERCENTILE_LOW = 0  # Filter bottom X% by sharpness (Laplacian variance)
SHARPNESS_PERCENTILE_HIGH = 0  # Filter top X% by sharpness
FACE_SIZE_PERCENTILE_LOW = 0   # Filter bottom X% by face size (small images)
FACE_SIZE_PERCENTILE_HIGH = 0  # Filter top X% by face size (large images)
RETOUCHING_PERCENTILE = 0  # Filter bottom X% by skin smoothness (retouched images have lower values)
MASK_PERCENTILE = 0 # Filter top X% by mask likelihood (high score = mask)
GLASSES_PERCENTILE = 0 # Filter top X% by glasses likelihood (high score = glasses)

# Landmarks (InsightFace 106)
LEFT_INNER_EYE_IDX = 89
RIGHT_INNER_EYE_IDX = 39
UPPER_LIP_CENTER_IDX = 62
LOWER_LIP_CENTER_IDX = 60

# Eyebrow and Eye for distance calc
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
    
    # Upper Face (Forehead): Above eyebrows
    # LMK 72(L-Brow), 33(R-Brow) approx.
    # Use average Y of eyebrows upwards
    # InsightFace 106: Brows 33-42 (R), 64-73 (L)
    # Let's use Eye Center to Top
    # L: 89, R: 39
    
    # 簡易的に: 目のY座標より上をUpper、鼻先より下をLowerとする
    eye_y = int((lmk[89][1] + lmk[39][1]) / 2)
    nose_y = int(lmk[86][1])
    
    # Upper ROI
    upper_roi = mask[0:eye_y, :]
    upper_skin_ratio = (upper_roi > 0).sum() / (upper_roi.size + 1e-6)
    
    # Lower ROI
    lower_roi = mask[nose_y:h, :]
    lower_skin_ratio = (lower_roi > 0).sum() / (lower_roi.size + 1e-6)
    
    # もし上顔面がしっかり肌色であれば、下顔面の肌色率が低い＝マスクの可能性
    # Upperが肌色でない(前髪等)場合は信頼度低いが、とりあえず比率で見る
    
    # Score: 1.0 - (Lower / (Upper + epsilon))
    # Upperが0の場合は考慮して、単純に Lower Ratio が低いかどうかを見るだけでも良いが、
    # 暗い画像対策で相対評価にする
    
    if upper_skin_ratio < 0.1:
        # 上顔面も肌が見えない -> 全体的に暗いか顔検出がおかしい -> マスク判定は危険（判定しない=0）
        return 0.0
        
    # 比率スコア (大きいほどマスク)
    score = 1.0 - (lower_skin_ratio / (upper_skin_ratio + 1e-6))
    return max(0.0, score)

def calculate_glasses_score(img, lmk):
    """
    眼鏡の可能性をスコア化 (高いほど眼鏡の可能性大)
    目周辺のエッジ量を顔全体のエッジ量と比較、または絶対値
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Face Box (lmk min/max)
    min_x = int(np.min(lmk[:, 0]))
    max_x = int(np.max(lmk[:, 0]))
    min_y = int(np.min(lmk[:, 1]))
    max_y = int(np.max(lmk[:, 1]))
    
    # Eye Region Box
    # L-Eye: 89, R-Eye: 39. Expand to cover frames.
    # Dist between eyes
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
    # Forehead: Above eyes
    fh_min_y = max(0, eye_min_y - pad)
    fh_max_y = eye_min_y
    fh_roi = gray[fh_min_y:fh_max_y, eye_min_x:eye_max_x]
    
    if fh_roi.size > 0:
        sobel_x = cv2.Sobel(fh_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(fh_roi, cv2.CV_64F, 0, 1, ksize=3)
        mag_fh = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
        
        # Raito: Eye Edge / Forehead Edge
        # Glasses add edges to eye region
        return mag_eye / (mag_fh + 1e-6)
    
    return mag_eye # Fallback (Absolute value)

def init_worker():
    global face_app
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(320, 320))

def analyze_single_image(args):
    path, label = args
    res = {'path': path, 'label': label, 'valid': False, 'metrics': {}, 'reason': ''}
    
    # ... logic continues ...
    
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
    
    # Symmetry (Face Center Offset)
    lx, ly = lmk[LEFT_INNER_EYE_IDX]
    rx, ry = lmk[RIGHT_INNER_EYE_IDX]
    center_x = w / 2.0
    
    # Calculate offset of face center (midpoint of eyes) from image center
    face_center_x = (lx + rx) / 2.0
    symmetry = abs(face_center_x - center_x)
    
    # Face Position filter removed as requested.
    # Symmetry now acts as center offset filter.
    
    # Y Diff (Raw Pixel)
    y_diff = abs(ly - ry)
    
    # Mouth Open (Raw Pixel)
    ul_y = lmk[UPPER_LIP_CENTER_IDX][1]
    ll_y = lmk[LOWER_LIP_CENTER_IDX][1]
    mouth_open = abs(ll_y - ul_y)
    
    # Eyebrow Eye Dist (Raw Pixel Average)
    # R: 49(brow) - 40(eye)
    # L: 104(brow) - 94(eye)
    rb_y = lmk[RIGHT_EYEBROW_IDX][1]
    re_y = lmk[RIGHT_EYE_IDX][1]
    lb_y = lmk[LEFT_EYEBROW_IDX][1]
    le_y = lmk[LEFT_EYE_IDX][1]
    
    dist_r = abs(rb_y - re_y)
    dist_l = abs(lb_y - le_y)
    eb_eye_dist = (dist_r + dist_l) / 2.0
    
    # Sharpness (Laplacian Variance) - higher = sharper
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Skin Smoothness for Retouching Detection
    # 肌領域のスムージング度を計算（加工画像は低い値になる）
    skin_smoothness = calculate_skin_smoothness(img)

    # Aspect Ratio (Height / Width)
    box = face.bbox
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    aspect_ratio = box_h / (box_w + 1e-6)
    
    # Face Size from filename (e.g., xxx_sz150.jpg -> 150)
    import re
    filename = os.path.basename(path)
    sz_match = re.search(r'_sz(\d+)', filename)
    face_size = int(sz_match.group(1)) if sz_match else 0
    
    res['valid'] = True
    res['metrics'] = {
        'pitch': pitch,
        'symmetry': symmetry,
        'y_diff': y_diff,
        'mouth_open': mouth_open,
        'eb_eye_dist': eb_eye_dist,
        'sharpness': sharpness,
        'skin_smoothness': skin_smoothness,
        'aspect_ratio': aspect_ratio,
        'face_size': face_size
    }
    # Mask & Glasses
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
        'skin_smoothness': skin_smoothness,
        'aspect_ratio': aspect_ratio,
        'face_size': face_size,
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
                # 日本語パス対応: cv2.imwriteの代わりにimencodeとtofileを使用
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

def calculate_thresholds(*args, **kwargs):
    # Dummy function for compatibility if imported
    logger.warning("calculate_thresholds is deprecated and does nothing.")
    return 0,0,0,0

def process_dataset(src_root, dst_root, args, skip_undersampling=False):
    if not os.path.exists(src_root):
        logger.warning(f"Source dir not found: {src_root}")
        return 0, 0, 0

    logger.info(f"Scanning {src_root}...")
    files = []
    for root, dirs, filenames in os.walk(src_root):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                rel_dir = os.path.relpath(root, src_root)
                if rel_dir == '.': rel_dir = 'root'
                else: rel_dir = rel_dir.replace(os.sep, '/')
                # グループ化キー = ディレクトリパス（タスク/個人など）。個人単位で percentile / undersampling するため。
                files.append((os.path.join(root, f), rel_dir))
    
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
    cache_key = f"{os.path.basename(src_root)}_{total}"
    cache_file = os.path.join(cache_dir, f"metrics_{hashlib.md5(cache_key.encode()).hexdigest()}.pkl")
    
    results = []
    
    # Adjust workers safely
    max_workers = max(1, os.cpu_count() // 2)

    if os.path.exists(cache_file):
        logger.info(f"Loading analysis results from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            # キャッシュが古いコード（label=タスクのみ）で保存されていても、パスから個人単位のlabelに上書きする
            for r in results:
                rel_dir = os.path.relpath(os.path.dirname(r['path']), src_root)
                r['label'] = (rel_dir.replace(os.sep, '/') if rel_dir else 'root')
            logger.info("Cache loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-analyzing...")
            results = []
            
    if not results:
        logger.info(f"Analyzing {total} images...")
        
        # WindowsのProcessPoolExecutor問題を回避するため、ThreadPoolExecutorに変更
        # init_workerは不要になるので削除し、メインスレッドで初期化したface_appを共有する形にするか
        # あるいは各スレッドで呼び出すか。
        # ここではinit_workerでglobal face_appを作っているので、実はThreadPoolならメインのface_appをそのまま使える。
        # ただしinit_workerはProcessPool用だったので、メインで一度初期化する必要がある。
        
        init_worker() # メインプロセスで初期化
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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

    # Sharpness threshold (upper bound - filter noisy/too sharp images)
    th_sharpness_high = 999999
    if args.sharpness_percentile_high > 0:
        if 'sharp_vals' not in locals():
            sharp_vals = [r['metrics']['sharpness'] for r in valid_items]
        th_sharpness_high = np.percentile(sharp_vals, 100 - args.sharpness_percentile_high)
    
    # Face Size threshold (from filename sz)
    th_face_size_low = 0
    th_face_size_high = 999999
    face_size_vals = [r['metrics'].get('face_size', 0) for r in valid_items if r['metrics'].get('face_size', 0) > 0]
    if face_size_vals:
        if args.face_size_percentile_low > 0:
            th_face_size_low = np.percentile(face_size_vals, args.face_size_percentile_low)
        if args.face_size_percentile_high > 0:
            th_face_size_high = np.percentile(face_size_vals, 100 - args.face_size_percentile_high)
        
    # Aspect Ratio threshold (Two-sided)
    th_ar_low = 0
    th_ar_high = 999
    if args.aspect_ratio_cutoff > 0:
        ar_vals = [r['metrics']['aspect_ratio'] for r in valid_items if 'aspect_ratio' in r['metrics']]
        if ar_vals:
            th_ar_low = np.percentile(ar_vals, args.aspect_ratio_cutoff)
            th_ar_high = np.percentile(ar_vals, 100 - args.aspect_ratio_cutoff)
    
    # Retouching threshold (lower bound - filter retouched/smoothed images)
    th_retouching = 0
    if args.retouching_percentile > 0:
        # skin_smoothnessがinfでないものだけ使う
        retouch_vals = [r['metrics'].get('skin_smoothness', float('inf')) for r in valid_items]
        retouch_vals = [v for v in retouch_vals if v != float('inf')]
        if retouch_vals:
            th_retouching = np.percentile(retouch_vals, args.retouching_percentile)
    
    logger.info(f"Global Thresh: Pitch>={th_pitch:.2f}, Sym>={th_sym:.3f}, YDiff>={th_y:.4f}, Mouth>={th_mouth:.4f}, Sharpness {th_sharpness_low:.1f}~{th_sharpness_high:.1f}, AR<={th_ar_low:.3f}|>={th_ar_high:.3f}, Retouch>={th_retouching:.1f}")
    
    # Mask thresholds (filter top X% - likely mask)
    th_mask = 999.0
    if args.mask_percentile > 0:
        mask_vals = [r['metrics'].get('mask_score', 0) for r in valid_items]
        if mask_vals:
            # 高いスコア(マスク疑惑)をカットするので、Top X% を閾値とする
            # つまり、閾値より低いものを残す
            th_mask = np.percentile(mask_vals, 100 - args.mask_percentile)

    # Glasses thresholds (filter top X% - likely glasses)
    th_glasses = 999.0
    if args.glasses_percentile > 0:
        gl_vals = [r['metrics'].get('glasses_score', 0) for r in valid_items]
        if gl_vals:
             th_glasses = np.percentile(gl_vals, 100 - args.glasses_percentile)
             
    logger.info(f"Mask/Glasses Thresh: Mask<={th_mask:.3f}, Glasses<={th_glasses:.3f}")
    
    # --- Filtering & Grouping ---
    grouped = defaultdict(list)
    for r in valid_items:
        grouped[r['label']].append(r)
        
    final_copy_tasks = []
    skipped_count = 0
    skip_reasons = defaultdict(int)

    # --- Undersampling Logic ---
    # target_count = 2 番目に多いグループの採用枚数（2026-04-25 変更）。
    # 旧: int(mean(counts)) → 最多 1 人に引きずられて中位以下まで削られる副作用があったため、
    # 「最多の 1 人だけ 2 位に合わせて切る」方針に変更。グループが 1 つなら切らない。
    counts_sorted = sorted((len(items) for items in grouped.values()), reverse=True)
    if len(counts_sorted) >= 2:
        target_count = counts_sorted[1]
    elif len(counts_sorted) == 1:
        target_count = counts_sorted[0]
    else:
        target_count = 0
    logger.info(
        f"Undersampling target count (2nd-largest): {target_count} "
        f"(top counts={counts_sorted[:5]}{'...' if len(counts_sorted) > 5 else ''})"
    )
    
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
            elif args.sharpness_percentile_high > 0 and m['sharpness'] > th_sharpness_high: reason = 'sharpness_high_global'
            elif args.face_size_percentile_low > 0 and m.get('face_size', 0) > 0 and m.get('face_size', 0) < th_face_size_low: reason = 'face_size_low_global'
            elif args.face_size_percentile_high > 0 and m.get('face_size', 0) > 0 and m.get('face_size', 0) > th_face_size_high: reason = 'face_size_high_global'
            elif args.aspect_ratio_cutoff > 0 and (m.get('aspect_ratio', 1.0) < th_ar_low or m.get('aspect_ratio', 1.0) > th_ar_high): reason = 'aspect_ratio_global'
            elif args.retouching_percentile > 0 and m.get('skin_smoothness', float('inf')) != float('inf') and m.get('skin_smoothness', float('inf')) < th_retouching: reason = 'retouching_global'
            elif args.mask_percentile > 0 and m.get('mask_score', 0) > th_mask: reason = 'mask_global'
            elif args.glasses_percentile > 0 and m.get('glasses_score', 0) > th_glasses: reason = 'glasses_global'
            
            # Personal Check
            elif (args.eyebrow_eye_percentile_high > 0 and m['eb_eye_dist'] > th_eb_high): reason = 'eb_eye_high_personal'
            elif (args.eyebrow_eye_percentile_low > 0 and m['eb_eye_dist'] < th_eb_low): reason = 'eb_eye_low_personal'
            
            if reason:
                skipped_count += 1
                skip_reasons[reason] += 1
            else:
                label_valid_tasks.append(item)

        # Apply Undersampling Limitation (skip for validation)
        import random
        random.shuffle(label_valid_tasks) # Randomize for bias reduction
        
        count_before_cut = len(label_valid_tasks)
        if not skip_undersampling and count_before_cut > target_count:
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
            
            final_copy_tasks.append((src, dst, args.grayscale))

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
    
    # New Args
    parser.add_argument("--eyebrow_eye_percentile_high", type=int, default=EYEBROW_EYE_PERCENTILE_HIGH, help="Filter top X% of eyebrow-eye distance")
    parser.add_argument("--eyebrow_eye_percentile_low", type=int, default=EYEBROW_EYE_PERCENTILE_LOW, help="Filter bottom X% of eyebrow-eye distance")
    parser.add_argument("--sharpness_percentile_low", type=int, default=SHARPNESS_PERCENTILE_LOW, help="Filter bottom X% by sharpness (blurry images)")
    parser.add_argument("--sharpness_percentile_high", type=int, default=SHARPNESS_PERCENTILE_HIGH, help="Filter top X% by sharpness")
    parser.add_argument("--face_size_percentile_low", type=int, default=FACE_SIZE_PERCENTILE_LOW, help="Filter bottom X% by face size (small images)")
    parser.add_argument("--face_size_percentile_high", type=int, default=FACE_SIZE_PERCENTILE_HIGH, help="Filter top X% by face size (large images)")
    
    # Aspect Ratio
    parser.add_argument("--aspect_ratio_cutoff", type=int, default=0, help="Filter both top/bottom X% outliers in aspect ratio")
    
    # Retouching (美肌加工・SNSフィルター検出)
    # Retouching (美肌加工・SNSフィルター検出)
    parser.add_argument("--retouching_percentile", type=int, default=RETOUCHING_PERCENTILE, help="Filter bottom X% by skin smoothness (retouched images)")
    
    # Mask & Glasses
    parser.add_argument("--mask_percentile", type=int, default=MASK_PERCENTILE, help="Filter top X% by mask likelihood")
    parser.add_argument("--glasses_percentile", type=int, default=GLASSES_PERCENTILE, help="Filter top X% by glasses likelihood")
    
    # Grayscale
    parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")
    
    args = parser.parse_args()
    
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
        logger.info(f"  Sharpness Pct Low/High: {args.sharpness_percentile_low} / {args.sharpness_percentile_high}")
        logger.info(f"  Face Size Pct Low/High: {args.face_size_percentile_low} / {args.face_size_percentile_high}")
        logger.info(f"  Aspect Ratio Cutoff: {args.aspect_ratio_cutoff}")
        logger.info(f"  Retouching Pct: {args.retouching_percentile}")
        logger.info(f"  Mask Pct: {args.mask_percentile}")
        logger.info(f"  Glasses Pct: {args.glasses_percentile}")
        logger.info(f"  Grayscale: {args.grayscale}")
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