import os
import re
import cv2
import json
import logging
import shutil
import argparse
import concurrent.futures
import datetime
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
MEAN_BRIGHTNESS_PERCENTILE_LOW = 0  # Filter bottom X% by mean grayscale luminance (dark images)
FACE_SIZE_PERCENTILE_LOW = 0   # Filter bottom X% by face size (small images)
FACE_SIZE_PERCENTILE_HIGH = 0  # Filter top X% by face size (large images)
# 元画像に対する in-plane 傾き補正量（絶対度数）。ファイル名 `_rz` のみ。無ければ 0。
ROTATION_FILTER_PERCENTILE = 0
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

MANIFEST_SCHEMA_VERSION = 1


def face_roll_abs_deg_from_filename(filename: str):
    """
    part1 が付与する _rz<ミリ度>（度×1000 の整数、符号あり）を読む。
    無ければ None。
    """
    m = re.search(r"_rz(-?\d+)", filename, flags=re.IGNORECASE)
    if not m:
        return None
    mdeg = int(m.group(1))
    return abs(mdeg / 1000.0)


def _manifest_filter_percentile_args(args):
    """CLI で渡したフィルタ関連引数のスナップショット（再現用）。"""
    return {
        "pitch_percentile": int(args.pitch_percentile),
        "symmetry_percentile": int(args.symmetry_percentile),
        "y_diff_percentile": int(args.y_diff_percentile),
        "mouth_open_percentile": int(args.mouth_open_percentile),
        "eyebrow_eye_percentile_high": int(args.eyebrow_eye_percentile_high),
        "eyebrow_eye_percentile_low": int(args.eyebrow_eye_percentile_low),
        "sharpness_percentile_low": int(args.sharpness_percentile_low),
        "sharpness_percentile_high": int(args.sharpness_percentile_high),
        "mean_brightness_percentile_low": int(getattr(args, "mean_brightness_percentile_low", 0)),
        "face_size_percentile_low": int(args.face_size_percentile_low),
        "face_size_percentile_high": int(args.face_size_percentile_high),
        "rotation_percentile": int(getattr(args, "rotation_percentile", 0)),
        "aspect_ratio_cutoff": int(args.aspect_ratio_cutoff),
        "retouching_percentile": int(args.retouching_percentile),
        "mask_percentile": int(args.mask_percentile),
        "glasses_percentile": int(args.glasses_percentile),
        "grayscale": bool(getattr(args, "grayscale", False)),
        "skip_class_balance": bool(getattr(args, "skip_class_balance", False)),
        "class_internal_cap_mode": str(getattr(args, "class_internal_cap_mode", "min")),
        "class_internal_cap_rank": int(getattr(args, "class_internal_cap_rank", 2)),
    }


def _build_split_filter_manifest(
    split_output_subdir,
    src_root,
    dst_root,
    args,
    total,
    valid_count,
    saved_count,
    skipped_count,
    th_pitch,
    th_sym,
    th_y,
    th_mouth,
    th_sharpness_low,
    th_sharpness_high,
    th_mean_brightness_low,
    th_face_size_low,
    th_face_size_high,
    th_ar_low,
    th_ar_high,
    th_retouching,
    th_mask,
    th_glasses,
    th_rotation,
    per_label_eyebrow,
    no_valid_faces=False,
):
    """
    当該スプリットで実際にフィルタ判定に使った実数閾値を記録する。
    キーは preprocess_multitask.process_dataset 内の elif チェーンと対応。
    """
    def _rec(th, use):
        return float(th) if use else None

    use_pitch = getattr(args, "pitch_threshold", None) is not None or args.pitch_percentile > 0
    use_sym = getattr(args, "symmetry_threshold", None) is not None or args.symmetry_percentile > 0
    use_y = getattr(args, "y_diff_threshold", None) is not None or args.y_diff_percentile > 0
    use_mouth = getattr(args, "mouth_open_threshold", None) is not None or args.mouth_open_percentile > 0
    use_sharp_low = getattr(args, "sharpness_low_threshold", None) is not None or args.sharpness_percentile_low > 0
    use_sharp_high = getattr(args, "sharpness_high_threshold", None) is not None or args.sharpness_percentile_high > 0
    use_mb = getattr(args, "mean_brightness_low_threshold", None) is not None or int(
        getattr(args, "mean_brightness_percentile_low", 0) or 0
    ) > 0
    use_fs_low = getattr(args, "face_size_low_threshold", None) is not None or args.face_size_percentile_low > 0
    use_fs_high = getattr(args, "face_size_high_threshold", None) is not None or args.face_size_percentile_high > 0
    use_rot = getattr(args, "rotation_threshold", None) is not None or int(getattr(args, "rotation_percentile", 0) or 0) > 0
    use_retouch = getattr(args, "retouching_threshold", None) is not None or args.retouching_percentile > 0
    use_mask = getattr(args, "mask_threshold", None) is not None or args.mask_percentile > 0
    use_glasses = getattr(args, "glasses_threshold", None) is not None or args.glasses_percentile > 0

    g = {
        "pitch_upper_reject_if_strictly_greater": _rec(th_pitch, use_pitch),
        "symmetry_upper_reject_if_strictly_greater": _rec(th_sym, use_sym),
        "y_diff_upper_reject_if_strictly_greater": _rec(th_y, use_y),
        "mouth_open_upper_reject_if_strictly_greater": _rec(th_mouth, use_mouth),
        "sharpness_lower_reject_if_strictly_less": _rec(th_sharpness_low, use_sharp_low),
        "sharpness_upper_reject_if_strictly_greater": _rec(th_sharpness_high, use_sharp_high),
        "mean_brightness_lower_reject_if_strictly_less": _rec(th_mean_brightness_low, use_mb),
        "face_size_lower_reject_if_strictly_less": _rec(th_face_size_low, use_fs_low),
        "face_size_upper_reject_if_strictly_greater": _rec(th_face_size_high, use_fs_high),
        "face_roll_abs_deg_upper_reject_if_strictly_greater": _rec(th_rotation, use_rot),
        "aspect_ratio_lower_reject_if_strictly_less": float(th_ar_low) if args.aspect_ratio_cutoff > 0 else None,
        "aspect_ratio_upper_reject_if_strictly_greater": float(th_ar_high) if args.aspect_ratio_cutoff > 0 else None,
        "skin_smoothness_lower_reject_if_strictly_less": _rec(th_retouching, use_retouch),
        "mask_score_upper_reject_if_strictly_greater": _rec(th_mask, use_mask),
        "glasses_score_upper_reject_if_strictly_greater": _rec(th_glasses, use_glasses),
    }
    return {
        "split_output_subdir": split_output_subdir,
        "source_root": src_root,
        "destination_root": dst_root,
        "total_files_scanned": int(total),
        "valid_face_count": int(valid_count),
        "saved_count": int(saved_count),
        "skipped_count_reported": int(skipped_count),
        "no_valid_faces": bool(no_valid_faces),
        "filter_percentile_args": _manifest_filter_percentile_args(args),
        "global_numeric_thresholds": g,
        "per_label_eyebrow_thresholds": per_label_eyebrow,
    }


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
    mean_brightness = float(np.mean(gray))
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
    filename = os.path.basename(path)
    sz_match = re.search(r'_sz(\d+)', filename)
    face_size = int(sz_match.group(1)) if sz_match else 0

    # In-plane roll（元画像に対する傾き補正の大きさ［度・絶対値］）。ファイル名の _rz のみ。
    # 正立化済み切り出し画像からは元の回転角は復元不能のため、無ければ 0。
    roll_fn = face_roll_abs_deg_from_filename(filename)
    face_roll_deg_abs = roll_fn if roll_fn is not None else 0.0

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
        'mean_brightness': mean_brightness,
        'skin_smoothness': skin_smoothness,
        'aspect_ratio': aspect_ratio,
        'face_size': face_size,
        'face_roll_deg_abs': face_roll_deg_abs,
        'mask_score': mask_score,
        'glasses_score': glasses_score,
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


def _class_key_from_rel_label(label):
    """
    process_dataset のグループラベル（src_root からの相対ディレクトリ、'/' 区切り）から
    「クラス」キーを取る。先頭セグメント = 単一タスクのクラスフォルダ名（例: a, z）。
    スラッシュが無いときはラベル全体をクラスとみなす（従来バグ: 空文字にまとめない）。
    """
    if not label or label == "root":
        return label if label else "root"
    if "/" in label:
        return label.split("/", 1)[0]
    return label


def _apply_class_internal_person_cap(
    filtered_by_label, skip_reasons, *, round_name, cap_mode, cap_rank=2
):
    """
    クラス内で各人物（フルラベル＝フォルダ）バケツの枚数をクラス内で揃えて切り詰める。

    cap_mode:
      - "min": そのクラス内の最少人物枚数（最下位）を上限とする。
      - "rank": cap_rank 番目に多いバケツの枚数を上限（cap_rank=2 が旧「2位上限」）。

    filtered_by_label を in-place 更新し、skip_reasons['undersampling_post_filter'] に落とした枚数を加算する。
    戻り値: 落とした枚数の合計（skipped_count 用）。
    """
    dropped_total = 0
    mode = str(cap_mode).strip().lower()
    cr = max(1, int(cap_rank))
    class_to_person_labels = defaultdict(list)
    for lb in filtered_by_label:
        class_key = _class_key_from_rel_label(lb)
        class_to_person_labels[class_key].append(lb)

    for class_key, person_labels in class_to_person_labels.items():
        post_counts = [len(filtered_by_label[pl]) for pl in person_labels]
        if not post_counts:
            target_post = 0
        elif mode == "min":
            target_post = min(post_counts)
        elif mode == "rank":
            sorted_desc = sorted(post_counts, reverse=True)
            idx = min(cr - 1, len(sorted_desc) - 1)
            target_post = sorted_desc[idx]
        else:
            raise ValueError(f"unknown class_internal_cap_mode: {cap_mode!r}")
        if mode == "min":
            cap_desc = f"mode=min -> bucket_count_ceiling={target_post} (class min among persons)"
        else:
            cap_desc = (
                f"mode=rank cap_rank={cr} -> bucket_count_ceiling={target_post}"
            )
        logger.info(
            f"Undersampling (post-filter round {round_name}, per-class) class={class_key!r}: "
            f"{cap_desc}, persons={len(person_labels)}, counts={sorted(post_counts, reverse=True)}"
        )
        for pl in person_labels:
            tasks = filtered_by_label[pl]
            random.shuffle(tasks)
            n0 = len(tasks)
            if n0 > target_post:
                drop = n0 - target_post
                tasks = tasks[:target_post]
                dropped_total += drop
                skip_reasons["undersampling_post_filter"] += drop
            filtered_by_label[pl] = tasks
    return dropped_total


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
    cache_key = f"{os.path.basename(src_root)}_{total}_rz2_mb"
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
        split_sub = os.path.basename(os.path.normpath(dst_root))
        empty_manifest = {
            "split_output_subdir": split_sub,
            "source_root": src_root,
            "destination_root": dst_root,
            "total_files_scanned": int(total),
            "valid_face_count": 0,
            "saved_count": 0,
            "skipped_count_reported": int(total),
            "no_valid_faces": True,
            "filter_percentile_args": _manifest_filter_percentile_args(args),
            "global_numeric_thresholds": None,
            "per_label_eyebrow_thresholds": {},
            "note": "valid 顔が0件のためパーセンタイル閾値は未算出。",
        }
        return total, 0, total, empty_manifest

    # --- Global Threshold Calculation ---
    def get_th(key, pct):
        vals = [r['metrics'][key] for r in valid_items]
        if not vals:
            return 0
        return np.percentile(vals, 100 - pct)

    fix_pitch = getattr(args, "pitch_threshold", None)
    if fix_pitch is not None:
        th_pitch = float(fix_pitch)
    elif args.pitch_percentile > 0:
        th_pitch = get_th("pitch", args.pitch_percentile)
    else:
        th_pitch = 0.0
    use_pitch = fix_pitch is not None or args.pitch_percentile > 0

    fix_sym = getattr(args, "symmetry_threshold", None)
    if fix_sym is not None:
        th_sym = float(fix_sym)
    elif args.symmetry_percentile > 0:
        th_sym = get_th("symmetry", args.symmetry_percentile)
    else:
        th_sym = 0.0
    use_sym = fix_sym is not None or args.symmetry_percentile > 0

    fix_y = getattr(args, "y_diff_threshold", None)
    if fix_y is not None:
        th_y = float(fix_y)
    elif args.y_diff_percentile > 0:
        th_y = get_th("y_diff", args.y_diff_percentile)
    else:
        th_y = 0.0
    use_y = fix_y is not None or args.y_diff_percentile > 0

    fix_mouth = getattr(args, "mouth_open_threshold", None)
    if fix_mouth is not None:
        th_mouth = float(fix_mouth)
    elif args.mouth_open_percentile > 0:
        th_mouth = get_th("mouth_open", args.mouth_open_percentile)
    else:
        th_mouth = 0.0
    use_mouth = fix_mouth is not None or args.mouth_open_percentile > 0

    fix_rotation = getattr(args, "rotation_threshold", None)
    if fix_rotation is not None:
        th_rotation = float(fix_rotation)
    elif getattr(args, "rotation_percentile", 0) > 0:
        th_rotation = get_th("face_roll_deg_abs", args.rotation_percentile)
    else:
        th_rotation = 0.0
    use_rotation = fix_rotation is not None or getattr(args, "rotation_percentile", 0) > 0

    fix_sharp_low = getattr(args, "sharpness_low_threshold", None)
    if fix_sharp_low is not None:
        th_sharpness_low = float(fix_sharp_low)
    elif args.sharpness_percentile_low > 0:
        sharp_vals = [r["metrics"]["sharpness"] for r in valid_items]
        th_sharpness_low = np.percentile(sharp_vals, args.sharpness_percentile_low)
    else:
        th_sharpness_low = 0
    use_sharp_low = fix_sharp_low is not None or args.sharpness_percentile_low > 0

    fix_sharp_high = getattr(args, "sharpness_high_threshold", None)
    if fix_sharp_high is not None:
        th_sharpness_high = float(fix_sharp_high)
    elif args.sharpness_percentile_high > 0:
        if "sharp_vals" not in locals():
            sharp_vals = [r["metrics"]["sharpness"] for r in valid_items]
        th_sharpness_high = np.percentile(sharp_vals, 100 - args.sharpness_percentile_high)
    else:
        th_sharpness_high = 999999
    use_sharp_high = fix_sharp_high is not None or args.sharpness_percentile_high > 0

    mb_pct = int(getattr(args, "mean_brightness_percentile_low", 0) or 0)
    fix_mb_low = getattr(args, "mean_brightness_low_threshold", None)
    if fix_mb_low is not None:
        th_mean_brightness_low = float(fix_mb_low)
    elif mb_pct > 0:
        mb_vals = [r["metrics"]["mean_brightness"] for r in valid_items]
        th_mean_brightness_low = float(np.percentile(mb_vals, mb_pct))
    else:
        th_mean_brightness_low = 0.0
    use_mb_low = fix_mb_low is not None or mb_pct > 0

    # Face Size threshold (from filename sz)
    th_face_size_low = 0
    th_face_size_high = 999999
    face_size_vals = [
        r["metrics"].get("face_size", 0) for r in valid_items if r["metrics"].get("face_size", 0) > 0
    ]
    fix_fs_low = getattr(args, "face_size_low_threshold", None)
    fix_fs_high = getattr(args, "face_size_high_threshold", None)
    if face_size_vals:
        if fix_fs_low is not None:
            th_face_size_low = float(fix_fs_low)
        elif args.face_size_percentile_low > 0:
            th_face_size_low = np.percentile(face_size_vals, args.face_size_percentile_low)
        if fix_fs_high is not None:
            th_face_size_high = float(fix_fs_high)
        elif args.face_size_percentile_high > 0:
            th_face_size_high = np.percentile(face_size_vals, 100 - args.face_size_percentile_high)
    else:
        if fix_fs_low is not None:
            th_face_size_low = float(fix_fs_low)
        if fix_fs_high is not None:
            th_face_size_high = float(fix_fs_high)
    use_face_low = fix_fs_low is not None or (bool(face_size_vals) and args.face_size_percentile_low > 0)
    use_face_high = fix_fs_high is not None or (bool(face_size_vals) and args.face_size_percentile_high > 0)
        
    # Aspect Ratio threshold (Two-sided)
    th_ar_low = 0
    th_ar_high = 999
    if args.aspect_ratio_cutoff > 0:
        ar_vals = [r['metrics']['aspect_ratio'] for r in valid_items if 'aspect_ratio' in r['metrics']]
        if ar_vals:
            th_ar_low = np.percentile(ar_vals, args.aspect_ratio_cutoff)
            th_ar_high = np.percentile(ar_vals, 100 - args.aspect_ratio_cutoff)
    
    # Retouching threshold (lower bound - filter retouched/smoothed images)
    fix_retouch = getattr(args, "retouching_threshold", None)
    if fix_retouch is not None:
        th_retouching = float(fix_retouch)
        use_retouching = True
    elif args.retouching_percentile > 0:
        retouch_vals = [r["metrics"].get("skin_smoothness", float("inf")) for r in valid_items]
        retouch_vals = [v for v in retouch_vals if v != float("inf")]
        th_retouching = (
            np.percentile(retouch_vals, args.retouching_percentile) if retouch_vals else 0
        )
        use_retouching = bool(retouch_vals)
    else:
        th_retouching = 0
        use_retouching = False

    roll_log = f", RollAbsDeg>{th_rotation:.4f}" if use_rotation else ""
    mb_log = f", MeanBright>={th_mean_brightness_low:.1f}" if use_mb_low else ""
    logger.info(
        f"Global Thresh: Pitch>={th_pitch:.2f}, Sym>={th_sym:.3f}, YDiff>={th_y:.4f}, Mouth>={th_mouth:.4f}, "
        f"Sharpness {th_sharpness_low:.1f}~{th_sharpness_high:.1f}{mb_log}, "
        f"AR<={th_ar_low:.3f}|>={th_ar_high:.3f}, Retouch>={th_retouching:.1f}"
        f"{roll_log}"
    )
    
    fix_mask = getattr(args, "mask_threshold", None)
    if fix_mask is not None:
        th_mask = float(fix_mask)
        use_mask = True
    elif args.mask_percentile > 0:
        mask_vals = [r["metrics"].get("mask_score", 0) for r in valid_items]
        th_mask = np.percentile(mask_vals, 100 - args.mask_percentile) if mask_vals else 999.0
        use_mask = bool(mask_vals)
    else:
        th_mask = 999.0
        use_mask = False

    fix_glasses = getattr(args, "glasses_threshold", None)
    if fix_glasses is not None:
        th_glasses = float(fix_glasses)
        use_glasses = True
    elif args.glasses_percentile > 0:
        gl_vals = [r["metrics"].get("glasses_score", 0) for r in valid_items]
        th_glasses = np.percentile(gl_vals, 100 - args.glasses_percentile) if gl_vals else 999.0
        use_glasses = bool(gl_vals)
    else:
        th_glasses = 999.0
        use_glasses = False
             
    logger.info(f"Mask/Glasses Thresh: Mask<={th_mask:.3f}, Glasses<={th_glasses:.3f}")
    
    # --- Filtering & Grouping ---
    grouped = defaultdict(list)
    for r in valid_items:
        grouped[r['label']].append(r)

    final_copy_tasks = []
    skipped_count = 0
    skip_reasons = defaultdict(int)

    # フィルタのみ → dict[label, tasks]
    filtered_by_label = {}
    per_label_eyebrow = {}
    for label, items in grouped.items():
        eb_vals = [r['metrics']['eb_eye_dist'] for r in items]
        th_eb_high = 999.0
        th_eb_low = -1.0

        if eb_vals:
            if args.eyebrow_eye_percentile_high > 0:
                th_eb_high = np.percentile(eb_vals, 100 - args.eyebrow_eye_percentile_high)
            if args.eyebrow_eye_percentile_low > 0:
                th_eb_low = np.percentile(eb_vals, args.eyebrow_eye_percentile_low)

        per_label_eyebrow[label] = {
            "eb_eye_dist_upper_reject_if_strictly_greater": float(th_eb_high)
            if args.eyebrow_eye_percentile_high > 0
            else None,
            "eb_eye_dist_lower_reject_if_strictly_less": float(th_eb_low)
            if args.eyebrow_eye_percentile_low > 0
            else None,
        }

        label_valid_tasks = []
        for item in items:
            m = item['metrics']
            reason = None

            if use_pitch and m["pitch"] > th_pitch:
                reason = "pitch_global"
            elif use_sym and m["symmetry"] > th_sym:
                reason = "symmetry_global"
            elif use_y and m["y_diff"] > th_y:
                reason = "y_diff_global"
            elif use_mouth and m["mouth_open"] > th_mouth:
                reason = "mouth_open_global"
            elif use_sharp_low and m["sharpness"] < th_sharpness_low:
                reason = "sharpness_low_global"
            elif use_sharp_high and m["sharpness"] > th_sharpness_high:
                reason = "sharpness_high_global"
            elif use_mb_low and m["mean_brightness"] < th_mean_brightness_low:
                reason = "mean_brightness_low_global"
            elif use_face_low and m.get("face_size", 0) > 0 and m.get("face_size", 0) < th_face_size_low:
                reason = "face_size_low_global"
            elif use_face_high and m.get("face_size", 0) > 0 and m.get("face_size", 0) > th_face_size_high:
                reason = "face_size_high_global"
            elif use_rotation and m.get("face_roll_deg_abs", 0) > th_rotation:
                reason = "rotation_global"
            elif args.aspect_ratio_cutoff > 0 and (
                m.get("aspect_ratio", 1.0) < th_ar_low or m.get("aspect_ratio", 1.0) > th_ar_high
            ):
                reason = "aspect_ratio_global"
            elif use_retouching and m.get("skin_smoothness", float("inf")) != float("inf") and m.get(
                "skin_smoothness", float("inf")
            ) < th_retouching:
                reason = "retouching_global"
            elif use_mask and m.get("mask_score", 0) > th_mask:
                reason = "mask_global"
            elif use_glasses and m.get("glasses_score", 0) > th_glasses:
                reason = "glasses_global"
            elif (args.eyebrow_eye_percentile_high > 0 and m['eb_eye_dist'] > th_eb_high):
                reason = 'eb_eye_high_personal'
            elif (args.eyebrow_eye_percentile_low > 0 and m['eb_eye_dist'] < th_eb_low):
                reason = 'eb_eye_low_personal'

            if reason:
                skipped_count += 1
                skip_reasons[reason] += 1
            else:
                label_valid_tasks.append(item)

        filtered_by_label[label] = label_valid_tasks

    cap_mode = str(getattr(args, "class_internal_cap_mode", "min")).strip().lower()
    cap_rank = max(1, int(getattr(args, "class_internal_cap_rank", 2)))
    # --- Undersampling: (1) クラス内人物均し → (2) クラス間均衡 → (3) 再度クラス内人物均し ---
    if not skip_undersampling:
        skipped_count += _apply_class_internal_person_cap(
            filtered_by_label,
            skip_reasons,
            round_name="1",
            cap_mode=cap_mode,
            cap_rank=cap_rank,
        )

    if not skip_undersampling and not getattr(args, "skip_class_balance", False):
        label_to_class = {lb: _class_key_from_rel_label(lb) for lb in filtered_by_label}
        totals = defaultdict(int)
        for lb, tasks in filtered_by_label.items():
            totals[label_to_class[lb]] += len(tasks)
        active_totals = {k: v for k, v in totals.items() if v > 0}
        if len(active_totals) >= 2:
            target_bal = min(active_totals.values())
            logger.info(
                f"Undersampling (class balance) target total per class={target_bal} "
                f"(class totals before={dict(sorted(active_totals.items()))})"
            )
            for ck, total_ck in list(active_totals.items()):
                if total_ck <= target_bal:
                    continue
                cls_labels = [lb for lb in filtered_by_label if label_to_class[lb] == ck]
                pooled = []
                for lb in cls_labels:
                    for it in filtered_by_label[lb]:
                        pooled.append((lb, it))
                random.shuffle(pooled)
                keep_pairs = pooled[:target_bal]
                kept_by_label = defaultdict(list)
                for lb, it in keep_pairs:
                    kept_by_label[lb].append(it)
                for lb in cls_labels:
                    before_n = len(filtered_by_label[lb])
                    new_tasks = kept_by_label.get(lb, [])
                    dropped = before_n - len(new_tasks)
                    if dropped > 0:
                        skipped_count += dropped
                        skip_reasons["undersampling_class_balance"] += dropped
                    filtered_by_label[lb] = new_tasks

    if not skip_undersampling:
        skipped_count += _apply_class_internal_person_cap(
            filtered_by_label,
            skip_reasons,
            round_name="2",
            cap_mode=cap_mode,
            cap_rank=cap_rank,
        )

    for label, label_valid_tasks in filtered_by_label.items():
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

    split_sub = os.path.basename(os.path.normpath(dst_root))
    manifest = _build_split_filter_manifest(
        split_sub,
        src_root,
        dst_root,
        args,
        total,
        len(valid_items),
        saved_count,
        skipped_count,
        th_pitch,
        th_sym,
        th_y,
        th_mouth,
        th_sharpness_low,
        th_sharpness_high,
        th_mean_brightness_low,
        th_face_size_low,
        th_face_size_high,
        th_ar_low,
        th_ar_high,
        th_retouching,
        th_mask,
        th_glasses,
        th_rotation,
        per_label_eyebrow,
    )
    return total, saved_count, skipped_count, manifest

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
    parser.add_argument(
        "--mean_brightness_percentile_low",
        type=int,
        default=MEAN_BRIGHTNESS_PERCENTILE_LOW,
        help="Filter bottom X% by mean grayscale brightness (dark images)",
    )
    parser.add_argument("--face_size_percentile_low", type=int, default=FACE_SIZE_PERCENTILE_LOW, help="Filter bottom X% by face size (small images)")
    parser.add_argument("--face_size_percentile_high", type=int, default=FACE_SIZE_PERCENTILE_HIGH, help="Filter top X% by face size (large images)")
    parser.add_argument(
        "--rotation_percentile",
        type=int,
        default=ROTATION_FILTER_PERCENTILE,
        help="Filter top X% by in-plane roll correction (deg abs); only when filename has part1 _rz tag, else metric is 0",
    )

    # Aspect Ratio
    parser.add_argument("--aspect_ratio_cutoff", type=int, default=0, help="Filter both top/bottom X% outliers in aspect ratio")
    
    # Retouching (美肌加工・SNSフィルター検出)
    # Retouching (美肌加工・SNSフィルター検出)
    parser.add_argument("--retouching_percentile", type=int, default=RETOUCHING_PERCENTILE, help="Filter bottom X% by skin smoothness (retouched images)")
    
    # Mask & Glasses
    parser.add_argument("--mask_percentile", type=int, default=MASK_PERCENTILE, help="Filter top X% by mask likelihood")
    parser.add_argument("--glasses_percentile", type=int, default=GLASSES_PERCENTILE, help="Filter top X% by glasses likelihood")

    # 固定実数閾値（指定時は当該軸でパーセンタイルより優先。optimize Phase2 greedy 用）
    parser.add_argument(
        "--pitch_threshold",
        type=float,
        default=None,
        help="固定: pitch 上限（これを超えると除外）。指定時 --pitch_percentile は無視",
    )
    parser.add_argument(
        "--symmetry_threshold",
        type=float,
        default=None,
        help="固定: symmetry 上限。指定時 --symmetry_percentile は無視",
    )
    parser.add_argument(
        "--y_diff_threshold",
        type=float,
        default=None,
        help="固定: y_diff 上限。指定時 --y_diff_percentile は無視",
    )
    parser.add_argument(
        "--mouth_open_threshold",
        type=float,
        default=None,
        help="固定: mouth_open 上限。指定時 --mouth_open_percentile は無視",
    )
    parser.add_argument(
        "--sharpness_low_threshold",
        type=float,
        default=None,
        help="固定: sharpness 下限（未満除外）。指定時 --sharpness_percentile_low は無視",
    )
    parser.add_argument(
        "--sharpness_high_threshold",
        type=float,
        default=None,
        help="固定: sharpness 上限。指定時 --sharpness_percentile_high は無視",
    )
    parser.add_argument(
        "--mean_brightness_low_threshold",
        type=float,
        default=None,
        help="固定: mean_brightness 下限。指定時 --mean_brightness_percentile_low は無視",
    )
    parser.add_argument(
        "--face_size_low_threshold",
        type=float,
        default=None,
        help="固定: face_size 下限。指定時 --face_size_percentile_low は無視",
    )
    parser.add_argument(
        "--face_size_high_threshold",
        type=float,
        default=None,
        help="固定: face_size 上限。指定時 --face_size_percentile_high は無視",
    )
    parser.add_argument(
        "--rotation_threshold",
        type=float,
        default=None,
        help="固定: face_roll_abs_deg 上限。指定時 --rotation_percentile は無視",
    )
    parser.add_argument(
        "--retouching_threshold",
        type=float,
        default=None,
        help="固定: skin_smoothness 下限。指定時 --retouching_percentile は無視",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=None,
        help="固定: mask_score 上限。指定時 --mask_percentile は無視",
    )
    parser.add_argument(
        "--glasses_threshold",
        type=float,
        default=None,
        help="固定: glasses_score 上限。指定時 --glasses_percentile は無視",
    )

    # Grayscale
    parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")

    parser.add_argument(
        "--filter_manifest_path",
        type=str,
        default="",
        help=(
            "フィルタ実数閾値JSONの出力パス。"
            "空文字なら <out_dir>/filter_threshold_manifest.json 。"
            "\"none\"（大小無視）で出力スキップ。"
        ),
    )
    parser.add_argument(
        "--no_filter_manifest",
        action="store_true",
        help="filter_threshold_manifest.json を書き出さない",
    )
    parser.add_argument(
        "--skip_class_balance",
        action="store_true",
        help=(
            "中段アンダーサンプリング（クラス間・各クラス合計を最少クラスに揃える）をスキップ。"
            "前後のクラス内人物均し（第1・第3段、--class_internal_cap_mode）は従来どおり実行"
        ),
    )
    parser.add_argument(
        "--class_internal_cap_mode",
        choices=("min", "rank"),
        default="min",
        help=(
            "クラス内アンダーサンプル（第1・第3段）。min=各人物をクラス内の最少枚数（最下位）に揃える（既定）。"
            "rank=クラス内で N 番目に多いバケツ枚数を上限（--class_internal_cap_rank）。"
        ),
    )
    parser.add_argument(
        "--class_internal_cap_rank",
        type=int,
        default=2,
        help=(
            "--class_internal_cap_mode=rank のときのみ有効。"
            "各人物バケツの上限を「そのクラス内で N 番目に多い枚数」に揃える（N=2 が旧2位上限）。N>=1。"
        ),
    )

    args = parser.parse_args()

    if str(getattr(args, "class_internal_cap_mode", "min")).strip().lower() not in (
        "min",
        "rank",
    ):
        parser.error("--class_internal_cap_mode は min または rank を指定してください")
    if int(getattr(args, "class_internal_cap_rank", 2)) < 1:
        parser.error("--class_internal_cap_rank は 1 以上を指定してください")
    
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
        logger.info(f"  Mean brightness Pct Low: {args.mean_brightness_percentile_low}")
        logger.info(f"  Face Size Pct Low/High: {args.face_size_percentile_low} / {args.face_size_percentile_high}")
        logger.info(f"  Rotation (roll abs deg) Pct: {getattr(args, 'rotation_percentile', 0)}")
        logger.info(f"  Aspect Ratio Cutoff: {args.aspect_ratio_cutoff}")
        logger.info(f"  Retouching Pct: {args.retouching_percentile}")
        logger.info(f"  Mask Pct: {args.mask_percentile}")
        logger.info(f"  Glasses Pct: {args.glasses_percentile}")
        logger.info(f"  Grayscale: {args.grayscale}")
        logger.info(
            f"  Class-internal undersample: mode={args.class_internal_cap_mode}"
            + (
                f", cap_rank={args.class_internal_cap_rank}"
                if args.class_internal_cap_mode == "rank"
                else " (per-class min person count)"
            )
        )
        fmp_raw = (getattr(args, "filter_manifest_path", "") or "").strip()
        write_manifest = not getattr(args, "no_filter_manifest", False)
        manifest_out = ""
        if write_manifest:
            if fmp_raw.lower() == "none":
                write_manifest = False
            else:
                manifest_out = fmp_raw if fmp_raw else os.path.join(prepro_dir, "filter_threshold_manifest.json")
        if write_manifest:
            logger.info(f"  Filter manifest -> {manifest_out}")
        logger.info("=" * 60)
        
        _, _, _, m_train = process_dataset(args.train_dir, prepro_train, args)
        _, _, _, m_val = process_dataset(args.val_dir, prepro_valid, args)
        split_manifests = {"train": m_train, "validation": m_val}
        if os.path.exists(args.test_dir):
            _, _, _, m_test = process_dataset(args.test_dir, prepro_test, args)
            split_manifests["test"] = m_test

        if write_manifest and manifest_out:
            manifest_abspath = os.path.abspath(manifest_out)
            mdir = os.path.dirname(manifest_abspath)
            if mdir:
                os.makedirs(mdir, exist_ok=True)
            combined = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "out_dir": os.path.abspath(prepro_dir),
                "preprocess_filter_args": _manifest_filter_percentile_args(args),
                "notes": [
                    "train / validation / test は各スプリットの valid 顔集合に対して別々にパーセンタイル閾値を算出している。",
                    "推論で学習データと同じ基準に揃える場合は通常 train の global / per_label を参照する。",
                    "眉-目距離はラベル（人物フォルダ等）単位。同一キーで per_label_eyebrow_thresholds を参照すること。",
                    "アンダーサンプリングは閾値だけでは再現できない（枚数上限・ランダムシャッフルあり）。"
                    "第1段＝クラス内人物均し（--class_internal_cap_mode 既定 min＝最少人物枚数に揃える、rank で N 位上限）、第2段＝クラス間均衡、第3段＝再度クラス内人物均し（--skip_class_balance で第2段のみ無効化）。",
                ],
                "splits": split_manifests,
            }
            with open(manifest_abspath, "w", encoding="utf-8") as _mf:
                json.dump(combined, _mf, indent=2, ensure_ascii=False)
            logger.info(f"Wrote filter threshold manifest: {manifest_abspath}")
        
        logger.info("All processing complete.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    # Windows specific fix
    # multiprocessing.freeze_support() 
    main()