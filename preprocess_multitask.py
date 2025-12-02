import os
import cv2
import logging
import shutil
import argparse
import concurrent.futures
import numpy as np
from insightface.app import FaceAnalysis

# 簡潔ログ
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", encoding='utf-8', force=True)
logger = logging.getLogger(__name__)

# --- Default Configuration (Fallback) ---
DEFAULT_TRAIN_DIR = "train"
DEFAULT_VALIDATION_DIR = "validation"
DEFAULT_PREPRO_DIR = "preprocessed_multitask"
# チェック閾値（画像幅/高さに対する割合）
DEFAULT_THRESH_RATIO = 1.0

# Pose filtering: 上位何%をフィルタリングするか（ここを変更するだけでOK）
PITCH_FILTER_PERCENTILE = 0      # 上位0%（前傾後傾）をフィルタ = フィルタ無効
SYMMETRY_FILTER_PERCENTILE = 40  # 上位40%（左右非対称）をフィルタ
SAMPLE_SIZE = 500                # 閾値計算用のサンプル数

# InsightFaceのランドマークインデックス
LEFT_CHEEK_IDX = 28
RIGHT_CHEEK_IDX = 12

# Global variables for worker processes
face_app = None
pitch_threshold = None
symmetry_threshold = None

def init_worker(pitch_th, symmetry_th):
    """
    Worker process initialization.
    Each process needs its own FaceAnalysis instance.
    """
    global face_app, pitch_threshold, symmetry_threshold
    pitch_threshold = pitch_th
    symmetry_threshold = symmetry_th
    # Providers can be adjusted based on environment (e.g., CPU only for workers if GPU memory is tight)
    # For now, we try CUDA if available, else CPU.
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(320, 320))

def extract_face_info(img_bgr):
    """画像から顔情報を抽出する"""
    if img_bgr is None:
        return None
    # face_app is global in worker process
    faces = face_app.get(img_bgr)
    if not faces:
        return None
    return faces[0]

def calculate_thresholds(src_dir, sample_size=500, pitch_percentile=50, symmetry_percentile=50):
    """
    データセットをサンプリングしてpitchと左右対称性の閾値を計算
    Args:
        src_dir: ソースディレクトリ
        sample_size: サンプル数
        pitch_percentile: pitchのパーセンタイル（上位 X% をフィルタ）
        symmetry_percentile: 対称性のパーセンタイル（上位 Y% をフィルタ）
    Returns:
        (pitch_threshold, symmetry_threshold)
    """
    logger.info(f"データセットをサンプリング中... (最大{sample_size}枚)")
    
    # 画像ファイルを収集
    image_files = []
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_files.append(os.path.join(root, fn))
                if len(image_files) >= sample_size:
                    break
        if len(image_files) >= sample_size:
            break
    
    if len(image_files) == 0:
        logger.warning("サンプル画像が見つかりません。デフォルト閾値を使用します。")
        return 15.0, 0.2  # デフォルト値 (pitch, symmetry)
    
    # InsightFaceを初期化
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    
    pitch_values = []
    symmetry_values = []
    
    logger.info(f"{len(image_files)}枚の画像を分析中...")
    for img_path in image_files:
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            faces = app.get(img)
            if not faces:
                continue
            
            face = faces[0]
            
            # Pitch値を収集
            if face.pose is not None:
                pitch, yaw, roll = face.pose
                pitch_values.append(abs(pitch))
            
            # 対称性比率を計算
            if face.landmark_2d_106 is not None:
                landmarks = face.landmark_2d_106
                if len(landmarks) > max(LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX):
                    h, w = img.shape[:2]
                    lx, ly = landmarks[LEFT_CHEEK_IDX]
                    rx, ry = landmarks[RIGHT_CHEEK_IDX]
                    
                    center_x = w / 2.0
                    diff_left_screen = lx - center_x
                    diff_right_screen = center_x - rx
                    
                    # 位置が異常な顔はスキップ
                    if diff_right_screen <= 0 or diff_left_screen <= 0:
                        continue
                    
                    # 左右の距離の比率（対称性）
                    ratio = abs(diff_left_screen / diff_right_screen - 1)
                    symmetry_values.append(ratio)
        except:
            continue
    
    if len(pitch_values) < 10 or len(symmetry_values) < 10:
        logger.warning(f"pose値が十分に取得できませんでした。デフォルト閾値を使用します。")
        return 15.0, 0.2
    
    # パーセンタイル計算
    pitch_th = np.percentile(pitch_values, 100 - pitch_percentile)
    symmetry_th = np.percentile(symmetry_values, 100 - symmetry_percentile)
    
    logger.info(f"閾値計算完了:")
    logger.info(f"  Pitch (前傾後傾): 上位{pitch_percentile}%をフィルタ (>= {pitch_th:.2f}°)")
    logger.info(f"  Symmetry (左右対称): 上位{symmetry_percentile}%をフィルタ (>= {symmetry_th:.3f})")
    
    return pitch_th, symmetry_th

def process_single_image(args):
    """
    Single image processing function for parallel execution.
    Args:
        args: tuple (src_path, dst_path, thresh_ratio)
    Returns:
        tuple: (status, reason)
        status: 'saved', 'skipped', or 'error'
        reason: detailed reason for skipping or error
    """
    src_path, dst_path, thresh_ratio = args
    
    try:
        img = cv2.imread(src_path)
        if img is None:
            return 'skipped', 'read_error'

        h, w = img.shape[:2]
        face = extract_face_info(img)
        
        if face is None:
            return 'skipped', 'no_face_detected'
        
        landmarks = face.landmark_2d_106
        if landmarks is None or max(LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX) >= len(landmarks):
            return 'skipped', 'landmarks_missing'

        # ランドマーク座標を取得
        lx, ly = landmarks[LEFT_CHEEK_IDX]
        rx, ry = landmarks[RIGHT_CHEEK_IDX]

        # 画面中心
        center_x = w / 2.0

        # ユーザー指定の計算式に基づく差分
        # 左頬(画面右側, rx) - 中心 -> 正
        diff_left_screen = lx - center_x
        # 中心 - 右頬(画面左側, lx) -> 正
        diff_right_screen = center_x - rx 

        if diff_right_screen <= 0 or diff_left_screen <= 0:
            return 'skipped', f'face_position_invalid_left={diff_left_screen:.1f}_right={diff_right_screen:.1f}'


        # 左右の距離の比率チェック（対称性）
        # SYMMETRY_FILTER_PERCENTILE > 0 の場合のみチェック
        if symmetry_threshold is not None and SYMMETRY_FILTER_PERCENTILE > 0:
            # ユーザー指定: 左右の比率が閾値を超えたらスキップ
            if abs(diff_left_screen / diff_right_screen - 1) >= symmetry_threshold:
                ratio = abs(diff_left_screen / diff_right_screen - 1)
                return 'skipped', f'symmetry_ratio_{ratio:.3f}>={symmetry_threshold:.3f}'

        # y差チェック (既存のまま)
        if abs(ry - ly) >= thresh_ratio * h:
            return 'skipped', f'y_diff_{abs(ry-ly):.1f}>={thresh_ratio*h:.1f}'
        
        # ピッチチェック（顔の向き - 前傾後傾のみ）
        # PITCH_FILTER_PERCENTILE > 0 の場合のみチェック
        if face.pose is not None and pitch_threshold is not None and PITCH_FILTER_PERCENTILE > 0:
            pitch, yaw, roll = face.pose
            # グローバル閾値を使用（パーセンタイルベース）
            if abs(pitch) > pitch_threshold:
                return 'skipped', f'pitch_{abs(pitch):.1f}>{pitch_threshold:.1f}'

        # フィルタを通過した画像をコピー
        shutil.copy(src_path, dst_path)
        return 'saved', None

    except Exception as e:
        return 'error', str(e)

def process_folder(src_dir, dst_dir, thresh_ratio, pitch_th, symmetry_th):
    """
    フォルダを再帰的に処理し、フィルタリング後の画像を保存する。
    並列処理を使用。
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    logger.info(f"Scanning files in {src_dir}...")
    
    tasks = []
    
    # 1. Collect all tasks first
    for root, _, files in os.walk(src_dir):
        if root == src_dir:
            continue

        rel_path = os.path.relpath(root, src_dir)
        label_name = rel_path.split(os.sep)[0]
        dst_label_path = os.path.join(dst_dir, label_name)
        os.makedirs(dst_label_path, exist_ok=True)

        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            
            src_path = os.path.join(root, fn)
            dst_path = os.path.join(dst_label_path, fn)
            tasks.append((src_path, dst_path, thresh_ratio))

    total = len(tasks)
    logger.info(f"Found {total} images. Starting parallel processing...")

    saved = 0
    skipped = 0
    errors = 0

    # 2. Execute in parallel
    # Adjust max_workers based on CPU cores or GPU memory constraints
    # Since InsightFace uses GPU/CPU, too many workers might OOM if using GPU.
    # If running on CPU, more workers is fine.
    # Let's default to a safe number like 4 or os.cpu_count() // 2
    max_workers = max(1, os.cpu_count() // 2)
    
    # Pass thresholds to worker initializer
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, 
        initializer=init_worker,
        initargs=(pitch_th, symmetry_th)
    ) as executor:
        # Map returns results in order
        results = executor.map(process_single_image, tasks)
        
        for i, (status, reason) in enumerate(results):
            if status == 'saved':
                saved += 1
            elif status == 'skipped':
                skipped += 1
                # スキップ理由をログに出力（最初の10件または特定のエラーのみ詳細表示など調整可能）
                # ここでは全て出すと多すぎるかもしれないが、デバッグのため出す
                src_file = tasks[i][0]
                logger.info(f"Skipped {os.path.basename(src_file)}: {reason}")
            else:
                errors += 1
                src_file = tasks[i][0]
                logger.error(f"Error {os.path.basename(src_file)}: {reason}")
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{total} images...")

    logger.info(f"Processing complete for {src_dir}: Total={total}, Saved={saved}, Skipped={skipped}, Errors={errors}")
    return total, saved, skipped

def main():
    parser = argparse.ArgumentParser(description="Multitask Preprocessing with Face Filtering")
    parser.add_argument("--train_dir", default=DEFAULT_TRAIN_DIR, help="Source training directory")
    parser.add_argument("--val_dir", default=DEFAULT_VALIDATION_DIR, help="Source validation directory")
    parser.add_argument("--out_dir", default=DEFAULT_PREPRO_DIR, help="Output directory")
    parser.add_argument("--thresh", type=float, default=DEFAULT_THRESH_RATIO, help="Threshold ratio for filtering")
    parser.add_argument("--pitch_percentile", type=int, default=PITCH_FILTER_PERCENTILE, help="Pitch filter percentile (0-100)")
    parser.add_argument("--symmetry_percentile", type=int, default=SYMMETRY_FILTER_PERCENTILE, help="Symmetry filter percentile (0-100)")
    
    args = parser.parse_args()

    # Use args (which default to constants if not provided)
    train_dir = args.train_dir
    val_dir = args.val_dir
    prepro_dir = args.out_dir
    thresh = args.thresh
    
    prepro_train = os.path.join(prepro_dir, "train")
    prepro_valid = os.path.join(prepro_dir, "validation")

    try:
        if os.path.exists(prepro_dir):
            logger.info(f"Deleting existing '{prepro_dir}' folder...")
            shutil.rmtree(prepro_dir)
            logger.info(f"Deleted '{prepro_dir}' folder.")

        logger.info(f"Starting preprocessing with threshold={thresh}...")
        
        # Calculate pose thresholds by sampling the training dataset
        logger.info("=" * 60)
        logger.info("Filtering configuration:")
        logger.info(f"  Pitch filter: Top {args.pitch_percentile}% will be filtered")
        logger.info(f"  Symmetry filter: Top {args.symmetry_percentile}% will be filtered")
        logger.info("=" * 60)
        
        pitch_th, symmetry_th = calculate_thresholds(
            train_dir, 
            sample_size=SAMPLE_SIZE,
            pitch_percentile=args.pitch_percentile,
            symmetry_percentile=args.symmetry_percentile
        )
        
        logger.info("=" * 60)
        logger.info(f"Train Source: {train_dir} -> {prepro_train}")
        logger.info(f"Val Source:   {val_dir} -> {prepro_valid}")

        process_folder(train_dir, prepro_train, thresh, pitch_th, symmetry_th)
        process_folder(val_dir, prepro_valid, thresh, pitch_th, symmetry_th)
        
        logger.info("All processing complete.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    # Windows specific fix for multiprocessing
    # multiprocessing.freeze_support() # Not strictly needed for script execution but good practice
    main()
