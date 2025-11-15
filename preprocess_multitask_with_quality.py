import os
import cv2
import logging
import shutil
import numpy as np
from insightface.app import FaceAnalysis

# 簡潔ログ
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", encoding='utf-8', force=True)
logger = logging.getLogger(__name__)

# 入出力ディレクトリ
TRAIN_DIR = "train"
VALIDATION_DIR = "validation"
PREPRO_DIR = "preprocessed_multitask" # 出力先
PREPRO_TRAIN = os.path.join(PREPRO_DIR, "train")
PREPRO_VALID = os.path.join(PREPRO_DIR, "validation")

# ===== 品質フィルタリング設定 =====
TOP_PERCENT_TO_KEEP = 50      # 最終的に保存する割合（2段階フィルタリング適用）

# 2段階フィルタリングの各段階での割合
# 例: 50%残したい → 各段階で sqrt(50) ≈ 70.7% → 70.7% × 70.7% ≈ 50%
import math
STAGE_PERCENT = math.sqrt(TOP_PERCENT_TO_KEEP) * 10  # 50 → 70.7%

# 品質スコアの重み（合計1.0にする）
WEIGHT_DET_SCORE = 0.5        # 顔検出信頼度の重み
WEIGHT_SHARPNESS = 0.3        # 鮮明度の重み
WEIGHT_SIZE = 0.2             # 顔サイズの重み

# =====================================

# InsightFace 初期化
face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(320, 320))

def calculate_frontal_score(face):
    """
    正面向き度スコアを計算（0-1の範囲、1が完全に正面）
    yaw, pitchが0に近いほど高スコア
    """
    if not hasattr(face, 'pose') or face.pose is None:
        return 0.5  # poseデータがない場合は中間値

    pitch, yaw, roll = face.pose

    # 角度の絶対値の合計が小さいほど正面向き
    # 最大角度を60度と仮定（それ以上は0スコア）
    angle_sum = abs(yaw) + abs(pitch)

    # スコア計算: 0度で1.0, 60度以上で0.0
    score = max(0.0, 1.0 - angle_sum / 60.0)

    return score

def calculate_sharpness(img):
    """
    画像の鮮明度を計算（Laplacian variance）
    値が高いほど鮮明、低いほどぼやけている
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_quality_score(face, img):
    """
    画像品質スコアを計算（0-1の範囲）

    Returns:
        score: 総合スコア (0-1)
        details: 各要素の詳細
    """
    h, w = img.shape[:2]

    # 1. 顔検出の信頼度 (0-1)
    det_score = face.det_score if hasattr(face, 'det_score') else 0.5

    # 2. 鮮明度
    sharpness = calculate_sharpness(img)
    # 正規化（経験的に100-1000の範囲）
    # 100以下はかなりぼやけ、1000以上はかなり鮮明
    sharpness_norm = np.clip((sharpness - 100) / 900, 0, 1)

    # 3. 顔のサイズ比率
    x1, y1, x2, y2 = face.bbox
    face_area = (x2 - x1) * (y2 - y1)
    img_area = h * w
    size_ratio = face_area / img_area

    # 顔サイズのスコアリング（理想は10-50%程度）
    if size_ratio < 0.05:  # 小さすぎる（遠すぎる）
        size_score = size_ratio / 0.05
    elif size_ratio > 0.7:  # 大きすぎる（近すぎる）
        size_score = (1.0 - size_ratio) / 0.3
    else:  # 適切なサイズ
        size_score = 1.0

    size_score = np.clip(size_score, 0, 1)

    # 総合スコア（重み付き平均）
    score = (
        det_score * WEIGHT_DET_SCORE +
        sharpness_norm * WEIGHT_SHARPNESS +
        size_score * WEIGHT_SIZE
    )

    return score, {
        'det_score': det_score,
        'sharpness': sharpness,
        'sharpness_norm': sharpness_norm,
        'size_ratio': size_ratio,
        'size_score': size_score
    }

def process_folder_with_quality_filter(src_dir, dst_dir):
    """
    フォルダを処理し、2段階フィルタリング後の画像を保存する

    処理フロー:
    1. 全画像を走査して基本チェック（顔検出、ランドマーク）
    2. 正面向き度スコアで上位sqrt(TOP_PERCENT)%をフィルタ
    3. その中で品質スコアで上位sqrt(TOP_PERCENT)%を保存
    """
    os.makedirs(dst_dir, exist_ok=True)

    # ===== 第1パス: 全画像を走査して正面向き度スコアを計算 =====
    logger.info(f"Starting to scan {src_dir}...")
    all_images = []
    total_scanned = 0
    no_face = 0
    failed_load = 0

    for root, _, files in os.walk(src_dir):
        # src_dir自体はスキップ
        if root == src_dir:
            continue

        # rootからsrc_dirを除いた相対パスを取得
        rel_path = os.path.relpath(root, src_dir)
        label_name = rel_path.split(os.sep)[0]

        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            total_scanned += 1
            src_path = os.path.join(root, fn)

            # 画像読み込み
            img = cv2.imread(src_path)
            if img is None:
                logger.warning(f"Failed to read: {src_path}")
                failed_load += 1
                continue

            h, w = img.shape[:2]

            # 顔検出
            faces = face_app.get(img)
            if not faces:
                no_face += 1
                continue

            face = faces[0]

            # ===== 正面向き度スコア計算 =====
            frontal_score = calculate_frontal_score(face)

            all_images.append({
                'src_path': src_path,
                'label_name': label_name,
                'filename': fn,
                'face': face,
                'img': img,
                'frontal_score': frontal_score
            })

    logger.info(f"\n=== Stage 1: Scanning Summary ===")
    logger.info(f"Total scanned: {total_scanned}")
    logger.info(f"Failed to load: {failed_load}")
    logger.info(f"No face detected: {no_face}")
    logger.info(f"Valid images with face: {len(all_images)}")

    if not all_images:
        logger.warning("No valid images found!")
        return 0, 0, 0

    # ===== 第2パス: 正面向き度で上位STAGE_PERCENT%をフィルタ =====
    all_images.sort(key=lambda x: x['frontal_score'], reverse=True)

    num_frontal = int(len(all_images) * STAGE_PERCENT / 100)
    frontal_filtered = all_images[:num_frontal]

    logger.info(f"\n=== Stage 2: Frontal Score Filtering ===")
    logger.info(f"Keeping top {STAGE_PERCENT:.1f}% by frontal score ({num_frontal} images)")
    logger.info(f"Frontal score range: {frontal_filtered[-1]['frontal_score']:.4f} - {frontal_filtered[0]['frontal_score']:.4f}")

    frontal_scores = [img['frontal_score'] for img in all_images]
    logger.info(f"Frontal score stats: mean={np.mean(frontal_scores):.4f}, std={np.std(frontal_scores):.4f}")

    # ===== 第3パス: 品質スコアを計算 =====
    logger.info(f"\n=== Stage 3: Quality Score Calculation ===")
    for img_info in frontal_filtered:
        quality_score, details = calculate_quality_score(img_info['face'], img_info['img'])
        img_info['quality_score'] = quality_score
        img_info['quality_details'] = details
        # メモリ節約のため画像とfaceオブジェクトを削除
        del img_info['img']
        del img_info['face']

    # ===== 第4パス: 品質スコアで上位STAGE_PERCENT%をフィルタ =====
    frontal_filtered.sort(key=lambda x: x['quality_score'], reverse=True)

    num_to_keep = int(len(frontal_filtered) * STAGE_PERCENT / 100)
    final_images = frontal_filtered[:num_to_keep]

    logger.info(f"\n=== Stage 4: Quality Score Filtering ===")
    logger.info(f"Keeping top {STAGE_PERCENT:.1f}% by quality score ({num_to_keep} images)")
    logger.info(f"Quality score range: {final_images[-1]['quality_score']:.4f} - {final_images[0]['quality_score']:.4f}")

    quality_scores = [img['quality_score'] for img in frontal_filtered]
    logger.info(f"Quality score stats: mean={np.mean(quality_scores):.4f}, std={np.std(quality_scores):.4f}")

    logger.info(f"\nFinal retention rate: {num_to_keep}/{total_scanned} = {100*num_to_keep/total_scanned:.1f}%")

    # ===== 第5パス: 画像を保存 =====
    logger.info(f"\n=== Stage 5: Saving Images ===")
    saved = 0

    for img_info in final_images:
        dst_label_path = os.path.join(dst_dir, img_info['label_name'])
        os.makedirs(dst_label_path, exist_ok=True)

        dst_path = os.path.join(dst_label_path, img_info['filename'])

        try:
            shutil.copy(img_info['src_path'], dst_path)
            saved += 1

            if saved <= 5 or saved % 100 == 0:
                details = img_info['quality_details']
                logger.info(f"Saved [{saved}/{num_to_keep}]: {img_info['filename']} "
                           f"(frontal={img_info['frontal_score']:.4f}, quality={img_info['quality_score']:.4f}, "
                           f"det={details['det_score']:.3f}, sharp={details['sharpness']:.1f})")
        except Exception as e:
            logger.warning(f"Failed to save {dst_path}: {e}")

    logger.info(f"\n=== Processing Complete ===")
    logger.info(f"Total scanned: {total_scanned}")
    logger.info(f"Saved: {saved}")
    logger.info(f"Removed: {total_scanned - saved}")

    return total_scanned, saved, total_scanned - saved

def main():
    try:
        logger.info("="*60)
        logger.info("Starting preprocessing with 2-stage filtering...")
        logger.info(f"Configuration:")
        logger.info(f"  - Target retention rate: {TOP_PERCENT_TO_KEEP}%")
        logger.info(f"  - Stage filtering rate: {STAGE_PERCENT:.1f}% (each stage)")
        logger.info(f"  - Calculated retention: {STAGE_PERCENT:.1f}% × {STAGE_PERCENT:.1f}% ≈ {(STAGE_PERCENT/100)**2*100:.1f}%")
        logger.info(f"  - Quality score weights: det={WEIGHT_DET_SCORE}, sharp={WEIGHT_SHARPNESS}, size={WEIGHT_SIZE}")
        logger.info("="*60)

        if os.path.exists(PREPRO_DIR):
            logger.info(f"Deleting existing '{PREPRO_DIR}' folder...")
            shutil.rmtree(PREPRO_DIR)
            logger.info(f"Deleted '{PREPRO_DIR}' folder.")

        logger.info("\n" + "="*60)
        logger.info("Processing TRAINING data...")
        logger.info("="*60)
        process_folder_with_quality_filter(TRAIN_DIR, PREPRO_TRAIN)

        logger.info("\n" + "="*60)
        logger.info("Processing VALIDATION data...")
        logger.info("="*60)
        process_folder_with_quality_filter(VALIDATION_DIR, PREPRO_VALID)

        logger.info("\n" + "="*60)
        logger.info("All processing complete.")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
