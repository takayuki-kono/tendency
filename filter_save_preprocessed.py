import os
import cv2
import logging
import numpy as np
import shutil
from insightface.app import FaceAnalysis

# 簡潔ログ
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", encoding='utf-8', force=True)
logger = logging.getLogger(__name__)

# 入出力ディレクトリ
TRAIN_DIR = "train"
VALIDATION_DIR = "validation"
PREPRO_DIR = "preprocessed"
PREPRO_TRAIN = os.path.join(PREPRO_DIR, "train")
PREPRO_VALID = os.path.join(PREPRO_DIR, "validation")

# チェック閾値（画像幅/高さに対する割合）
THRESH_RATIO = 0.1  # 10%

# InsightFace 初期化
face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(320, 320))

# InsightFaceのランドマークインデックス
# Mediapipe 454 (左頬) -> InsightFace 28
# Mediapipe 234 (右頬) -> InsightFace 12
LEFT_CHEEK_IDX = 28
RIGHT_CHEEK_IDX = 12

def extract_landmarks(img_bgr):
    if img_bgr is None:
        return None
    faces = face_app.get(img_bgr)
    if not faces:
        return None
    # 最初の顔のランドマークを返す
    return faces[0].landmark_2d_106

def process_folder(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    total = 0
    saved = 0
    skipped = 0
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            total += 1
            src_path = os.path.join(root, fn)
            dst_subdir = os.path.join(dst_dir, rel) if rel != "." else dst_dir
            os.makedirs(dst_subdir, exist_ok=True)
            dst_path = os.path.join(dst_subdir, fn)

            img = cv2.imread(src_path)
            if img is None:
                logger.warning(f"読み込み失敗: {src_path}")
                skipped += 1
                continue

            h, w = img.shape[:2]
            landmarks = extract_landmarks(img)
            if landmarks is None or max(LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX) >= len(landmarks):
                logger.info(f"ランドマーク未検出でスキップ: {src_path}")
                skipped += 1
                continue

            # 座標を取得
            lx, ly = landmarks[LEFT_CHEEK_IDX]
            rx, ry = landmarks[RIGHT_CHEEK_IDX]

            # x座標フィルタリング
            if rx >= THRESH_RATIO * w:
                logger.info(f"右頬が左端から離れているためスキップ: {src_path} (rx={rx} >= {THRESH_RATIO*w:.1f})")
                skipped += 1
                continue

            if lx <= (1.0 - THRESH_RATIO) * w:
                logger.info(f"左頬が右端から離れているためスキップ: {src_path} (lx={lx} <= {(1.0-THRESH_RATIO)*w:.1f})")
                skipped += 1
                continue

            # y差チェック
            if abs(ry - ly) >= THRESH_RATIO * h:
                logger.info(f"y差が大きいのでスキップ: {src_path} ({abs(ry-ly)} >= {THRESH_RATIO*h:.1f})")
                skipped += 1
                continue

            # preprocessed に保存
            try:
                shutil.copy(src_path, dst_path)
                saved += 1
            except Exception as e:
                logger.warning(f"保存失敗 {dst_path}: {e}")
                skipped += 1

    logger.info(f"処理完了: {src_dir} -> {dst_dir} (total={total}, saved={saved}, skipped={skipped})")
    return total, saved, skipped

def main():
    try:
        if os.path.exists(PREPRO_DIR):
            logger.info(f"既存の '{PREPRO_DIR}' フォルダを削除します...")
            shutil.rmtree(PREPRO_DIR)
            logger.info(f"'{PREPRO_DIR}' フォルダを削除しました。")

        logger.info("開始: InsightFaceによるフィルタリング")
        process_folder(TRAIN_DIR, PREPRO_TRAIN)
        process_folder(VALIDATION_DIR, PREPRO_VALID)
        logger.info("全処理完了")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)

if __name__ == "__main__":
    main()
