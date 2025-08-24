import os
import cv2
import logging
import numpy as np
import mediapipe as mp
import shutil

# 簡潔ログ
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 入出力ディレクトリ（変更しない場合はこのまま）
TRAIN_DIR = "train"
VALIDATION_DIR = "validation"
PREPRO_DIR = "preprocessed"
PREPRO_TRAIN = os.path.join(PREPRO_DIR, "train")
PREPRO_VALID = os.path.join(PREPRO_DIR, "validation")

# チェック閾値（画像幅/高さに対する割合）
THRESH_RATIO = 0.1  # 5%

# MediaPipe FaceMesh 初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# 顔の頬に相当する FaceMesh の代表インデックス（一般的に使われるペア）
LEFT_CHEEK_IDX = 454   # 左頬（画像左寄り）
RIGHT_CHEEK_IDX = 234   # 右頬（画像右寄り）


def extract_landmarks(img_bgr):
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res or not res.multi_face_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    return pts


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
            lm = extract_landmarks(img)
            if lm is None or max(LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX) >= len(lm):
                logger.info(f"ランドマーク未検出でスキップ: {src_path}")
                skipped += 1
                continue

            lx, ly = lm[LEFT_CHEEK_IDX]
            rx, ry = lm[RIGHT_CHEEK_IDX]

            # 修正: x座標フィルタリング
            # 右頬のx座標が左端から画像幅の5%以上離れている場合スキップ
            if rx >= THRESH_RATIO * w:
                logger.info(f"右頬が左端から離れているためスキップ: {src_path} (rx={rx} >= {THRESH_RATIO*w:.1f})")
                skipped += 1
                continue

            # 左頬のx座標が右端から画像幅の5%以上離れている場合スキップ
            if lx <= (1.0 - THRESH_RATIO) * w:
                logger.info(f"左頬が右端から離れているためスキップ: {src_path} (lx={lx} <= {(1.0-THRESH_RATIO)*w:.1f})")
                skipped += 1
                continue

            # ② y差チェック（画像高さの5%以上離れている → スキップ）
            if abs(ry - ly) >= THRESH_RATIO * h:
                logger.info(f"y差が大きいのでスキップ: {src_path} ({abs(ry-ly)} >= {THRESH_RATIO*h:.1f})")
                skipped += 1
                continue

            # ③ グレースケール変換
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ④ preprocessed に保存（元ファイル名維持）
            try:
                cv2.imwrite(dst_path, img)
                saved += 1
            except Exception as e:
                logger.warning(f"保存失敗 {dst_path}: {e}")
                skipped += 1

    logger.info(f"処理完了: {src_dir} -> {dst_dir} (total={total}, saved={saved}, skipped={skipped})")
    return total, saved, skipped


def main():
    try:
        # 処理前に preprocessed フォルダを削除
        if os.path.exists(PREPRO_DIR):
            logger.info(f"既存の '{PREPRO_DIR}' フォルダを削除します...")
            shutil.rmtree(PREPRO_DIR)
            logger.info(f"'{PREPRO_DIR}' フォルダを削除しました。")

        logger.info("開始: フィルタ + グレースケール保存")
        process_folder(TRAIN_DIR, PREPRO_TRAIN)
        process_folder(VALIDATION_DIR, PREPRO_VALID)
        logger.info("全処理完了")
    finally:
        try:
            face_mesh.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()