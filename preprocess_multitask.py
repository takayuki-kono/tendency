import os
import cv2
import logging
import shutil
from insightface.app import FaceAnalysis

# 簡潔ログ
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", encoding='utf-8', force=True)
logger = logging.getLogger(__name__)

# 入出力ディレクトリ
# 'train' と 'validation' フォルダには 'adfh', 'adfi' のような名前のサブフォルダがあると想定
TRAIN_DIR = "train"
VALIDATION_DIR = "validation"
PREPRO_DIR = "preprocessed_multitask" # 出力先
PREPRO_TRAIN = os.path.join(PREPRO_DIR, "train")
PREPRO_VALID = os.path.join(PREPRO_DIR, "validation")

# チェック閾値（画像幅/高さに対する割合）
# 0.125 50% 0.47 0.49
# 0.19 66% 0.44 0.31
# 0.25 75% 0.43 0.44
THRESH_RATIO = 0.19

# InsightFace 初期化
face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(320, 320))

# InsightFaceのランドマークインデックス
LEFT_CHEEK_IDX = 28
RIGHT_CHEEK_IDX = 12

def extract_landmarks(img_bgr):
    """画像からランドマークを抽出する"""
    if img_bgr is None:
        return None
    faces = face_app.get(img_bgr)
    if not faces:
        return None
    return faces[0].landmark_2d_106

def process_folder(src_dir, dst_dir):
    """
    フォルダを再帰的に処理し、フィルタリング後の画像を保存する。
    os.walkを使用して深い階層にある画像も対象とする。
    """
    os.makedirs(dst_dir, exist_ok=True)
    total = 0
    saved = 0
    skipped = 0
    
    logger.info(f"Starting to walk through {src_dir}...")

    for root, _, files in os.walk(src_dir):
        # src_dir自体はスキップ
        if root == src_dir:
            continue

        # rootからsrc_dirを除いた相対パスを取得 (例: adfh/harukaAyase)
        rel_path = os.path.relpath(root, src_dir)
        
        # 最初の部分をラベル名として取得 (例: adfh)
        label_name = rel_path.split(os.sep)[0]
        
        # 出力先のラベルフォルダパスを決定
        dst_label_path = os.path.join(dst_dir, label_name)
        os.makedirs(dst_label_path, exist_ok=True)

        # 各画像ファイルを処理
        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            
            total += 1
            src_path = os.path.join(root, fn)
            # 出力ファイル名は、衝突を避けるために元のフォルダ構造を一部含めることも検討できるが、
            # ここではシンプルに同じファイル名でコピーする。
            dst_path = os.path.join(dst_label_path, fn)

            img = cv2.imread(src_path)
            if img is None:
                logger.warning(f"Failed to read: {src_path}")
                skipped += 1
                continue

            h, w = img.shape[:2]
            landmarks = extract_landmarks(img)
            if landmarks is None or max(LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX) >= len(landmarks):
                logger.info(f"Skipping due to no landmarks: {src_path}")
                skipped += 1
                continue

            # ランドマーク座標を取得
            lx, ly = landmarks[LEFT_CHEEK_IDX]
            rx, ry = landmarks[RIGHT_CHEEK_IDX]

            # x座標フィルタリング
            if rx >= THRESH_RATIO * w:
                logger.info(f"Skipping because right cheek is far from left edge: {src_path} (rx={rx} >= {THRESH_RATIO*w:.1f})")
                skipped += 1
                continue

            if lx <= (1.0 - THRESH_RATIO) * w:
                logger.info(f"Skipping because left cheek is far from right edge: {src_path} (lx={lx} <= {(1.0-THRESH_RATIO)*w:.1f})")
                skipped += 1
                continue

            # y差チェック
            if abs(ry - ly) >= THRESH_RATIO * h:
                logger.info(f"Skipping due to large y-difference: {src_path} ({abs(ry-ly)} >= {THRESH_RATIO*h:.1f})")
                skipped += 1
                continue

            # フィルタを通過した画像をコピー
            try:
                shutil.copy(src_path, dst_path)
                saved += 1
            except Exception as e:
                logger.warning(f"Failed to save {dst_path}: {e}")
                skipped += 1
    
    logger.info(f"Processing complete for {src_dir}: Total={total}, Saved={saved}, Skipped={skipped}")
    return total, saved, skipped

def main():
    try:
        if os.path.exists(PREPRO_DIR):
            logger.info(f"Deleting existing '{PREPRO_DIR}' folder...")
            shutil.rmtree(PREPRO_DIR)
            logger.info(f"Deleted '{PREPRO_DIR}' folder.")

        logger.info("Starting preprocessing and filtering for multitask...")
        process_folder(TRAIN_DIR, PREPRO_TRAIN)
        process_folder(VALIDATION_DIR, PREPRO_VALID)
        logger.info("All processing complete.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
