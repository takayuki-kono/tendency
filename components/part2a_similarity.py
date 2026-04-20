# Part 2a: Similarity Check (Argument-based)
# 類似判別は「最小 face_size に揃えた複製」で実施し、解像度差による見逃しを防ぐ。
import os
import re
import cv2
import numpy as np
import logging
import shutil
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
import sys

import argparse

# part1 保存時の 224x224 とファイル名 sz* に合わせる
NORM_SIZE = 224

# --- Globals ---
METRIC = 'cosine'
DEFAULT_DEDUPLICATION_TOLERANCE = 0.25 # Tight tolerance for duplicates (cosine distance)
DEFAULT_MIN_SAMPLES = 2
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'log_part2a.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
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

def parse_face_size_from_basename(basename):
    """ファイル名の sz* から face_size (px) を取得。例: *_sz200.jpg -> 200"""
    m = re.search(r"sz(\d+)", basename, re.IGNORECASE)
    return int(m.group(1)) if m else None

# --- Function ---
def find_similar_images(input_dir, physical_delete, eps=DEFAULT_DEDUPLICATION_TOLERANCE, min_samples=DEFAULT_MIN_SAMPLES):
    """input_dir 内の画像から類似画像を検出し、重複を deleted へ移動する。
    解像度差をなくすため、最小 face_size に揃えた複製で embedding を取得して類似判別する。
    類似ペアでは解像度の低い元画像を削除し、複製はすべて削除する。
    """
    logger.info(f"Starting similarity search with InsightFace embeddings (face_size-normalized). eps={eps}, min_samples={min_samples}")
    
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("FaceAnalysis initialized.")
    except Exception as e:
        logger.error(f"FaceAnalysis initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # input_dir を直接処理（サブディレクトリは探さない）
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        logger.warning(f"Input directory not found or is empty: {input_dir}. Skipping.")
        return

    # deleted ディレクトリは親ディレクトリに作成（論理削除の場合のみ）
    parent_dir = os.path.dirname(input_dir)
    deleted_dir = os.path.join(parent_dir, "deleted_duplicates")
    if not physical_delete and not os.path.exists(deleted_dir):
        os.makedirs(deleted_dir)

    # Filter for standard image extensions
    image_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(input_dir, f))
    ]
    logger.info(f"Found {len(image_files)} images in input directory.")

    if not image_files:
        logger.warning("No images found in processed directory.")
        return

    # face_size をファイル名から取得し、最小値を求める
    items = []  # (original_path, face_size or None)
    for p in image_files:
        fs = parse_face_size_from_basename(os.path.basename(p))
        items.append((p, fs))
    parsed_sizes = [fs for _, fs in items if fs is not None]
    min_face_size = min(parsed_sizes) if parsed_sizes else None

    # 最小 face_size に揃えた複製を作成し、複製パスで embedding 用リストを構築
    temp_dir = os.path.join(input_dir, "_similarity_norm")
    temp_paths_to_delete = []
    path_for_embedding = []   # 実際に embedding に使うパス（複製 or 元）
    embed_index_to_original = []  # path_for_embedding[i] に対応する元パス
    embed_index_to_face_size = []  # 元画像の face_size（削除時は解像度高い方を残すため）

    if min_face_size is not None:
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Normalizing to min face_size={min_face_size} for similarity comparison.")

    for idx, (orig_path, face_size) in enumerate(items):
        bgr = imread_safe(orig_path)
        if bgr is None:
            continue
        if min_face_size is None or face_size is None or face_size <= 0:
            path_for_embedding.append(orig_path)
            embed_index_to_original.append(orig_path)
            embed_index_to_face_size.append(face_size if face_size is not None else 0)
            continue
        scale = min_face_size / face_size
        new_side = max(1, int(NORM_SIZE * scale))
        resized = cv2.resize(bgr, (new_side, new_side))
        base = os.path.basename(orig_path)
        name, ext = os.path.splitext(base)
        temp_path = os.path.join(temp_dir, f"{idx}_{name}{ext}")
        try:
            if cv2.imwrite(temp_path, resized):
                path_for_embedding.append(temp_path)
                temp_paths_to_delete.append(temp_path)
                embed_index_to_original.append(orig_path)
                embed_index_to_face_size.append(face_size)
            else:
                path_for_embedding.append(orig_path)
                embed_index_to_original.append(orig_path)
                embed_index_to_face_size.append(face_size)
        except Exception as e:
            logger.warning(f"Failed to write temp copy for {base}: {e}, using original.")
            path_for_embedding.append(orig_path)
            embed_index_to_original.append(orig_path)
            embed_index_to_face_size.append(face_size)

    # --- Get Embeddings (正規化済み複製 or 元画像で取得) ---
    encodings = []
    valid_indices = []
    for i, embed_path in enumerate(path_for_embedding):
        try:
            bgr_image = imread_safe(embed_path)
            if bgr_image is None:
                continue
            faces = app.get(bgr_image)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                encodings.append(face.embedding)
                valid_indices.append(i)
            else:
                logger.warning(f"No face found: {os.path.basename(embed_path)}")
        except Exception as e:
            logger.error(f"Error processing {embed_path}: {e}")

    if len(encodings) < min_samples:
        logger.info("Not enough faces for clustering.")
        _cleanup_temp(temp_dir, temp_paths_to_delete)
        return

    # --- Clustering ---
    logger.info(f"Clustering with DBSCAN, eps={eps}, min_samples={min_samples}")
    clustering = DBSCAN(metric=METRIC, eps=eps, min_samples=min_samples).fit(np.array(encodings))
    labels = clustering.labels_

    clusters = {}
    for j, label in enumerate(labels):
        if label != -1:
            i = valid_indices[j]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

    logger.info(f"Found {len(clusters)} duplicate groups.")

    # --- Remove Duplicates: クラスタ内では解像度(face_size)が高い方を残し、低い元画像を削除 ---
    for label, indices in clusters.items():
        originals = [embed_index_to_original[i] for i in indices]
        face_sizes = [embed_index_to_face_size[i] if i < len(embed_index_to_face_size) else 0 for i in indices]
        # face_size 降順（解像度高い順）、同値ならファイルサイズで
        combined = list(zip(originals, face_sizes))
        combined.sort(key=lambda x: (x[1], os.path.getsize(x[0]) if os.path.exists(x[0]) else 0), reverse=True)
        keep_one = combined[0][0]
        to_remove = [x[0] for x in combined[1:]]
        logger.info(f"Cluster {label}: Keeping {os.path.basename(keep_one)} (face_size={combined[0][1]}), removing {len(to_remove)} duplicates.")

        for img_path in to_remove:
            try:
                if os.path.exists(img_path):
                    if physical_delete:
                        os.remove(img_path)
                        logger.info(f"Deleted: {os.path.basename(img_path)}")
                    else:
                        dst = os.path.join(deleted_dir, os.path.basename(img_path))
                        if os.path.exists(dst):
                            os.remove(dst)
                        shutil.move(img_path, dst)
                        logger.info(f"Moved to deleted: {os.path.basename(img_path)}")
            except Exception as e:
                logger.error(f"Error removing duplicate {img_path}: {e}")

    # 複製画像を全削除
    _cleanup_temp(temp_dir, temp_paths_to_delete)

def _cleanup_temp(temp_dir, temp_paths_to_delete):
    """一時複製ファイルとディレクトリを削除"""
    for p in temp_paths_to_delete:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception as e:
            logger.warning(f"Failed to remove temp copy {p}: {e}")
    try:
        if os.path.isdir(temp_dir):
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
            else:
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to remove temp dir {temp_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Part 2a: Deduplication")
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("--physical_delete", action="store_true", help="Enable physical deletion (default: False)")
    parser.add_argument("--eps", type=float, default=DEFAULT_DEDUPLICATION_TOLERANCE,
                        help=f"DBSCAN eps (cosine distance). default={DEFAULT_DEDUPLICATION_TOLERANCE}")
    parser.add_argument("--min_samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        help=f"DBSCAN min_samples. default={DEFAULT_MIN_SAMPLES}")
    args = parser.parse_args()

    logger.info(f"Part 2a starting for directory: {args.input_dir}, physical_delete: {args.physical_delete}, eps={args.eps}, min_samples={args.min_samples}")
    try:
        find_similar_images(args.input_dir, args.physical_delete, eps=args.eps, min_samples=args.min_samples)
        logger.info("Part 2a finished.")
    except Exception as e:
        logger.error(f"Fatal error in Part 2a: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
