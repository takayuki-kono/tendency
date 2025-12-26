"""
前処理済みデータに対して類似画像をフィルタリングするスクリプト
preprocessed_multitask/train と validation からほぼ同じ顔画像を削除
"""
import os
import cv2
import numpy as np
import logging
import shutil
import argparse
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis

# --- 設定 ---
DEDUPLICATION_TOLERANCE = 0.25  # コサイン距離閾値（小さいほど厳しい）
MIN_SAMPLES = 2
METRIC = 'cosine'

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

def collect_images(base_dir):
    """指定ディレクトリ内の全画像パスを収集"""
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, f))
    return image_paths

def get_embeddings(app, image_paths):
    """画像からInsightFace埋め込みを抽出"""
    embeddings = []
    valid_paths = []
    
    logger.info(f"{len(image_paths)}枚の画像から埋め込みを抽出中...")
    
    for i, img_path in enumerate(image_paths):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            faces = app.get(img)
            if faces and len(faces) > 0:
                embeddings.append(faces[0].embedding)
                valid_paths.append(img_path)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  {i + 1}/{len(image_paths)} 処理完了...")
                
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            continue
    
    logger.info(f"有効な埋め込み数: {len(embeddings)}")
    return np.array(embeddings), valid_paths

def get_source_prefix(filepath):
    """ファイル名から頭4桁のソースIDを取得"""
    basename = os.path.basename(filepath)
    # 頭4桁を取得（例: 1433_001_0.jpg -> 1433）
    if len(basename) >= 4:
        return basename[:4]
    return basename

def find_and_remove_duplicates(base_dir, tolerance=0.26, dry_run=False):
    """類似画像を検出して削除（またはバックアップ）- 同一ソース内のみ"""
    
    # InsightFace初期化
    logger.info("InsightFaceを初期化中...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    
    # 画像収集
    image_paths = collect_images(base_dir)
    logger.info(f"検出された画像数: {len(image_paths)}")
    
    if len(image_paths) < MIN_SAMPLES:
        logger.info("画像数が少なすぎます。スキップします。")
        return 0
    
    # ソースIDごとにグループ化
    source_groups = {}
    for path in image_paths:
        prefix = get_source_prefix(path)
        if prefix not in source_groups:
            source_groups[prefix] = []
        source_groups[prefix].append(path)
    
    logger.info(f"ソースグループ数: {len(source_groups)}")
    
    # 削除用ディレクトリ
    deleted_dir = os.path.join(base_dir, "_deleted_similar")
    if not dry_run:
        os.makedirs(deleted_dir, exist_ok=True)
    
    total_removed = 0
    cluster_id = 0
    
    # 各ソースグループ内で類似判定
    for prefix, group_paths in source_groups.items():
        if len(group_paths) < MIN_SAMPLES:
            continue
        
        # 埋め込み抽出
        embeddings, valid_paths = get_embeddings(app, group_paths)
        
        if len(embeddings) < MIN_SAMPLES:
            continue
        
        # クラスタリング
        clustering = DBSCAN(metric=METRIC, eps=tolerance, min_samples=MIN_SAMPLES).fit(embeddings)
        labels = clustering.labels_
        
        # クラスタをグループ化
        clusters = {}
        for i, label in enumerate(labels):
            if label != -1:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_paths[i])
        
        # 重複削除（このソースグループ内）
        for label, group in clusters.items():
            keep = group[0]
            to_remove = group[1:]
            
            logger.info(f"[{prefix}] クラスタ {cluster_id}: 保持={os.path.basename(keep)}, 削除数={len(to_remove)}")
            
            for img_path in to_remove:
                if dry_run:
                    logger.info(f"  [DRY-RUN] 削除対象: {os.path.basename(img_path)}")
                else:
                    try:
                        dest = os.path.join(deleted_dir, os.path.basename(img_path))
                        if os.path.exists(dest):
                            base, ext = os.path.splitext(os.path.basename(img_path))
                            dest = os.path.join(deleted_dir, f"{base}_{cluster_id}{ext}")
                        shutil.move(img_path, dest)
                        logger.info(f"  削除: {os.path.basename(img_path)}")
                    except Exception as e:
                        logger.error(f"  移動失敗: {img_path}: {e}")
                total_removed += 1
            cluster_id += 1
    
    logger.info(f"削除された画像数: {total_removed}")
    return total_removed

def main():
    parser = argparse.ArgumentParser(description="前処理済みデータから類似画像を削除")
    parser.add_argument("--dir", default="preprocessed_multitask", help="対象ディレクトリ")
    parser.add_argument("--tolerance", type=float, default=DEDUPLICATION_TOLERANCE, 
                        help=f"類似度閾値 (default: {DEDUPLICATION_TOLERANCE})")
    parser.add_argument("--dry_run", action="store_true", help="実際には削除せずログのみ出力")
    parser.add_argument("--train_only", action="store_true", help="trainのみ処理")
    parser.add_argument("--val_only", action="store_true", help="validationのみ処理")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("類似画像フィルタリング開始")
    logger.info(f"対象ディレクトリ: {args.dir}")
    logger.info(f"類似度閾値: {args.tolerance}")
    logger.info(f"Dry-run: {args.dry_run}")
    logger.info("=" * 60)
    
    total_removed = 0
    
    if not args.val_only:
        train_dir = os.path.join(args.dir, "train")
        if os.path.exists(train_dir):
            logger.info(f"\n--- Processing {train_dir} ---")
            total_removed += find_and_remove_duplicates(train_dir, args.tolerance, args.dry_run)
    
    if not args.train_only:
        val_dir = os.path.join(args.dir, "validation")
        if os.path.exists(val_dir):
            logger.info(f"\n--- Processing {val_dir} ---")
            total_removed += find_and_remove_duplicates(val_dir, args.tolerance, args.dry_run)
    
    logger.info("=" * 60)
    logger.info(f"完了！合計削除数: {total_removed}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
