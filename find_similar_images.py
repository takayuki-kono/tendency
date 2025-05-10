import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import defaultdict
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # ターミナル出力
        logging.FileHandler('console_log.txt', mode='w')  # ファイル出力
    ],
    force=True
)
logger = logging.getLogger(__name__)

# 設定
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VAL_DIR = 'preprocessed/validation'
BLACK_MAX = 85
GRAY_MIN = 86
GRAY_MAX = 170
WHITE_MIN = 171
COLOR_RATIO_THRESHOLD = 0.2
TILT_THRESHOLD = 0.05
OUTPUT_CSV = 'similar_images_color_tilt_split.csv'
TOP_N = 100
IMG_SIZE = 112
TRIANGLE_LANDMARK_INDICES = [(33, 263), (263, 1), (1, 33)]

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def extract_landmarks(img_path):
    """画像から顔ランドマークを抽出（2D座標のみ）"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Image not loaded: {img_path}")
            return None
        img_rgb = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            coords = [(l.x * IMG_SIZE, l.y * IMG_SIZE) for l in landmarks]
            nose_center = coords[1]
            coords = [(x - nose_center[0], y - nose_center[1]) for x, y in coords]
            return np.array(coords)
        else:
            logger.info(f"No landmarks detected in {img_path}")
            return None
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return None

def compute_tilt(landmarks, indices):
    """2点間の勾配を計算（dy/dx）"""
    try:
        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dx) < 1e-10:
            return float('inf')
        tilt = dy / dx
        return tilt
    except Exception as e:
        logger.error(f"Error computing tilt: {e}")
        return float('inf')

def compute_color_ratios(img_path):
    """画像を4分割（左上、右上、左下、右下）し、各領域の黒・灰・白の割合を計算"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Image not loaded: {img_path}")
            return None, None, None, None, None, None, None, None, None, None, None, None
        left_upper = img[:IMG_SIZE//2, :IMG_SIZE//2]
        right_upper = img[:IMG_SIZE//2, IMG_SIZE//2:]
        left_lower = img[IMG_SIZE//2:, :IMG_SIZE//2]
        right_lower = img[IMG_SIZE//2:, IMG_SIZE//2:]
        total_pixels = left_upper.size
        lu_black_pixels = np.sum(left_upper <= BLACK_MAX)
        lu_gray_pixels = np.sum((left_upper >= GRAY_MIN) & (left_upper <= GRAY_MAX))
        lu_white_pixels = np.sum(left_upper >= WHITE_MIN)
        lu_black_ratio = lu_black_pixels / total_pixels
        lu_gray_ratio = lu_gray_pixels / total_pixels
        lu_white_ratio = lu_white_pixels / total_pixels
        ru_black_pixels = np.sum(right_upper <= BLACK_MAX)
        ru_gray_pixels = np.sum((right_upper >= GRAY_MIN) & (right_upper <= GRAY_MAX))
        ru_white_pixels = np.sum(right_upper >= WHITE_MIN)
        ru_black_ratio = ru_black_pixels / total_pixels
        ru_gray_ratio = ru_gray_pixels / total_pixels
        ru_white_ratio = ru_white_pixels / total_pixels
        ll_black_pixels = np.sum(left_lower <= BLACK_MAX)
        ll_gray_pixels = np.sum((left_lower >= GRAY_MIN) & (left_lower <= GRAY_MAX))
        ll_white_pixels = np.sum(left_lower >= WHITE_MIN)
        ll_black_ratio = ll_black_pixels / total_pixels
        ll_gray_ratio = ll_gray_pixels / total_pixels
        ll_white_ratio = ll_white_pixels / total_pixels
        rl_black_pixels = np.sum(right_lower <= BLACK_MAX)
        rl_gray_pixels = np.sum((right_upper >= GRAY_MIN) & (right_upper <= GRAY_MAX))
        rl_white_pixels = np.sum(right_lower >= WHITE_MIN)
        rl_black_ratio = rl_black_pixels / total_pixels
        rl_gray_ratio = rl_gray_pixels / total_pixels
        rl_white_ratio = rl_white_pixels / total_pixels
        return (lu_black_ratio, lu_gray_ratio, lu_white_ratio,
                ru_black_ratio, ru_gray_ratio, ru_white_ratio,
                ll_black_ratio, ll_gray_ratio, ll_white_ratio,
                rl_black_ratio, rl_gray_ratio, rl_white_ratio)
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return None, None, None, None, None, None, None, None, None, None, None, None

def get_filename_prefix(filename, prefix_length=3):
    """ファイル名の先頭3文字を取得"""
    try:
        prefix = os.path.basename(filename)[:prefix_length]
        return prefix
    except Exception as e:
        logger.error(f"Error getting prefix for {filename}: {e}")
        return ""

def find_similar_images(train_dir, color_ratio_threshold=COLOR_RATIO_THRESHOLD, tilt_threshold=TILT_THRESHOLD, top_n=TOP_N):
    """先頭3文字一致かつ4分割の色割合・三角形傾き差が小さい画像グループを特定（3枚以上、重複なし）"""
    logger.info(f"Starting find_similar_images for {train_dir}")
    print(f"Starting find_similar_images for {train_dir}")
    image_files = []
    for category in ['category1', 'category2']:
        cat_dir = os.path.join(train_dir, category)
        if not os.path.exists(cat_dir):
            logger.info(f"Directory {cat_dir} not found")
            print(f"Directory {cat_dir} not found")
            continue
        for filename in os.listdir(cat_dir):
            img_path = os.path.join(cat_dir, filename)
            image_files.append((img_path, category))
    
    logger.info(f"Found {len(image_files)} images in {train_dir}")
    print(f"Found {len(image_files)} images in {train_dir}")
    
    # ファイル名先頭3文字でグループ化
    prefix_groups = defaultdict(list)
    for img_path, category in image_files:
        prefix = get_filename_prefix(img_path)
        prefix_groups[prefix].append((img_path, category))
    
    for prefix, images in prefix_groups.items():
        logger.info(f"Prefix {prefix}: {len(images)} images")
        print(f"Prefix {prefix}: {len(images)} images")
    
    # 各画像の特徴量を計算
    image_data = {}
    for img_path, category in image_files:
        ratios = compute_color_ratios(img_path)
        landmarks = extract_landmarks(img_path)
        if all(r is not None for r in ratios) and landmarks is not None:
            image_data[img_path] = {
                'category': category,
                'ratios': ratios,
                'landmarks': landmarks
            }
        else:
            logger.info(f"Skipping {img_path} due to invalid ratios or landmarks")
            print(f"Skipping {img_path} due to invalid ratios or landmarks")
    
    logger.info(f"Valid images with features: {len(image_data)}")
    print(f"Valid images with features: {len(image_data)}")
    
    # グループ化
    groups = []
    used_images = set()
    
    for prefix, images in prefix_groups.items():
        if len(images) < 3:
            logger.info(f"Skipping prefix {prefix}: only {len(images)} images")
            print(f"Skipping prefix {prefix}: only {len(images)} images")
            continue
        
        valid_images = [(img_path, cat) for img_path, cat in images if img_path in image_data]
        for i in range(len(valid_images)):
            img1_path, cat1 = valid_images[i]
            if img1_path in used_images:
                continue
            current_group = [(img1_path, cat1)]
            current_diffs = []
            
            for j in range(len(valid_images)):
                if i == j:
                    continue
                img2_path, cat2 = valid_images[j]
                if img2_path in used_images:
                    continue
                
                data1 = image_data[img1_path]
                data2 = image_data[img2_path]
                ratios1 = data1['ratios']
                ratios2 = data2['ratios']
                lm1 = data1['landmarks']
                lm2 = data2['landmarks']
                
                lu_black_diff = abs(ratios1[0] - ratios2[0])
                lu_gray_diff = abs(ratios1[1] - ratios2[1])
                lu_white_diff = abs(ratios1[2] - ratios2[2])
                ru_black_diff = abs(ratios1[3] - ratios2[3])
                ru_gray_diff = abs(ratios1[4] - ratios2[4])
                ru_white_diff = abs(ratios1[5] - ratios2[5])
                ll_black_diff = abs(ratios1[6] - ratios2[6])
                ll_gray_diff = abs(ratios1[7] - ratios2[7])
                ll_white_diff = abs(ratios1[8] - ratios2[8])
                rl_black_diff = abs(ratios1[9] - ratios2[9])
                rl_gray_diff = abs(ratios1[10] - ratios2[10])
                rl_white_diff = abs(ratios1[11] - ratios2[11])
                
                if not (lu_black_diff <= color_ratio_threshold and
                        lu_gray_diff <= color_ratio_threshold and
                        lu_white_diff <= color_ratio_threshold and
                        ru_black_diff <= color_ratio_threshold and
                        ru_gray_diff <= color_ratio_threshold and
                        ru_white_diff <= color_ratio_threshold and
                        ll_black_diff <= color_ratio_threshold and
                        ll_gray_diff <= color_ratio_threshold and
                        ll_white_diff <= color_ratio_threshold and
                        rl_black_diff <= color_ratio_threshold and
                        rl_gray_diff <= color_ratio_threshold and
                        rl_white_diff <= color_ratio_threshold):
                    continue
                
                tilt_diffs = []
                for idx1, idx2 in TRIANGLE_LANDMARK_INDICES:
                    t1 = compute_tilt(lm1, [idx1, idx2])
                    t2 = compute_tilt(lm2, [idx1, idx2])
                    if t1 == float('inf') or t2 == float('inf'):
                        diff = float('inf') if t1 != t2 else 0.0
                    else:
                        diff = abs(t1 - t2)
                    tilt_diffs.append(diff)
                
                if any(diff > tilt_threshold for diff in tilt_diffs):
                    continue
                
                current_group.append((img2_path, cat2))
                current_diffs.append({
                    'left_upper_black_ratio_difference': lu_black_diff,
                    'left_upper_gray_ratio_difference': lu_gray_diff,
                    'left_upper_white_ratio_difference': lu_white_diff,
                    'right_upper_black_ratio_difference': ru_black_diff,
                    'right_upper_gray_ratio_difference': ru_gray_diff,
                    'right_upper_white_ratio_difference': ru_white_diff,
                    'left_lower_black_ratio_difference': ll_black_diff,
                    'left_lower_gray_ratio_difference': ll_gray_diff,
                    'left_lower_white_ratio_difference': ll_white_diff,
                    'right_lower_black_ratio_difference': rl_black_diff,
                    'right_lower_gray_ratio_difference': rl_gray_diff,
                    'right_lower_white_ratio_difference': rl_white_diff,
                    'tilt_difference_eye_eye': tilt_diffs[0],
                    'tilt_difference_eye_nose': tilt_diffs[1],
                    'tilt_difference_nose_eye': tilt_diffs[2]
                })
            
            if len(current_group) >= 3:
                if current_diffs:
                    avg_diffs = {
                        'left_upper_black_ratio_difference': np.mean([d['left_upper_black_ratio_difference'] for d in current_diffs]),
                        'left_upper_gray_ratio_difference': np.mean([d['left_upper_gray_ratio_difference'] for d in current_diffs]),
                        'left_upper_white_ratio_difference': np.mean([d['left_upper_white_ratio_difference'] for d in current_diffs]),
                        'right_upper_black_ratio_difference': np.mean([d['right_upper_black_ratio_difference'] for d in current_diffs]),
                        'right_upper_gray_ratio_difference': np.mean([d['right_upper_gray_ratio_difference'] for d in current_diffs]),
                        'right_upper_white_ratio_difference': np.mean([d['right_upper_white_ratio_difference'] for d in current_diffs]),
                        'left_lower_black_ratio_difference': np.mean([d['left_lower_black_ratio_difference'] for d in current_diffs]),
                        'left_lower_gray_ratio_difference': np.mean([d['left_lower_gray_ratio_difference'] for d in current_diffs]),
                        'left_lower_white_ratio_difference': np.mean([d['left_lower_white_ratio_difference'] for d in current_diffs]),
                        'right_lower_black_ratio_difference': np.mean([d['right_lower_black_ratio_difference'] for d in current_diffs]),
                        'right_lower_gray_ratio_difference': np.mean([d['right_lower_gray_ratio_difference'] for d in current_diffs]),
                        'right_lower_white_ratio_difference': np.mean([d['right_lower_white_ratio_difference'] for d in current_diffs]),
                        'tilt_difference_eye_eye': np.mean([d['tilt_difference_eye_eye'] for d in current_diffs if d['tilt_difference_eye_eye'] != float('inf')]) or 0.0,
                        'tilt_difference_eye_nose': np.mean([d['tilt_difference_eye_nose'] for d in current_diffs if d['tilt_difference_eye_nose'] != float('inf')]) or 0.0,
                        'tilt_difference_nose_eye': np.mean([d['tilt_difference_nose_eye'] for d in current_diffs if d['tilt_difference_nose_eye'] != float('inf')]) or 0.0
                    }
                else:
                    avg_diffs = {
                        'left_upper_black_ratio_difference': 0.0,
                        'left_upper_gray_ratio_difference': 0.0,
                        'left_upper_white_ratio_difference': 0.0,
                        'right_upper_black_ratio_difference': 0.0,
                        'right_upper_gray_ratio_difference': 0.0,
                        'right_upper_white_ratio_difference': 0.0,
                        'left_lower_black_ratio_difference': 0.0,
                        'left_lower_gray_ratio_difference': 0.0,
                        'left_lower_white_ratio_difference': 0.0,
                        'right_lower_black_ratio_difference': 0.0,
                        'right_lower_gray_ratio_difference': 0.0,
                        'right_lower_white_ratio_difference': 0.0,
                        'tilt_difference_eye_eye': 0.0,
                        'tilt_difference_eye_nose': 0.0,
                        'tilt_difference_nose_eye': 0.0
                    }
                    logger.info(f"Empty current_diffs for group starting with {img1_path}")
                    print(f"Empty current_diffs for group starting with {img1_path}")
                
                groups.append({
                    'images': current_group,
                    'avg_diffs': avg_diffs
                })
                for img_path, _ in current_group:
                    used_images.add(img_path)
    
    logger.info(f"Found {len(groups)} groups")
    print(f"Found {len(groups)} groups")
    
    # ソート
    def safe_sort_key(x):
        diffs = x['avg_diffs']
        return (
            diffs.get('left_upper_black_ratio_difference', 0.0),
            diffs.get('left_upper_gray_ratio_difference', 0.0),
            diffs.get('left_upper_white_ratio_difference', 0.0),
            diffs.get('right_upper_black_ratio_difference', 0.0),
            diffs.get('right_upper_gray_ratio_difference', 0.0),
            diffs.get('right_upper_white_ratio_difference', 0.0),
            diffs.get('left_lower_black_ratio_difference', 0.0),
            diffs.get('left_lower_gray_ratio_difference', 0.0),
            diffs.get('left_lower_white_ratio_difference', 0.0),
            diffs.get('right_lower_black_ratio_difference', 0.0),
            diffs.get('right_lower_gray_ratio_difference', 0.0),
            diffs.get('right_lower_white_ratio_difference', 0.0),
            diffs.get('tilt_difference_eye_eye', float('inf')) or float('inf'),
            diffs.get('tilt_difference_eye_nose', float('inf')) or float('inf'),
            diffs.get('tilt_difference_nose_eye', float('inf')) or float('inf')
        )
    
    try:
        groups = sorted(groups, key=safe_sort_key)
    except Exception as e:
        logger.error(f"Error sorting groups: {e}")
        print(f"Error sorting groups: {e}")
        return
    
    # グループ番号と画像パスの出力
    logger.info(f"Top {min(top_n, len(groups))} similar image groups:")
    print(f"Top {min(top_n, len(groups))} similar image groups:")
    displayed_groups = 0
    for group in groups[:top_n]:
        images = group['images']
        logger.info(f"Group {displayed_groups+1}")
        print(f"Group {displayed_groups+1}")
        for img_path, _ in images:
            logger.info(f"  {img_path}")
            print(f"  {img_path}")
        displayed_groups += 1
        if displayed_groups >= top_n:
            break
    
    if displayed_groups == 0:
        logger.info("No groups to display")
        print("No groups to display")
    
    # グループごとに1枚を残して削除
    csv_data = []
    deleted_images = set()
    logger.info("Starting image deletion process")
    print("Starting image deletion process")
    for group_id, group in enumerate(groups):
        images = group['images']
        if len(images) < 1:
            continue
        # 最初の画像を保持
        keep_img_path, keep_category = images[0]
        logger.info(f"Group {group_id+1}: Keeping first image, deleting others")
        print(f"Group {group_id+1}: Keeping first image, deleting others")
        logger.info(f"  Keeping: {keep_img_path}")
        print(f"  Keeping: {keep_img_path}")
        
        # CSVに保持画像を追加
        csv_data.append({
            'group_id': group_id,
            'image_path': keep_img_path,
            'category': keep_category,
            **group['avg_diffs']
        })
        
        # 2番目以降を削除
        for img_path, category in images[1:]:
            logger.info(f"  Deleting: {img_path}")
            print(f"  Deleting: {img_path}")
            try:
                os.remove(img_path)
                deleted_images.add(img_path)
                logger.info(f"  Successfully deleted: {img_path}")
                print(f"  Successfully deleted: {img_path}")
            except Exception as e:
                logger.error(f"  Failed to delete {img_path}: {e}")
                print(f"  Failed to delete {img_path}: {e}")
    
    # CSV出力（保持画像のみ）
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Results saved to {OUTPUT_CSV}")
        print(f"Results saved to {OUTPUT_CSV}")
    else:
        logger.info("No images to save in CSV")
        print("No images to save in CSV")

def main():
    try:
        logger.info("Starting main function")
        print("Starting main function")
        find_similar_images(PREPROCESSED_TRAIN_DIR)
        find_similar_images(PREPROCESSED_VAL_DIR)
        logger.info("Main function completed")
        print("Main function completed")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error in main: {e}")
    finally:
        face_mesh.close()

if __name__ == "__main__":
    main()