import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import random
from collections import defaultdict
from icrawler.builtin import GoogleImageCrawler
import logging
import shutil

# 検索キーワードと最大ダウンロード数
KEYWORD = "稲森いずみ"
MAX_NUM = 50  # CPU環境向けに削減
OUTPUT_DIR = str(random.randint(0, 1000)).zfill(4)

# ログ設定
logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)

# 設定
VAN_RATIO = 0.35
IMG_SIZE = 112
COLOR_RATIO_THRESHOLD = 0.125
TILT_THRESHOLD = 0.125
TOP_N = 100
OUTPUT_CSV = f'similar_images_{KEYWORD}.csv'
TRIANGLE_LANDMARK_INDICES = [(33, 263), (263, 1), (1, 33)]
NOSE_INDEX = 4
RIGHT_EYE_INDEX = 33
LEFT_EYE_INDEX = 263
JAW_INDEX = 152

# MediaPipe（顔検出のみ）
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def setup_crawler(storage_dir, parser_threads=4, downloader_threads=4):
    """GoogleImageCrawlerのインスタンスを生成"""
    return GoogleImageCrawler(
        parser_threads=parser_threads,
        downloader_threads=downloader_threads,
        storage={'root_dir': storage_dir}
    )

def download_images(keyword, max_num):
    """3つのキーワードで画像をダウンロード"""
    search_terms = [
        (keyword, keyword),
        (f"{keyword} 過去", f"{keyword}_過去"),
        (f"{keyword} 現在", f"{keyword}_現在")
    ]
    
    for search_keyword, storage_dir in search_terms:
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
        os.makedirs(storage_dir, exist_ok=True)
        
        logger.info(f"Starting download for keyword: {search_keyword}, storage: {storage_dir}")
        print(f"Downloading images for: {search_keyword}")
        
        crawler = setup_crawler(storage_dir)
        crawler.crawl(keyword=search_keyword, max_num=max_num)
        
        downloaded_files = [f for f in os.listdir(storage_dir) if os.path.isfile(os.path.join(storage_dir, f))]
        logger.info(f"Downloaded {len(downloaded_files)} images for {search_keyword}")
        print(f"Downloaded {len(downloaded_files)} images for {search_keyword}")

def rename_files(keyword):
    """各フォルダ内のファイル名を親フォルダ名に基づいてリネーム"""
    folders = [keyword, f"{keyword}_過去", f"{keyword}_現在"]
    
    for folder in folders:
        if not os.path.exists(folder):
            logger.warning(f"Folder {folder} does not exist, skipping rename")
            print(f"Folder {folder} does not exist, skipping rename")
            continue
        
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            old_path = os.path.join(folder, file)
            new_filename = f"{folder}_{file}"
            new_path = os.path.join(folder, new_filename)
            
            try:
                os.rename(old_path, new_path)
                logger.info(f"Renamed {old_path} to {new_path}")
                print(f"Renamed {old_path} to {new_path}")
            except Exception as e:
                logger.error(f"Error renaming {old_path} to {new_path}: {e}")
                print(f"Error renaming {old_path} to {new_path}: {e}")

def consolidate_files(keyword):
    """全フォルダのファイルをランダム番号フォルダに統合し、連番リネーム"""
    output_dir = OUTPUT_DIR
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    folders = [keyword, f"{keyword}_過去", f"{keyword}_現在"]
    
    for folder in folders:
        if not os.path.exists(folder):
            logger.warning(f"Folder {folder} does not exist, skipping consolidation")
            print(f"Folder {folder} does not exist, skipping consolidation")
            continue
        
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            src_path = os.path.join(folder, file)
            dst_path = os.path.join(output_dir, file)
            
            try:
                shutil.move(src_path, dst_path)
                logger.info(f"Moved {src_path} to {dst_path}")
                print(f"Moved {src_path} to {dst_path}")
            except Exception as e:
                logger.error(f"Error moving {src_path} to {dst_path}: {e}")
                print(f"Error moving {src_path} to {dst_path}: {e}")
    
    for folder in folders:
        if os.path.exists(folder) and not os.listdir(folder):
            shutil.rmtree(folder)
            logger.info(f"Removed empty folder {folder}")
            print(f"Removed empty folder {folder}")
    
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    for i, file in enumerate(files, 1):
        old_path = os.path.join(output_dir, file)
        ext = os.path.splitext(file)[1].lower()
        new_filename = f"{OUTPUT_DIR}_{i:03d}{ext}"
        new_path = os.path.join(output_dir, new_filename)
        
        try:
            os.rename(old_path, new_path)
            logger.info(f"Renamed {old_path} to {new_path}")
            print(f"Renamed {old_path} to {new_path}")
        except Exception as e:
            logger.error(f"Error renaming {old_path} to {new_path}: {e}")
            print(f"Error renaming {old_path} to {new_path}: {e}")

def detect_and_crop_faces(input_dir):
    """顔検出、切り取り、リサイズ、グレースケール変換を<OUTPUT_DIR>/resizedと<OUTPUT_DIR>/processedに保存"""
    resized_dir = os.path.join(input_dir, "resized")
    processed_dir = os.path.join(input_dir, "processed")
    if os.path.exists(resized_dir):
        shutil.rmtree(resized_dir)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(resized_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    skip_counters = {'no_face': 0, 'multiple_faces': 0, 'deleted_no_face': 0, 'deleted_multiple_faces': 0}
    total_images = 0
    
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for filename in files:
        total_images += 1
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            skip_counters['no_face'] += 1
            logger.info(f"画像読み込み失敗 {img_path}")
            print(f"画像読み込み失敗 {img_path}")
            try:
                os.remove(img_path)
                skip_counters['deleted_no_face'] += 1
                logger.info(f"削除：{img_path} (no_face)")
                print(f"削除：{img_path} (no_face)")
            except Exception as e:
                logger.error(f"削除エラー {img_path} (no_face): {e}")
                print(f"削除エラー {img_path} (no_face): {e}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)
        
        if not results.detections:
            skip_counters['no_face'] += 1
            logger.info(f"顔検出失敗 {filename}")
            print(f"顔検出失敗 {filename}")
            try:
                os.remove(img_path)
                skip_counters['deleted_no_face'] += 1
                logger.info(f"削除：{img_path} (no_face)")
                print(f"削除：{img_path} (no_face)")
            except Exception as e:
                logger.error(f"削除エラー {img_path} (no_face): {e}")
                print(f"削除エラー {img_path} (no_face): {e}")
            continue
        
        if len(results.detections) > 1:
            skip_counters['multiple_faces'] += 1
            logger.info(f"複数顔検出 {filename}")
            print(f"複数顔検出 {filename}")
            try:
                os.remove(img_path)
                skip_counters['deleted_multiple_faces'] += 1
                logger.info(f"削除：{img_path} (multiple_faces)")
                print(f"削除：{img_path} (multiple_faces)")
            except Exception as e:
                logger.error(f"削除エラー {img_path} (multiple_faces): {e}")
                print(f"削除エラー {img_path} (multiple_faces): {e}")
            continue
        
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w = img.shape[:2]
        x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
        
        face_image = img[y:y + height, x:x + width]
        if face_image is None or face_image.size == 0:
            skip_counters['no_face'] += 1
            logger.info(f"顔領域切り取り失敗 {filename}")
            print(f"顔領域切り取り失敗 {filename}")
            try:
                os.remove(img_path)
                skip_counters['deleted_no_face'] += 1
                logger.info(f"削除：{img_path} (no_face)")
                print(f"削除：{img_path} (no_face)")
            except Exception as e:
                logger.error(f"削除エラー {img_path} (no_face): {e}")
                print(f"削除エラー {img_path} (no_face): {e}")
            continue
        
        # 先にリサイズ
        face_image_resized = cv2.resize(face_image, (IMG_SIZE, IMG_SIZE))
        resized_path = os.path.join(resized_dir, filename)
        cv2.imwrite(resized_path, face_image_resized)
        logger.info(f"リサイズ画像保存：{resized_path}")
        print(f"リサイズ画像保存：{resized_path}")
        
        # 次にグレースケール変換
        gray = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2GRAY)
        
        if gray.shape != (IMG_SIZE, IMG_SIZE):
            logger.error(f"無効な画像サイズ: {filename}")
            print(f"無効な画像サイズ: {filename}")
            continue
        
        processed_path = os.path.join(processed_dir, filename)
        cv2.imwrite(processed_path, gray)
        logger.info(f"グレースケール画像保存：{processed_path}")
        print(f"グレースケール画像保存：{processed_path}")
    
    for reason, count in skip_counters.items():
        rate = count / total_images * 100 if total_images > 0 else 0
        logger.info(f"{input_dir}: {reason} {count}/{total_images} ({rate:.1f}%)")
        print(f"{input_dir}: {reason} {count}/{total_images} ({rate:.1f}%)")

def extract_landmarks(img_path):
    """画像から顔ランドマークを抽出（2D座標のみ）"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"画像読み込み失敗: {img_path}")
            print(f"画像読み込み失敗: {img_path}")
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            coords = [(l.x * IMG_SIZE, l.y * IMG_SIZE) for l in landmarks]
            nose_center = coords[NOSE_INDEX]
            coords = [(x - nose_center[0], y - nose_center[1]) for x, y in coords]
            return np.array(coords)
        else:
            logger.info(f"ランドマーク検出失敗 {img_path}")
            print(f"ランドマーク検出失敗 {img_path}")
            return None
    except Exception as e:
        logger.error(f"処理エラー {img_path}: {e}")
        print(f"処理エラー {img_path}: {e}")
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
        logger.error(f"傾き計算エラー: {e}")
        print(f"傾き計算エラー: {e}")
        return float('inf')

def compute_color_ratios(img_path):
    """画像を4分割し、各領域の黒・灰・白の割合を計算"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"画像読み込み失敗: {img_path}")
            print(f"画像読み込み失敗: {img_path}")
            return None, None, None, None, None, None, None, None, None, None, None, None
        
        left_upper = img[:IMG_SIZE//2, :IMG_SIZE//2]
        right_upper = img[:IMG_SIZE//2, IMG_SIZE//2:]
        left_lower = img[IMG_SIZE//2:, :IMG_SIZE//2]
        right_lower = img[IMG_SIZE//2:, IMG_SIZE//2:]
        total_pixels = left_upper.size
        
        lu_black_pixels = np.sum(left_upper <= 85)
        lu_gray_pixels = np.sum((left_upper >= 86) & (left_upper <= 170))
        lu_white_pixels = np.sum(left_upper >= 171)
        lu_black_ratio = lu_black_pixels / total_pixels
        lu_gray_ratio = lu_gray_pixels / total_pixels
        lu_white_ratio = lu_white_pixels / total_pixels
        
        ru_black_pixels = np.sum(right_upper <= 85)
        ru_gray_pixels = np.sum((right_upper >= 86) & (right_upper <= 170))
        ru_white_pixels = np.sum(right_upper >= 171)
        ru_black_ratio = ru_black_pixels / total_pixels
        ru_gray_ratio = ru_gray_pixels / total_pixels
        ru_white_ratio = ru_white_pixels / total_pixels
        
        ll_black_pixels = np.sum(left_lower <= 85)
        ll_gray_pixels = np.sum((left_lower >= 86) & (left_lower <= 170))
        ll_white_pixels = np.sum(left_lower >= 171)
        ll_black_ratio = ll_black_pixels / total_pixels
        ll_gray_ratio = ll_gray_pixels / total_pixels
        ll_white_ratio = ll_white_pixels / total_pixels
        
        rl_black_pixels = np.sum(right_lower <= 85)
        rl_gray_pixels = np.sum((right_lower >= 86) & (right_lower <= 170))
        rl_white_pixels = np.sum(right_lower >= 171)
        rl_black_ratio = rl_black_pixels / total_pixels
        rl_gray_ratio = rl_gray_pixels / total_pixels
        rl_white_ratio = rl_white_pixels / total_pixels
        
        return (lu_black_ratio, lu_gray_ratio, lu_white_ratio,
                ru_black_ratio, ru_gray_ratio, ru_white_ratio,
                ll_black_ratio, ll_gray_ratio, ll_white_ratio,
                rl_black_ratio, rl_gray_ratio, rl_white_ratio)
    except Exception as e:
        logger.error(f"処理エラー {img_path}: {e}")
        print(f"処理エラー {img_path}: {e}")
        return None, None, None, None, None, None, None, None, None, None, None, None

def get_filename_prefix(filename, prefix_length=3):
    """ファイル名の先頭3文字を取得"""
    try:
        prefix = os.path.basename(filename)[:prefix_length]
        return prefix
    except Exception as e:
        logger.error(f"プレフィックス取得エラー {filename}: {e}")
        print(f"プレフィックス取得エラー {filename}: {e}")
        return ""

def find_similar_images(input_dir):
    """類似画像を検出して削除、ランドマーク非検出画像も削除"""
    input_dir = os.path.join(input_dir, "processed")
    logger.info(f"{input_dir} の類似画像検索開始")
    print(f"{input_dir} の類似画像検索開始")
    
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            img_path = os.path.join(root, filename)
            image_files.append((img_path, os.path.basename(os.path.dirname(img_path))))
    
    logger.info(f"{input_dir} で {len(image_files)} 画像を検出")
    print(f"{input_dir} で {len(image_files)} 画像を検出")
    
    prefix_groups = defaultdict(list)
    for img_path, category in image_files:
        prefix = get_filename_prefix(img_path)
        prefix_groups[prefix].append((img_path, category))
    
    for prefix, images in prefix_groups.items():
        logger.info(f"プレフィックス {prefix}: {len(images)} 画像")
        print(f"プレフィックス {prefix}: {len(images)} 画像")
    
    image_data = {}
    skip_counters = {'deleted_no_landmarks': 0}
    for img_path, category in image_files:
        if not os.path.exists(img_path):
            logger.error(f"画像が見つかりません: {img_path}")
            print(f"画像が見つかりません: {img_path}")
            continue
        ratios = compute_color_ratios(img_path)
        landmarks = extract_landmarks(img_path)
        if landmarks is None:
            skip_counters['deleted_no_landmarks'] += 1
            logger.info(f"ランドマーク検出失敗により削除 {img_path}")
            print(f"ランドマーク検出失敗により削除 {img_path}")
            try:
                base_img_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
                if os.path.exists(base_img_path):
                    os.remove(base_img_path)
                    logger.info(f"成功的に削除（ランドマーク検出失敗）: {base_img_path}")
                    print(f"成功的に削除（ランドマーク検出失敗）: {base_img_path}")
                else:
                    logger.info(f"ファイルが見つかりません（ランドマーク検出失敗）: {base_img_path}")
                    print(f"ファイルが見つかりません（ランドマーク検出失敗）: {base_img_path}")
            except Exception as e:
                logger.error(f"削除エラー（ランドマーク検出失敗） {img_path}: {e}")
                print(f"削除エラー（ランドマーク検出失敗） {img_path}: {e}")
            continue
        if all(r is not None for r in ratios):
            image_data[img_path] = {
                'category': category,
                'ratios': ratios,
                'landmarks': landmarks
            }
        else:
            logger.info(f"無効な比率のため {img_path} をスキップ")
            print(f"無効な比率のため {img_path} をスキップ")
    
    logger.info(f"有効な特徴を持つ画像: {len(image_data)}")
    print(f"有効な特徴を持つ画像: {len(image_data)}")
    for reason, count in skip_counters.items():
        rate = count / len(image_files) * 100 if len(image_files) > 0 else 0
        logger.info(f"{input_dir}: {reason} {count}/{len(image_files)} ({rate:.1f}%)")
        print(f"{input_dir}: {reason} {count}/{len(image_files)} ({rate:.1f}%)")
    
    groups = []
    used_images = set()
    
    for prefix, images in prefix_groups.items():
        if len(images) < 3:
            logger.info(f"プレフィックス {prefix} をスキップ: 画像 {len(images)} のみ")
            print(f"プレフィックス {prefix} をスキップ: 画像 {len(images)} のみ")
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
                
                if not (lu_black_diff <= COLOR_RATIO_THRESHOLD and
                        lu_gray_diff <= COLOR_RATIO_THRESHOLD and
                        lu_white_diff <= COLOR_RATIO_THRESHOLD and
                        ru_black_diff <= COLOR_RATIO_THRESHOLD and
                        ru_gray_diff <= COLOR_RATIO_THRESHOLD and
                        ru_white_diff <= COLOR_RATIO_THRESHOLD and
                        ll_black_diff <= COLOR_RATIO_THRESHOLD and
                        ll_gray_diff <= COLOR_RATIO_THRESHOLD and
                        ll_white_diff <= COLOR_RATIO_THRESHOLD and
                        rl_black_diff <= COLOR_RATIO_THRESHOLD and
                        rl_gray_diff <= COLOR_RATIO_THRESHOLD and
                        rl_white_diff <= COLOR_RATIO_THRESHOLD):
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
                
                if any(diff > TILT_THRESHOLD for diff in tilt_diffs):
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
                    logger.info(f"{img1_path} で空の current_diffs")
                    print(f"{img1_path} で空の current_diffs")
                
                groups.append({
                    'images': current_group,
                    'avg_diffs': avg_diffs
                })
                for img_path, _ in current_group:
                    used_images.add(img_path)
    
    logger.info(f"{len(groups)} グループを検出")
    print(f"{len(groups)} グループを検出")
    
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
        logger.error(f"グループソートエラー: {e}")
        print(f"グループソートエラー: {e}")
        return image_data
    
    logger.info(f"上位 {min(TOP_N, len(groups))} 類似画像グループ:")
    print(f"上位 {min(TOP_N, len(groups))} 類似画像グループ:")
    displayed_groups = 0
    for group in groups[:TOP_N]:
        images = group['images']
        logger.info(f"グループ {displayed_groups+1}")
        print(f"グループ {displayed_groups+1}")
        for img_path, _ in images:
            logger.info(f"  {img_path}")
            print(f"  {img_path}")
        displayed_groups += 1
        if displayed_groups >= TOP_N:
            break
    
    if displayed_groups == 0:
        logger.info("表示するグループなし")
        print("表示するグループなし")
    
    csv_data = []
    deleted_images = set()
    logger.info("類似画像削除処理開始")
    print("類似画像削除処理開始")
    for group_id, group in enumerate(groups):
        images = group['images']
        if len(images) < 1:
            continue
        keep_img_path, keep_category = images[0]
        logger.info(f"グループ {group_id+1}: 最初の画像を保持、その他を削除")
        print(f"グループ {group_id+1}: 最初の画像を保持、その他を削除")
        logger.info(f"  保持: {keep_img_path}")
        print(f"  保持: {keep_img_path}")
        
        csv_data.append({
            'group_id': group_id,
            'image_path': keep_img_path,
            'category': keep_category,
            **group['avg_diffs']
        })
        
        for img_path, category in images[1:]:
            logger.info(f"  削除: {img_path}")
            print(f"  削除: {img_path}")
            try:
                base_img_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
                if os.path.exists(base_img_path):
                    os.remove(base_img_path)
                    deleted_images.add(base_img_path)
                    logger.info(f"  成功的に削除: {base_img_path}")
                    print(f"  成功的に削除: {base_img_path}")
                else:
                    logger.info(f"  ファイルが見つかりません: {base_img_path}")
                    print(f"  ファイルが見つかりません: {base_img_path}")
            except Exception as e:
                logger.error(f"  削除エラー {base_img_path}: {e}")
                print(f"  削除エラー {base_img_path}: {e}")
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"結果を {OUTPUT_CSV} に保存")
        print(f"結果を {OUTPUT_CSV} に保存")
    else:
        logger.info("CSVに保存する画像なし")
        print("CSVに保存する画像なし")
    
    return image_data

def process_images(keyword):
    """ランダム番号フォルダの画像を処理"""
    input_dir = OUTPUT_DIR
    
    logger.info(f"画像処理開始：{input_dir} → {os.path.join(input_dir, 'resized')} と {os.path.join(input_dir, 'processed')}")
    print(f"画像処理開始：{input_dir} → {os.path.join(input_dir, 'resized')} と {os.path.join(input_dir, 'processed')}")
    
    detect_and_crop_faces(input_dir)
    image_data = find_similar_images(input_dir)
    
    logger.info(f"画像処理完了：{os.path.join(input_dir, 'resized')} と {os.path.join(input_dir, 'processed')}")
    print(f"画像処理完了：{os.path.join(input_dir, 'resized')} と {os.path.join(input_dir, 'processed')}")

def main():
    try:
        logger.info(f"処理開始 for keyword: {KEYWORD}")
        print(f"処理開始 for keyword: {KEYWORD}")
        
        download_images(KEYWORD, MAX_NUM)
        rename_files(KEYWORD)
        consolidate_files(KEYWORD)
        process_images(KEYWORD)
        
        logger.info(f"全処理完了 for keyword: {KEYWORD}")
        print(f"全処理完了 for keyword: {KEYWORD}")
    except Exception as e:
        logger.error(f"メインエラー: {e}")
        print(f"メインエラー: {e}")
    finally:
        face_detection.close()
        face_mesh.close()

if __name__ == "__main__":
    main()