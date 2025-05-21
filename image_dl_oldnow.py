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
MAX_NUM = 100
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
IMG_SIZE = 224
COLOR_RATIO_THRESHOLD = 0.02
EYE_SLOPE_THRESHOLD = 0.05  # 傾き差の閾値
TOP_N = 100
OUTPUT_CSV = f'similar_images_{KEYWORD}.csv'
QUAD_LANDMARK_INDICES = [33, 263, 291, 61]
NOSE_INDEX = 4
EYE_LEFT_INDEX = 159  # 左目
EYE_RIGHT_INDEX = 386  # 右目
MOUTH_LEFT_INDEX = 61  # 口の左端
MOUTH_RIGHT_INDEX = 291  # 口の右端

# MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def remove_background(img):
    """MediaPipe Selfie Segmentationで背景を透過"""
    try:
        logger.info(f"背景除去開始: img_shape={img.shape}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(img_rgb)
        
        if results.segmentation_mask is None or results.segmentation_mask.size == 0:
            logger.warning("セグメンテーションマスクが空または無効")
            output = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            output[:, :, 3] = 255
            logger.info(f"フォールバック画像生成: shape={output.shape}")
            return output

        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        logger.info(f"マスク生成: shape={mask.shape}, dtype={mask.dtype}")

        output = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        output[:, :, 3] = mask
        logger.info(f"透過画像生成: shape={output.shape}")
        return output
    except Exception as e:
        logger.error(f"背景除去エラー: {e}")
        output = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        output[:, :, 3] = 255
        logger.info(f"エラー時フォールバック: shape={output.shape}")
        return output

def setup_crawler(storage_dir, parser_threads=4, downloader_threads=4):
    """GoogleImageCrawlerのインスタンスを生成"""
    return GoogleImageCrawler(
        parser_threads=parser_threads,
        downloader_threads=downloader_threads,
        storage={'root_dir': storage_dir}
    )

def download_images(keyword, max_num):
    """複数のキーワードで画像をダウンロード"""
    search_terms = [
        (keyword, keyword),
        (f"{keyword} 正面", f"{keyword}_正面"),
        (f"{keyword} 顔", f"{keyword}_顔"),
        (f"{keyword} 昔", f"{keyword}_昔"),
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
    folders = [keyword, f"{keyword}_昔", f"{keyword}_現在", f"{keyword}_正面", f"{keyword}_顔"]
    
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
    
    folders = [keyword, f"{keyword}_昔", f"{keyword}_現在", f"{keyword}_正面", f"{keyword}_顔"]
    
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
    """すべての検出顔を切り取り、リサイズ、背景透過、グレースケール変換"""
    resized_dir = os.path.join(input_dir, "resized")
    processed_dir = os.path.join(input_dir, "processed")
    if os.path.exists(resized_dir):
        shutil.rmtree(resized_dir)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(resized_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    skip_counters = {
        'no_face': 0,
        'deleted_no_face': 0
    }
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
        
        h, w = img.shape[:2]
        for face_idx, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            area = bboxC.width * bboxC.height
            logger.info(f"{filename}: 顔{face_idx+1} (面積: {area:.6f})")
            
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            width = min(width, w-x)
            height = min(height, h-y)
            
            face_image = img[y:y + height, x:x + width]
            if face_image is None or face_image.size == 0:
                skip_counters['no_face'] += 1
                logger.info(f"顔領域切り取り失敗 {filename} (face{face_idx+1}, bbox: x={x}, y={y}, w={width}, h={height})")
                print(f"顔領域切り取り失敗 {filename} (face{face_idx+1})")
                continue
            
            face_image_resized = cv2.resize(face_image, (IMG_SIZE, IMG_SIZE))
            base_name, ext = os.path.splitext(filename)
            resized_filename = f"{base_name}_face{face_idx+1}{ext}"
            resized_path = os.path.join(resized_dir, resized_filename)
            face_image_transparent = remove_background(face_image_resized)
            cv2.imwrite(resized_path, face_image_transparent)
            logger.info(f"透過画像保存：{resized_path}")
            print(f"透過画像保存：{resized_path}")
            
            mask = face_image_transparent[:, :, 3] > 0
            gray = cv2.cvtColor(face_image_transparent, cv2.COLOR_BGRA2GRAY)
            gray[~mask] = 0
            
            if gray.shape != (IMG_SIZE, IMG_SIZE):
                logger.error(f"無効な画像サイズ: {resized_filename}")
                print(f"無効な画像サイズ: {resized_filename}")
                continue
            
            processed_filename = f"{base_name}_face{face_idx+1}.png"
            processed_path = os.path.join(processed_dir, processed_filename)
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
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"画像読み込み失敗: {img_path}")
            print(f"画像読み込み失敗: {img_path}")
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
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

def compute_landmark_slopes(landmarks):
    """左目-右目、左目-鼻、右目-鼻、口右-鼻、口左-鼻の傾きを計算"""
    try:
        left_eye = landmarks[EYE_LEFT_INDEX]
        right_eye = landmarks[EYE_RIGHT_INDEX]
        nose = landmarks[NOSE_INDEX]
        mouth_left = landmarks[MOUTH_LEFT_INDEX]
        mouth_right = landmarks[MOUTH_RIGHT_INDEX]
        
        slopes = {}
        # 左目-右目
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        slopes['eye_left_to_right'] = dy / (dx + 1e-10)
        
        # 左目-鼻
        dy = nose[1] - left_eye[1]
        dx = nose[0] - left_eye[0]
        slopes['eye_left_to_nose'] = dy / (dx + 1e-10)
        
        # 右目-鼻
        dy = nose[1] - right_eye[1]
        dx = nose[0] - right_eye[0]
        slopes['eye_right_to_nose'] = dy / (dx + 1e-10)
        
        # 口右-鼻
        dy = nose[1] - mouth_right[1]
        dx = nose[0] - mouth_right[0]
        slopes['mouth_right_to_nose'] = dy / (dx + 1e-10)
        
        # 口左-鼻
        dy = nose[1] - mouth_left[1]
        dx = nose[0] - mouth_left[0]
        slopes['mouth_left_to_nose'] = dy / (dx + 1e-10)
        
        return slopes
    except Exception as e:
        logger.error(f"ランドマーク傾き計算エラー: {e}")
        print(f"ランドマーク傾き計算エラー: {e}")
        return {key: float('inf') for key in [
            'eye_left_to_right', 'eye_left_to_nose', 'eye_right_to_nose',
            'mouth_right_to_nose', 'mouth_left_to_nose'
        ]}

def compute_color_ratios(img_path):
    """画像を4分割、上下左右2分割、全体で黒・灰・白の割合を計算"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"画像読み込み失敗: {img_path}")
            print(f"画像読み込み失敗: {img_path}")
            return [None] * 27
        
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img if len(img.shape) == 2 else img[:, :, 0]
            mask = np.ones_like(gray, dtype=np.uint8) * 255
            logger.info(f"グレースケール画像を処理: {img_path}")
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            mask = img[:, :, 3] > 0
            logger.info(f"BGRA画像を処理: {img_path}")
        
        left_upper = gray[:IMG_SIZE//2, :IMG_SIZE//2]
        right_upper = gray[:IMG_SIZE//2, IMG_SIZE//2:]
        left_lower = gray[IMG_SIZE//2:, :IMG_SIZE//2]
        right_lower = gray[IMG_SIZE//2:, IMG_SIZE//2:]
        upper_half = gray[:IMG_SIZE//2, :]
        lower_half = gray[IMG_SIZE//2:, :]
        left_half = gray[:, :IMG_SIZE//2]
        right_half = gray[:, IMG_SIZE//2:]
        whole = gray
        
        lu_mask = mask[:IMG_SIZE//2, :IMG_SIZE//2]
        ru_mask = mask[:IMG_SIZE//2, IMG_SIZE//2:]
        ll_mask = mask[IMG_SIZE//2:, :IMG_SIZE//2]
        rl_mask = mask[IMG_SIZE//2:, IMG_SIZE//2:]
        upper_mask = mask[:IMG_SIZE//2, :]
        lower_mask = mask[IMG_SIZE//2:, :]
        left_mask = mask[:, :IMG_SIZE//2]
        right_mask = mask[:, IMG_SIZE//2:]
        whole_mask = mask
        
        total_pixels = np.sum(whole_mask) or 1
        
        ratios = []
        for region, region_mask in [
            (left_upper, lu_mask), (right_upper, ru_mask), (left_lower, ll_mask), (right_lower, rl_mask),
            (upper_half, upper_mask), (lower_half, lower_mask), (left_half, left_mask), (right_half, right_mask),
            (whole, whole_mask)
        ]:
            black_pixels = np.sum((region <= 85) & region_mask)
            gray_pixels = np.sum((region >= 86) & (region <= 170) & region_mask)
            white_pixels = np.sum((region >= 171) & region_mask)
            ratios.extend([
                black_pixels / total_pixels,
                gray_pixels / total_pixels,
                white_pixels / total_pixels
            ])
        
        return ratios
    except Exception as e:
        logger.error(f"処理エラー {img_path}: {e}")
        print(f"処理エラー {img_path}: {e}")
        return [None] * 27

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
    """類似画像を検出して削除（9領域フィルタ、5つの傾きフィルタ、ログ出力）"""
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
        if landmarks is None or len(landmarks) <= max([EYE_LEFT_INDEX, EYE_RIGHT_INDEX, MOUTH_LEFT_INDEX, MOUTH_RIGHT_INDEX]):
            skip_counters['deleted_no_landmarks'] += 1
            logger.info(f"ランドマーク検出失敗または不足により削除 {img_path}")
            print(f"ランドマーク検出失敗または不足により削除 {img_path}")
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    logger.info(f"成功的に削除（ランドマーク検出失敗）: {img_path}")
                    print(f"成功的に削除（ランドマーク検出失敗）: {img_path}")
                else:
                    logger.info(f"ファイルが見つかりません（ランドマーク検出失敗）: {img_path}")
                    print(f"ファイルが見つかりません（ランドマーク検出失敗）: {img_path}")
            except Exception as e:
                logger.error(f"削除エラー（ランドマーク検出失敗） {img_path}: {e}")
                print(f"削除エラー（ランドマーク検出失敗） {img_path}: {e}")
            continue
        if all(r is not None for r in ratios):
            slopes = compute_landmark_slopes(landmarks)
            image_data[img_path] = {
                'category': category,
                'ratios': ratios,
                'landmarks': landmarks,
                'slopes': slopes
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
                slopes1 = data1['slopes']
                slopes2 = data2['slopes']
                
                # 9領域の差分
                diffs = [abs(r1 - r2) for r1, r2 in zip(ratios1, ratios2)]
                if not all(diff <= COLOR_RATIO_THRESHOLD for diff in diffs):
                    continue
                
                # 5つの傾きフィルタリング
                slope_ratios = {}
                all_slopes_valid = True
                for key in slopes1:
                    if slopes1[key] == float('inf') or slopes2[key] == float('inf'):
                        all_slopes_valid = False
                        break
                    slope_ratio = slopes1[key] / (slopes2[key] + 1e-10)
                    slope_ratios[key] = slope_ratio
                    if slope_ratio <= 1 - EYE_SLOPE_THRESHOLD or slope_ratio >= 1 + EYE_SLOPE_THRESHOLD:
                        all_slopes_valid = False
                        break
                
                if not all_slopes_valid:
                    continue
                
                # 類似画像のログ出力
                logger.info(f"類似画像検出: {img1_path} と {img2_path}")
                logger.info(f"  {img1_path} 画素割合: {', '.join([f'{r:.4f}' for r in ratios1])}")
                logger.info(f"  {img2_path} 画素割合: {', '.join([f'{r:.4f}' for r in ratios2])}")
                logger.info(f"  {img1_path} 傾き: {', '.join([f'{k}={v:.4f}' for k, v in slopes1.items()])}")
                logger.info(f"  {img2_path} 傾き: {', '.join([f'{k}={v:.4f}' for k, v in slopes2.items()])}")
                
                current_group.append((img2_path, cat2))
                current_diffs.append({
                    'left_upper_black_diff': diffs[0],
                    'left_upper_gray_diff': diffs[1],
                    'left_upper_white_diff': diffs[2],
                    'right_upper_black_diff': diffs[3],
                    'right_upper_gray_diff': diffs[4],
                    'right_upper_white_diff': diffs[5],
                    'left_lower_black_diff': diffs[6],
                    'left_lower_gray_diff': diffs[7],
                    'left_lower_white_diff': diffs[8],
                    'right_lower_black_diff': diffs[9],
                    'right_lower_gray_diff': diffs[10],
                    'right_lower_white_diff': diffs[11],
                    'upper_half_black_diff': diffs[12],
                    'upper_half_gray_diff': diffs[13],
                    'upper_half_white_diff': diffs[14],
                    'lower_half_black_diff': diffs[15],
                    'lower_half_gray_diff': diffs[16],
                    'lower_half_white_diff': diffs[17],
                    'left_half_black_diff': diffs[18],
                    'left_half_gray_diff': diffs[19],
                    'left_half_white_diff': diffs[20],
                    'right_half_black_diff': diffs[21],
                    'right_half_gray_diff': diffs[22],
                    'right_half_white_diff': diffs[23],
                    'whole_black_diff': diffs[24],
                    'whole_gray_diff': diffs[25],
                    'whole_white_diff': diffs[26],
                    **{f'{k}_ratio': v for k, v in slope_ratios.items()}
                })
            
            if len(current_group) >= 3:
                if current_diffs:
                    avg_diffs = {
                        key: np.mean([d[key] for d in current_diffs]) for key in current_diffs[0]
                    }
                else:
                    avg_diffs = {
                        key: 0.0 for key in [
                            'left_upper_black_diff', 'left_upper_gray_diff', 'left_upper_white_diff',
                            'right_upper_black_diff', 'right_upper_gray_diff', 'right_upper_white_diff',
                            'left_lower_black_diff', 'left_lower_gray_diff', 'left_lower_white_diff',
                            'right_lower_black_diff', 'right_lower_gray_diff', 'right_lower_white_diff',
                            'upper_half_black_diff', 'upper_half_gray_diff', 'upper_half_white_diff',
                            'lower_half_black_diff', 'lower_half_gray_diff', 'lower_half_white_diff',
                            'left_half_black_diff', 'left_half_gray_diff', 'left_half_white_diff',
                            'right_half_black_diff', 'right_half_gray_diff', 'right_half_white_diff',
                            'whole_black_diff', 'whole_gray_diff', 'whole_white_diff',
                            'eye_left_to_right_ratio', 'eye_left_to_nose_ratio', 'eye_right_to_nose_ratio',
                            'mouth_right_to_nose_ratio', 'mouth_left_to_nose_ratio'
                        ]
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
        return tuple(diffs.get(key, 0.0) for key in [
            'left_upper_black_diff', 'left_upper_gray_diff', 'left_upper_white_diff',
            'right_upper_black_diff', 'right_upper_gray_diff', 'right_upper_white_diff',
            'left_lower_black_diff', 'left_lower_gray_diff', 'left_lower_white_diff',
            'right_lower_black_diff', 'right_lower_gray_diff', 'right_lower_white_diff',
            'upper_half_black_diff', 'upper_half_gray_diff', 'upper_half_white_diff',
            'lower_half_black_diff', 'lower_half_gray_diff', 'lower_half_white_diff',
            'left_half_black_diff', 'left_half_gray_diff', 'left_half_white_diff',
            'right_half_black_diff', 'right_half_gray_diff', 'right_half_white_diff',
            'whole_black_diff', 'whole_gray_diff', 'whole_white_diff',
            'eye_left_to_right_ratio', 'eye_left_to_nose_ratio', 'eye_right_to_nose_ratio',
            'mouth_right_to_nose_ratio', 'mouth_left_to_nose_ratio'
        ])
    
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
    
    original_dir = os.path.dirname(input_dir)
    resized_dir = os.path.join(original_dir, "resized")
    
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
            logger.info(f"  削除: {img_path} (processed)")
            print(f"  削除: {img_path} (processed)")
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    deleted_images.add(img_path)
                    logger.info(f"  成功的に削除: {img_path} (processed)")
                    print(f"  成功的に削除: {img_path} (processed)")
                
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                for ext in ['.jpg', '.jpeg', '.png']:
                    original_path = os.path.join(original_dir, f"{base_name}{ext}")
                    if os.path.exists(original_path):
                        os.remove(original_path)
                        logger.info(f"  成功的に削除: {original_path} (original)")
                        print(f"  成功的に削除: {original_path} (original)")
                
                    resized_path = os.path.join(resized_dir, f"{base_name}{ext}")
                    if os.path.exists(resized_path):
                        os.remove(resized_path)
                        logger.info(f"  成功的に削除: {resized_path} (resized)")
                        print(f"  成功的に削除: {resized_path} (resized)")
            
            except Exception as e:
                logger.error(f"  削除エラー {img_path} (processed): {e}")
                print(f"  削除エラー {img_path} (processed): {e}")
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"結果を {OUTPUT_CSV} に保存")
        print(f"結果を {OUTPUT_CSV} に保存")
    else:
        logger.info("CSVに保存する画像なし")
        print("CSVに保存する画像なし")
    
    return image_data

def cleanup_directories():
    """元画像のみ削除、resizedとprocessedを保持"""
    try:
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info(f"元画像削除: {file_path}")
                print(f"元画像削除: {file_path}")
        
        logger.info("クリーンアップ完了：resizedとprocessedディレクトリを保持")
        print("クリーンアップ完了：resizedとprocessedディレクトリを保持")
    except Exception as e:
        logger.error(f"クリーンアップエラー: {e}")
        print(f"クリーンアップエラー: {e}")

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
        cleanup_directories()
        
        logger.info(f"全処理完了 for keyword: {KEYWORD}")
        print(f"全処理完了 for keyword: {KEYWORD}")
    except Exception as e:
        logger.error(f"メインエラー: {e}")
        print(f"メインエラー: {e}")
    finally:
        face_detection.close()
        face_mesh.close()
        selfie_segmentation.close()

if __name__ == "__main__":
    main()