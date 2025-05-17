import os
import shutil
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import math
import logging
from collections import defaultdict

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('console_log.txt', mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

# ディレクトリのパス
TRAIN_DIR = 'train'
VALIDATION_DIR = 'validation'
PREPROCESSED_TRAIN_DIR = 'preprocessed/train'
PREPROCESSED_VALIDATION_DIR = 'preprocessed/validation'

# 設定
VAN_RATIO = 0.14
img_size = 112
TARGET_NOSE_X = img_size / 2
TARGET_NOSE_Y = img_size / 2
Y_DIFF_THRESHOLD = VAN_RATIO * img_size / 2
BLACK_MAX = 85
GRAY_MIN = 86
GRAY_MAX = 170
WHITE_MIN = 171
COLOR_RATIO_THRESHOLD = 0.125
TILT_THRESHOLD = 0.125
OUTPUT_CSV = 'similar_images_color_tilt_split.csv'
TOP_N = 100
TRIANGLE_LANDMARK_INDICES = [(33, 263), (263, 1), (1, 33)]
NOSE_INDEX = 4
CHIN_INDEX = 152
RIGHT_CONTOUR_INDEX = 137
LEFT_CONTOUR_INDEX = 366

# MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def filter_cheek_nose_distance(img, img_path, filename, skip_counters):
    """回転前に適用：複数顔検出 → 頬-鼻距離の左右差チェック"""
    if img is None or img.size == 0:
        skip_counters['no_face'] += 1
        logger.info(f"回転前：画像読み込み失敗 {img_path}")
        return False, 'no_face'

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    if not results or not results.detections:
        skip_counters['no_face'] += 1
        logger.info(f"回転前：顔検出失敗 {filename}")
        return False, 'no_face'
    if len(results.detections) > 1:
        skip_counters['multiple_faces'] += 1
        logger.info(f"回転前：複数顔検出 {filename}")
        return False, 'multiple_faces'

    results_mesh = face_mesh.process(img_rgb)
    if not results_mesh or not results_mesh.multi_face_landmarks:
        skip_counters['no_landmarks'] += 1
        logger.info(f"回転前：ランドマーク検出失敗 {filename}")
        return False, 'no_landmarks'

    landmarks = results_mesh.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]
    nose = (landmarks[NOSE_INDEX].x * w, landmarks[NOSE_INDEX].y * h)
    right_contour = (landmarks[RIGHT_CONTOUR_INDEX].x * w, landmarks[RIGHT_CONTOUR_INDEX].y * h)
    left_contour = (landmarks[LEFT_CONTOUR_INDEX].x * w, landmarks[LEFT_CONTOUR_INDEX].y * h)

    nose_x = nose[0] * img_size / w
    nose_y = nose[1] * img_size / h
    right_contour_x = right_contour[0] * img_size / w
    right_contour_y = right_contour[1] * img_size / h
    left_contour_x = left_contour[0] * img_size / w
    left_contour_y = left_contour[1] * img_size / h
    dist_right = math.sqrt((nose_x - right_contour_x)**2 + (nose_y - right_contour_y)**2)
    dist_left = math.sqrt((nose_x - left_contour_x)**2 + (nose_y - left_contour_y)**2)

    if dist_right > 0:
        ratio = dist_left / dist_right
        if not (1 - VAN_RATIO <= ratio <= 1 + VAN_RATIO):
            skip_counters['cheek_nose_distance'] += 1
            logger.info(f"回転前：頬-鼻距離不均衡 {filename} (右: {dist_right:.1f}, 左: {dist_left:.1f}, 比率: {ratio:.3f})")
            return False, 'cheek_nose_distance'
    else:
        skip_counters['cheek_nose_distance'] += 1
        logger.info(f"回転前：右頬-鼻距離がゼロ {filename}")
        return False, 'cheek_nose_distance'

    return True, None

def filter_other_conditions(img, img_path, filename, skip_counters, is_preprocessed=False):
    """切り抜き後に適用：顔検出、ランドマーク、y座標、あごチェック"""
    if img is None or img.size == 0:
        skip_counters['no_face'] += 1
        logger.info(f"切り抜き後：画像読み込み失敗 {img_path}")
        return False, 'no_face'

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not is_preprocessed else cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if not results or not results.detections:
        skip_counters['no_face'] += 1
        logger.info(f"切り抜き後：顔検出失敗 {filename}")
        return False, 'no_face'

    results_mesh = face_mesh.process(img_rgb)
    if not results_mesh or not results_mesh.multi_face_landmarks:
        skip_counters['no_landmarks'] += 1
        logger.info(f"切り抜き後：ランドマーク検出失敗 {filename}")
        return False, 'no_landmarks'

    landmarks = results_mesh.multi_face_landmarks[0].landmark
    nose_y_scaled = landmarks[NOSE_INDEX].y * img_size
    right_contour_y_scaled = landmarks[RIGHT_CONTOUR_INDEX].y * img_size
    left_contour_y_scaled = landmarks[LEFT_CONTOUR_INDEX].y * img_size
    y_coords = [nose_y_scaled, right_contour_y_scaled, left_contour_y_scaled]
    y_diff_max = max(y_coords) - min(y_coords)
    if y_diff_max >= Y_DIFF_THRESHOLD:
        skip_counters['y_coordinate_diff'] += 1
        logger.info(f"切り抜き後：y座標差大 {filename} (最大差: {y_diff_max:.1f})")
        return False, 'y_coordinate_diff'

    chin_y_scaled = landmarks[CHIN_INDEX].y
    if not (1 - VAN_RATIO/2 < chin_y_scaled < 1 + VAN_RATIO/2):
        skip_counters['small_chin_y'] += 1
        logger.info(f"切り抜き後：あごy座標不適切 {filename} (あごY: {chin_y_scaled:.1f})")
        return False, 'small_chin_y'

    return True, None

def preprocess_and_cut_faces(input_dir, output_dir):
    """顔画像の前処理：複数顔＆頬-鼻チェック（回転前）→ 回転 → 切り抜き → 移動（鼻を中央に） → ランドマーク再検出 → その他チェック → ランドマーク描画 → 保存"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    skip_counters = {
        'no_face': 0, 'empty_face': 0, 'no_landmarks': 0, 'cheek_nose_distance': 0,
        'y_coordinate_diff': 0, 'no_post_crop_landmarks': 0, 'small_chin_y': 0,
        'multiple_faces': 0,
        'deleted_no_face': 0, 'deleted_empty_face': 0, 'deleted_no_landmarks': 0,
        'deleted_cheek_nose_distance': 0, 'deleted_y_coordinate_diff': 0,
        'deleted_no_post_crop_landmarks': 0, 'deleted_small_chin_y': 0,
        'deleted_multiple_faces': 0
    }
    total_images = 0

    for category in ['category1', 'category2']:
        category_input_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)

        for root, dirs, files in os.walk(category_input_dir):
            for filename in files:
                total_images += 1
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)

                # 回転前：複数顔＆頬-鼻距離チェック
                is_valid, reason = filter_cheek_nose_distance(img, img_path, filename, skip_counters)
                if not is_valid:
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters[f'deleted_{reason}'] += 1
                            logger.info(f"削除：{img_path} ({reason})")
                        else:
                            logger.info(f"ファイル {img_path} が見つかりません ({reason})")
                    except Exception as e:
                        logger.error(f"削除エラー {img_path} ({reason}): {e}")
                    continue

                # ランドマーク検出
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results_mesh = face_mesh.process(img_rgb)
                if not results_mesh or not results_mesh.multi_face_landmarks:
                    skip_counters['no_landmarks'] += 1
                    logger.info(f"回転前：ランドマーク検出失敗 {filename}")
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters['deleted_no_landmarks'] += 1
                            logger.info(f"削除：{img_path} (no_landmarks)")
                        else:
                            logger.info(f"ファイル {img_path} が見つかりません (no_landmarks)")
                    except Exception as e:
                        logger.error(f"削除エラー {img_path} (no_landmarks): {e}")
                    continue

                landmarks = results_mesh.multi_face_landmarks[0].landmark
                h, w = img.shape[:2]
                nose = (landmarks[NOSE_INDEX].x * w, landmarks[NOSE_INDEX].y * h)
                chin = (landmarks[CHIN_INDEX].x * w, landmarks[CHIN_INDEX].y * h)

                # 回転
                x_diff = nose[0] - chin[0]
                if abs(x_diff) < 1e-2:
                    rotated_image = img
                else:
                    dx = chin[0] - nose[0]
                    dy = chin[1] - nose[1]
                    angle_rad = math.atan2(dx, dy)
                    angle = -angle_rad * 180 / math.pi
                    center = (w / 2, h / 2)
                    M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_image = cv2.warpAffine(img, M_rotate, (w, h))

                # バウンディングボックス
                results = face_detection.process(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
                if not results or not results.detections:
                    skip_counters['no_face'] += 1
                    logger.info(f"回転後：顔検出失敗 {filename}")
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters['deleted_no_face'] += 1
                            logger.info(f"削除：{img_path} (no_face)")
                        else:
                            logger.info(f"ファイル {img_path} が見つかりません (no_face)")
                    except Exception as e:
                        logger.error(f"削除エラー {img_path} (no_face): {e}")
                    continue

                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # 顔領域切り取り
                face_image = rotated_image[y:y + height, x:x + width]
                if face_image is None or face_image.size == 0:
                    skip_counters['empty_face'] += 1
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters['deleted_empty_face'] += 1
                            logger.info(f"削除：{img_path} (empty_face)")
                        else:
                            logger.info(f"ファイル {img_path} が見つかりません (empty_face)")
                    except Exception as e:
                        logger.error(f"削除エラー {img_path} (empty_face): {e}")
                    continue

                # 切り抜き後：ランドマーク再検出（移動用）
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                results_post_crop = face_mesh.process(face_rgb)
                if not results_post_crop or not results_post_crop.multi_face_landmarks:
                    skip_counters['no_post_crop_landmarks'] += 1
                    logger.info(f"切り抜き後：ランドマーク再検出失敗 {filename}")
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters['deleted_no_post_crop_landmarks'] += 1
                            logger.info(f"削除：{img_path} (no_post_crop_landmarks)")
                        else:
                            logger.info(f"ファイル {img_path} が見つかりません (no_post_crop_landmarks)")
                    except Exception as e:
                        logger.error(f"削除エラー {img_path} (no_post_crop_landmarks): {e}")
                    continue

                # 移動（鼻を中央に）
                landmarks = results_post_crop.multi_face_landmarks[0].landmark
                face_h, face_w = face_image.shape[:2]
                nose = (landmarks[NOSE_INDEX].x * face_w, landmarks[NOSE_INDEX].y * face_h)
                shift_x = (TARGET_NOSE_X * face_w / img_size) - nose[0]
                shift_y = (TARGET_NOSE_Y * face_h / img_size) - nose[1]
                M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                shifted_image = cv2.warpAffine(face_image, M_shift, (face_w, face_h))

                # 移動後：ランドマーク再検出
                shifted_rgb = cv2.cvtColor(shifted_image, cv2.COLOR_BGR2RGB)
                results_shifted = face_mesh.process(shifted_rgb)
                if not results_shifted or not results_shifted.multi_face_landmarks:
                    skip_counters['no_post_crop_landmarks'] += 1
                    logger.info(f"移動後：ランドマーク再検出失敗 {filename}")
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters['deleted_no_post_crop_landmarks'] += 1
                            logger.info(f"削除：{img_path} (no_post_crop_landmarks)")
                        else:
                            logger.info(f"ファイル {img_path} が見つかりません (no_post_crop_landmarks)")
                    except Exception as e:
                        logger.error(f"削除エラー {img_path} (no_post_crop_landmarks): {e}")
                    continue

                # その他フィルタリング
                is_valid, reason = filter_other_conditions(shifted_image, img_path, filename, skip_counters)
                if not is_valid:
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters[f'deleted_{reason}'] += 1
                            logger.info(f"削除：{img_path} ({reason})")
                        else:
                            logger.info(f"ファイル {img_path} が見つかりません ({reason})")
                    except Exception as e:
                        logger.error(f"削除エラー {img_path} ({reason}): {e}")
                    continue

                # グレースケール変換とリサイズ
                gray = cv2.cvtColor(shifted_image, cv2.COLOR_BGR2GRAY)
                face_image_resized = cv2.resize(gray, (img_size, img_size))

                # ランドマーク描画（頬と顎）
                landmarks = results_shifted.multi_face_landmarks[0].landmark
                face_image_with_landmarks = cv2.cvtColor(face_image_resized, cv2.COLOR_GRAY2BGR)
                landmark_points = [
                    (RIGHT_CONTOUR_INDEX, (landmarks[RIGHT_CONTOUR_INDEX].x * img_size, landmarks[RIGHT_CONTOUR_INDEX].y * img_size)),
                    (LEFT_CONTOUR_INDEX, (landmarks[LEFT_CONTOUR_INDEX].x * img_size, landmarks[LEFT_CONTOUR_INDEX].y * img_size)),
                    (CHIN_INDEX, (landmarks[CHIN_INDEX].x * img_size, landmarks[CHIN_INDEX].y * img_size))
                ]
                for _, (x, y) in landmark_points:
                    cv2.circle(face_image_with_landmarks, (int(x), int(y)), 1, (255, 0, 0), -1)

                # 保存（ランドマーク付きBGRをグレースケールに変換）
                output_path = os.path.join(category_output_dir, filename)
                cv2.imwrite(output_path, cv2.cvtColor(face_image_with_landmarks, cv2.COLOR_BGR2GRAY))

    for reason, count in skip_counters.items():
        if total_images > 0:
            rate = count / total_images * 100
            logger.info(f"{input_dir}: {reason} {count}/{total_images} ({rate:.1f}%)")

def delete_invalid_preprocessed_images(input_dir):
    """前処理済み画像をフィルタリングし、無効な画像を削除"""
    skip_counters = {
        'no_face': 0, 'no_landmarks': 0, 'cheek_nose_distance': 0,
        'y_coordinate_diff': 0, 'small_chin_y': 0, 'multiple_faces': 0,
        'deleted_no_face': 0, 'deleted_no_landmarks': 0,
        'deleted_cheek_nose_distance': 0, 'deleted_y_coordinate_diff': 0,
        'deleted_small_chin_y': 0, 'deleted_multiple_faces': 0
    }
    total_images = 0

    for category in ['category1', 'category2']:
        category_input_dir = os.path.join(input_dir, category)
        if not os.path.exists(category_input_dir):
            continue

        for root, dirs, files in os.walk(category_input_dir):
            for filename in files:
                total_images += 1
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # 頬-鼻距離＆複数顔チェック（グレースケール画像をカラーに変換）
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                is_valid, reason = filter_cheek_nose_distance(img_color, img_path, filename, skip_counters)
                if not is_valid:
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters[f'deleted_{reason}'] += 1
                            logger.info(f"削除：{img_path} ({reason})")
                        else:
                            logger.info(f"ファイル {img_path} が見つかりません ({reason})")
                    except Exception as e:
                        logger.error(f"削除エラー {img_path} ({reason}): {e}")
                    continue

                # その他のフィルタリング
                is_valid, reason = filter_other_conditions(img, img_path, filename, skip_counters, is_preprocessed=True)
                if not is_valid:
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            skip_counters[f'deleted_{reason}'] += 1
                            logger.info(f"削除：{img_path} ({reason})")
                        else:
                            logger.info(f"ファイル {img_path} が見つかりません ({reason})")
                    except Exception as e:
                        logger.error(f"削除エラー {img_path} ({reason}): {e}")
                    continue

    for reason, count in skip_counters.items():
        if total_images > 0:
            rate = count / total_images * 100
            logger.info(f"{input_dir}: {reason} {count}/{total_images} ({rate:.1f}%)")

def extract_landmarks(img_path):
    """画像から顔ランドマークを抽出（2D座標のみ）"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            logger.error(f"画像読み込み失敗: {img_path}")
            return None
        img_rgb = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        if not results or not results.multi_face_landmarks:
            logger.info(f"ランドマーク検出失敗 {img_path}")
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        coords = [(l.x * img_size, l.y * img_size) for l in landmarks]
        nose_center = coords[1]
        coords = [(x - nose_center[0], y - nose_center[1]) for x, y in coords]
        return np.array(coords)
    except Exception as e:
        logger.error(f"処理エラー {img_path}: {e}")
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
        return float('inf')

def compute_color_ratios(img_path):
    """画像を4分割し、各領域の黒・灰・白の割合を計算"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            logger.error(f"画像読み込み失敗: {img_path}")
            return None, None, None, None, None, None, None, None, None, None, None, None
        left_upper = img[:img_size//2, :img_size//2]
        right_upper = img[:img_size//2, img_size//2:]
        left_lower = img[img_size//2:, :img_size//2]
        right_lower = img[img_size//2:, img_size//2:]
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
        rl_gray_pixels = np.sum((right_lower >= GRAY_MIN) & (right_lower <= GRAY_MAX))
        rl_white_pixels = np.sum(right_lower >= WHITE_MIN)
        rl_black_ratio = rl_black_pixels / total_pixels
        rl_gray_ratio = rl_gray_pixels / total_pixels
        rl_white_ratio = rl_white_pixels / total_pixels
        return (lu_black_ratio, lu_gray_ratio, lu_white_ratio,
                ru_black_ratio, ru_gray_ratio, ru_white_ratio,
                ll_black_ratio, ll_gray_ratio, ll_white_ratio,
                rl_black_ratio, rl_gray_ratio, rl_white_ratio)
    except Exception as e:
        logger.error(f"処理エラー {img_path}: {e}")
        return None, None, None, None, None, None, None, None, None, None, None, None

def get_filename_prefix(filename, prefix_length=3):
    """ファイル名の先頭3文字を取得"""
    try:
        prefix = os.path.basename(filename)[:prefix_length]
        return prefix
    except Exception as e:
        logger.error(f"プレフィックス取得エラー {filename}: {e}")
        return ""

def find_similar_images(train_dir, color_ratio_threshold=COLOR_RATIO_THRESHOLD, tilt_threshold=TILT_THRESHOLD, top_n=TOP_N):
    """先頭3文字一致かつ色割合・傾き差が小さい画像グループを特定し、ランドマーク非検出および類似画像を前処理済み画像から削除"""
    logger.info(f"{train_dir} の類似画像検索開始")
    image_files = []
    for category in ['category1', 'category2']:
        cat_dir = os.path.join(train_dir, category)
        if not os.path.exists(cat_dir):
            logger.info(f"ディレクトリ {cat_dir} が見つかりません")
            continue
        for filename in os.listdir(cat_dir):
            img_path = os.path.join(cat_dir, filename)
            image_files.append((img_path, category))
    
    logger.info(f"{train_dir} で {len(image_files)} 画像を検出")
    
    prefix_groups = defaultdict(list)
    for img_path, category in image_files:
        prefix = get_filename_prefix(img_path)
        prefix_groups[prefix].append((img_path, category))
    
    for prefix, images in prefix_groups.items():
        logger.info(f"プレフィックス {prefix}: {len(images)} 画像")
    
    image_data = {}
    skip_counters = {'deleted_no_landmarks': 0}
    for img_path, category in image_files:
        ratios = compute_color_ratios(img_path)
        landmarks = extract_landmarks(img_path)
        if landmarks is None:
            skip_counters['deleted_no_landmarks'] += 1
            logger.info(f"ランドマーク検出失敗により削除 {img_path}")
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    logger.info(f"成功的に削除（ランドマーク検出失敗）: {img_path}")
                else:
                    logger.info(f"ファイルが見つかりません（ランドマーク検出失敗）: {img_path}")
            except Exception as e:
                logger.error(f"削除エラー（ランドマーク検出失敗） {img_path}: {e}")
            continue
        if all(r is not None for r in ratios):
            image_data[img_path] = {
                'category': category,
                'ratios': ratios,
                'landmarks': landmarks
            }
        else:
            logger.info(f"無効な比率のため {img_path} をスキップ")
    
    logger.info(f"有効な特徴を持つ画像: {len(image_data)}")
    for reason, count in skip_counters.items():
        logger.info(f"{train_dir}: {reason} {count}/{len(image_files)} ({count/len(image_files)*100:.1f}%)")
    
    groups = []
    used_images = set()
    
    for prefix, images in prefix_groups.items():
        if len(images) < 3:
            logger.info(f"プレフィックス {prefix} をスキップ: 画像 {len(images)} のみ")
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
                    logger.info(f"{img1_path} で空の current_diffs")
                
                groups.append({
                    'images': current_group,
                    'avg_diffs': avg_diffs
                })
                for img_path, _ in current_group:
                    used_images.add(img_path)
    
    logger.info(f"{len(groups)} グループを検出")
    
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
        return
    
    logger.info(f"上位 {min(top_n, len(groups))} 類似画像グループ:")
    displayed_groups = 0
    for group in groups[:top_n]:
        images = group['images']
        logger.info(f"グループ {displayed_groups+1}")
        for img_path, _ in images:
            logger.info(f"  {img_path}")
        displayed_groups += 1
        if displayed_groups >= top_n:
            break
    
    if displayed_groups == 0:
        logger.info("表示するグループなし")
    
    csv_data = []
    deleted_images = set()
    logger.info("画像削除処理開始（前処理済みのみ）")
    for group_id, group in enumerate(groups):
        images = group['images']
        if len(images) < 1:
            continue
        keep_img_path, keep_category = images[0]
        logger.info(f"グループ {group_id+1}: 最初の画像を保持、その他を削除")
        logger.info(f"  保持: {keep_img_path}")
        
        csv_data.append({
            'group_id': group_id,
            'image_path': keep_img_path,
            'category': keep_category,
            **group['avg_diffs']
        })
        
        for img_path, category in images[1:]:
            logger.info(f"  前処理済み削除: {img_path}")
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    deleted_images.add(img_path)
                    logger.info(f"  成功的に削除（前処理済み）: {img_path}")
                else:
                    logger.info(f"  ファイルが見つかりません（前処理済み）: {img_path}")
            except Exception as e:
                logger.error(f"  削除失敗（前処理済み） {img_path}: {e}")
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"結果を {OUTPUT_CSV} に保存")
    else:
        logger.info("CSVに保存する画像なし")

def main():
    try:
        logger.info("前処理と重複除去開始")
        preprocess_and_cut_faces(TRAIN_DIR, PREPROCESSED_TRAIN_DIR)
        preprocess_and_cut_faces(VALIDATION_DIR, PREPROCESSED_VALIDATION_DIR)
        # delete_invalid_preprocessed_images(PREPROCESSED_TRAIN_DIR)
        # delete_invalid_preprocessed_images(PREPROCESSED_VALIDATION_DIR)
        find_similar_images(PREPROCESSED_TRAIN_DIR)
        find_similar_images(PREPROCESSED_VALIDATION_DIR)
        logger.info("前処理と重複除去完了")
    except Exception as e:
        logger.error(f"メインエラー: {e}")
    finally:
        face_detection.close()
        face_mesh.close()

if __name__ == "__main__":
    main()