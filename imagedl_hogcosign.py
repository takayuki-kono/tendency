import os
import cv2
import numpy as np
import random
import logging
import shutil
from icrawler.builtin import GoogleImageCrawler
import mediapipe as mp
from insightface.app import FaceAnalysis
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import DBSCAN
from collections import defaultdict
import face_recognition
from person_classification.clustering_utils import perform_dbscan_clustering

KEYWORD = "安藤サクラ"
MAX_NUM = 100
OUTPUT_DIR = str(random.randint(0, 1000)).zfill(4)
SIMILARITY_THRESHOLD = 0.7  # SSIM threshold (0 to 1, higher means more similar)
IMG_SIZE = 224

logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # ログフォーマットにレベルを追加
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# MediapipeとInsightFaceの初期化
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
logger.info("Initializing FaceAnalysis")
try:
    app = FaceAnalysis(providers=['CPUExecutionProvider'], det_thresh=0.3)
    app.prepare(ctx_id=0, det_size=(320, 320))
    logger.info("FaceAnalysis initialized successfully")
except Exception as e:
    logger.error(f"FaceAnalysis initialization failed: {str(e)}")
    raise

def setup_crawler(storage_dir):
    return GoogleImageCrawler(storage={'root_dir': storage_dir})

def download_images(keyword, max_num):
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
        crawler = setup_crawler(storage_dir)
        crawler.crawl(keyword=search_keyword, max_num=max_num)
        downloaded_files = [f for f in os.listdir(storage_dir) if os.path.isfile(os.path.join(storage_dir, f))]
        logger.info(f"Downloaded {len(downloaded_files)} images for {search_keyword}")

def rename_files(keyword):
    folders = [keyword, f"{keyword}_昔", f"{keyword}_現在", f"{keyword}_正面", f"{keyword}_顔"]
    for folder in folders:
        if not os.path.exists(folder):
            logger.warning(f"Folder {folder} does not exist, skipping rename")
            continue
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            old_path = os.path.join(folder, file)
            new_filename = f"{folder}_{file}"
            new_path = os.path.join(folder, new_filename)
            try:
                os.rename(old_path, new_path)
                logger.info(f"Renamed {old_path} to {new_path}")
            except Exception as e:
                logger.error(f"Error renaming {old_path} to {new_path}: {e}")

def consolidate_files():
    output_dir = OUTPUT_DIR
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    folders = [KEYWORD, f"{KEYWORD}_昔", f"{KEYWORD}_現在", f"{KEYWORD}_正面", f"{KEYWORD}_顔"]
    for folder in folders:
        if not os.path.exists(folder):
            logger.warning(f"Folder {folder} does not exist, skipping consolidation")
            continue
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            src_path = os.path.join(folder, file)
            dst_path = os.path.join(output_dir, file)
            try:
                shutil.move(src_path, dst_path)
                logger.info(f"Moved {src_path} to {dst_path}")
            except Exception as e:
                logger.error(f"Error moving {src_path} to {dst_path}: {e}")
    for folder in folders:
        if os.path.exists(folder) and not os.listdir(folder):
            shutil.rmtree(folder)
            logger.info(f"Removed empty folder {folder}")
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    for i, file in enumerate(files, 1):
        old_path = os.path.join(output_dir, file)
        ext = os.path.splitext(file)[1].lower()
        new_filename = f"{OUTPUT_DIR}_{i:03d}{ext}"
        new_path = os.path.join(output_dir, new_filename)
        try:
            os.rename(old_path, new_path)
            logger.info(f"Renamed {old_path} to {new_path}")
        except Exception as e:
            logger.error(f"Error renaming {old_path} to {new_path}: {e}")

def detect_and_crop_faces(input_dir):
    resized_dir = os.path.join(input_dir, "resized")
    processed_dir = os.path.join(input_dir, "processed")
    rotated_dir = os.path.join(input_dir, "rotated")
    bbox_cropped_dir = os.path.join(input_dir, "bbox_cropped")
    bbox_rotated_dir = os.path.join(input_dir, "bbox_rotated")
    if os.path.exists(resized_dir):
        shutil.rmtree(resized_dir)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    if os.path.exists(rotated_dir):
        shutil.rmtree(rotated_dir)
    if os.path.exists(bbox_cropped_dir):
        shutil.rmtree(bbox_cropped_dir)
    if os.path.exists(bbox_rotated_dir):
        shutil.rmtree(bbox_rotated_dir)
    os.makedirs(resized_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(rotated_dir, exist_ok=True)
    os.makedirs(bbox_cropped_dir, exist_ok=True)
    os.makedirs(bbox_rotated_dir, exist_ok=True)
    skip_counters = {'no_face': 0, 'deleted_no_face': 0}
    total_images = 0
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    processed_face_to_original_map = {}

    for filename in files:
        total_images += 1
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            skip_counters['no_face'] += 1
            logger.info(f"画像読み込み失敗 {img_path}")
            try:
                os.remove(img_path)
                skip_counters['deleted_no_face'] += 1
                logger.info(f"削除：{img_path} (no_face)")
            except Exception as e:
                logger.error(f"削除エラー {img_path} (no_face): {e}")
            continue

        # Mediapipe でバウンディングボックス検出
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detector.process(img_rgb)
        if not results.detections:
            skip_counters['no_face'] += 1
            logger.info(f"Mediapipeで顔検出失敗 {filename}")
            try:
                os.remove(img_path)
                skip_counters['deleted_no_face'] += 1
                logger.info(f"削除：{img_path} (no_face)")
            except Exception as e:
                logger.error(f"削除エラー {img_path} (no_face): {e}")
            continue

        at_least_one_face_processed = False
        for face_idx, detection in enumerate(results.detections):
            try:
                original_base_name, ext = os.path.splitext(filename)
                current_face_base_name = f"{original_base_name}_{face_idx}"

                bbox = detection.location_data.relative_bounding_box
                h, w = img.shape[:2]
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                x_max = int((bbox.xmin + bbox.width) * w)
                y_max = int((bbox.ymin + bbox.height) * h)
                x_min = max(0, x_min - int(0.1 * (x_max - x_min)))  # 10% 拡大
                y_min = max(0, y_min - int(0.1 * (y_max - y_min)))
                x_max = min(w, x_max + int(0.1 * (x_max - x_min)))
                y_max = min(h, y_max + int(0.1 * (y_max - y_min)))

                # バウンディングボックスで切り抜き
                face_img = img[y_min:y_max, x_min:x_max]
                if face_img is None or face_img.size == 0:
                    skip_counters['no_face'] += 1
                    logger.info(f"顔領域切り取り失敗 {filename} (face_idx: {face_idx})")
                    continue
                logger.info(f"バウンディングボックス画像サイズ: {face_img.shape} for {filename} (face_idx: {face_idx})")

                # バウンディングボックス画像を保存
                bbox_filename = f"{current_face_base_name}{ext}"
                bbox_path = os.path.join(bbox_cropped_dir, bbox_filename)
                cv2.imwrite(bbox_path, face_img)
                logger.info(f"バウンディングボックス画像保存：{bbox_path}")

                # バウンディングボックス画像からランドマーク検出
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(face_img_rgb)
                mp_landmarks = None
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0].landmark
                    mp_landmarks = {
                        'left_eyebrow': np.array([landmarks[105].x * face_img.shape[1], landmarks[105].y * face_img.shape[0]]),
                        'right_eyebrow': np.array([landmarks[334].x * face_img.shape[1], landmarks[334].y * face_img.shape[0]]),
                        'chin': np.array([landmarks[152].x * face_img.shape[1], landmarks[152].y * face_img.shape[0]]),
                        'nose': np.array([landmarks[1].x * face_img.shape[1], landmarks[1].y * face_img.shape[0]])
                    }

                faces = app.get(face_img_rgb)
                ins_landmarks = None
                if faces:
                    lmk = faces[0].landmark_2d_106
                    ins_landmarks = {
                        'left_eyebrow': np.array([lmk[49][0], lmk[49][1]]),
                        'right_eyebrow': np.array([lmk[104][0], lmk[104][1]]),
                        'chin': np.array([lmk[0][0], lmk[0][1]]),
                        'nose': np.array([lmk[86][0], lmk[86][1]])
                    }
                    logger.info(f"InsightFace landmarks detected for {filename} (face_idx: {face_idx}): {ins_landmarks}")
                else:
                    skip_counters['no_face'] += 1
                    logger.warning(f"InsightFace failed to detect face for {filename} (face_idx: {face_idx})")
                    continue

                if mp_landmarks is None or ins_landmarks is None:
                    skip_counters['no_face'] += 1
                    logger.info(f"ランドマーク取得失敗 {filename} (face_idx: {face_idx})")
                    continue

                # ランドマークの平均を計算
                landmarks = {
                    'left_eyebrow': (mp_landmarks['left_eyebrow'] + ins_landmarks['left_eyebrow']) / 2,
                    'right_eyebrow': (mp_landmarks['right_eyebrow'] + ins_landmarks['right_eyebrow']) / 2,
                    'chin': (mp_landmarks['chin'] + ins_landmarks['chin']) / 2,
                    'nose': (mp_landmarks['nose'] + ins_landmarks['nose']) / 2
                }

                # バウンディングボックス画像の傾き修正
                dx = landmarks['nose'][0] - landmarks['chin'][0]
                dy = landmarks['nose'][1] - landmarks['chin'][1]
                angle = np.arctan2(dx, -dy) * 180 / np.pi
                center = (face_img.shape[1] / 2, face_img.shape[0] / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                bbox_rotated_img = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))

                # 傾き修正後のバウンディングボックス画像を保存 (ファイル名は変更しない)
                bbox_rotated_filename = f"{current_face_base_name}{ext}"
                bbox_rotated_path = os.path.join(bbox_rotated_dir, bbox_rotated_filename)
                cv2.imwrite(bbox_rotated_path, bbox_rotated_img)
                logger.info(f"傾き修正バウンディングボックス画像保存：{bbox_rotated_path}")

                # バウンディングボックス画像から切り抜き
                y_top = min(landmarks['left_eyebrow'][1], landmarks['right_eyebrow'][1])
                y_bottom = landmarks['chin'][1]
                x_center = landmarks['nose'][0]
                size = max(y_bottom - y_top, face_img.shape[1] // 2)
                x_min_crop = int(x_center - size // 2)
                x_max_crop = int(x_center + size // 2)
                y_min_crop = int(y_top)
                y_max_crop = int(y_top + size)

                # パディング処理
                h, w = face_img.shape[:2]
                pad_top = max(0, -y_min_crop)
                pad_bottom = max(0, y_max_crop - h)
                pad_left = max(0, -x_min_crop)
                pad_right = max(0, x_max_crop - w)
                if pad_top or pad_bottom or pad_left or pad_right:
                    face_img = cv2.copyMakeBorder(face_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    landmarks = {k: v + np.array([pad_left, pad_top]) for k, v in landmarks.items()}
                    x_min_crop += pad_left
                    x_max_crop += pad_left
                    y_min_crop += pad_top
                    y_max_crop += pad_top

                face_image = face_img[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
                if face_image is None or face_image.size == 0:
                    skip_counters['no_face'] += 1
                    logger.info(f"顔領域切り取り失敗 {filename} (face_idx: {face_idx})")
                    continue
                face_image_resized = cv2.resize(face_image, (IMG_SIZE, IMG_SIZE))

                # ランドマークを描画
                # # ランドマーク表示を有効にする場合は、以下のブロックのコメントを解除してください
                # scale_x = IMG_SIZE / (x_max_crop - x_min_crop)
                # scale_y = IMG_SIZE / (y_max_crop - y_min_crop)
                # for point in ['left_eyebrow', 'right_eyebrow', 'chin', 'nose']:
                #     x = int((landmarks[point][0] - x_min_crop) * scale_x)
                #     y = int((landmarks[point][1] - y_min_crop) * scale_y)
                #     if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
                #         cv2.circle(face_image_resized, (x, y), 3, (0, 0, 255), -1)
                #     else:
                #         logger.warning(f"Invalid landmark position for {point} in {filename} (face_idx: {face_idx}): ({x}, {y})")

                resized_filename = f"{current_face_base_name}{ext}" # ファイル名は変更しない
                resized_path = os.path.join(resized_dir, resized_filename)
                cv2.imwrite(resized_path, face_image_resized)
                logger.info(f"リサイズ画像保存：{resized_path}")

                gray = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2GRAY)
                if gray.shape != (IMG_SIZE, IMG_SIZE):
                    logger.error(f"無効な画像サイズ: {filename} (face_idx: {face_idx})")
                    continue
                processed_filename = f"{current_face_base_name}.png" # グレースケールは常に.png
                processed_path = os.path.join(processed_dir, processed_filename)
                cv2.imwrite(processed_path, gray)
                logger.info(f"グレースケール画像保存：{processed_path}")

                # 回転後のバウンディングボックス画像から顔を切り抜く
                rotated_img = cv2.imread(bbox_rotated_path)
                if rotated_img is None:
                    logger.info(f"回転後バウンディングボックス画像読み込み失敗 {bbox_rotated_path}")
                    continue
                rotated_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(rotated_img_rgb)
                mp_rot_landmarks = None
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0].landmark
                    mp_rot_landmarks = {
                        'left_eyebrow': np.array([landmarks[105].x * rotated_img.shape[1], landmarks[105].y * rotated_img.shape[0]]),
                        'right_eyebrow': np.array([landmarks[334].x * rotated_img.shape[1], landmarks[334].y * rotated_img.shape[0]]),
                        'chin': np.array([landmarks[152].x * rotated_img.shape[1], landmarks[152].y * rotated_img.shape[0]]),
                        'nose': np.array([landmarks[1].x * rotated_img.shape[1], landmarks[1].y * rotated_img.shape[0]])
                    }

                faces = app.get(rotated_img_rgb)
                ins_rot_landmarks = None
                if faces:
                    lmk = faces[0].landmark_2d_106
                    ins_rot_landmarks = {
                        'left_eyebrow': np.array([lmk[49][0], lmk[49][1]]),
                        'right_eyebrow': np.array([lmk[104][0], lmk[104][1]]),
                        'chin': np.array([lmk[0][0], lmk[0][1]]),
                        'nose': np.array([lmk[86][0], lmk[86][1]])
                    }
                    logger.info(f"InsightFace landmarks detected for rotated {filename} (face_idx: {face_idx}): {ins_rot_landmarks}")
                else:
                    logger.warning(f"InsightFace failed to detect face for rotated {filename} (face_idx: {face_idx})")
                    continue

                if mp_rot_landmarks is None or ins_rot_landmarks is None:
                    logger.info(f"回転後ランドマーク取得失敗 {filename} (face_idx: {face_idx})")
                    continue

                rot_landmarks = {
                    'left_eyebrow': (mp_rot_landmarks['left_eyebrow'] + ins_rot_landmarks['left_eyebrow']) / 2,
                    'right_eyebrow': (mp_rot_landmarks['right_eyebrow'] + ins_rot_landmarks['right_eyebrow']) / 2,
                    'chin': (mp_rot_landmarks['chin'] + ins_rot_landmarks['chin']) / 2,
                    'nose': (mp_rot_landmarks['nose'] + ins_rot_landmarks['nose']) / 2
                }

                # 回転後バウンディングボックス画像の切り抜き
                ry_top = min(rot_landmarks['left_eyebrow'][1], rot_landmarks['right_eyebrow'][1])
                ry_bottom = rot_landmarks['chin'][1]
                rx_center = rot_landmarks['nose'][0]
                r_size = max(ry_bottom - ry_top, rotated_img.shape[1] // 2)
                rx_min_crop = int(rx_center - r_size // 2)
                rx_max_crop = int(rx_center + r_size // 2)
                ry_min_crop = int(ry_top)
                ry_max_crop = int(ry_top + r_size)

                rh, rw = rotated_img.shape[:2]
                r_pad_top = max(0, -ry_min_crop)
                r_pad_bottom = max(0, ry_max_crop - rh)
                r_pad_left = max(0, -rx_min_crop)
                r_pad_right = max(0, rx_max_crop - rw)
                if r_pad_top or r_pad_bottom or r_pad_left or r_pad_right:
                    rotated_img = cv2.copyMakeBorder(rotated_img, r_pad_top, r_pad_bottom, r_pad_left, r_pad_right,
                                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    rot_landmarks = {k: v + np.array([r_pad_left, r_pad_top]) for k, v in rot_landmarks.items()}
                    rx_min_crop += r_pad_left
                    rx_max_crop += r_pad_left
                    ry_min_crop += r_pad_top
                    ry_max_crop += r_pad_top

                rotated_face = rotated_img[ry_min_crop:ry_max_crop, rx_min_crop:rx_max_crop]
                if rotated_face is None or rotated_face.size == 0:
                    logger.info(f"回転後顔領域切り取り失敗 {filename} (face_idx: {face_idx})")
                    continue
                rotated_face_resized = cv2.resize(rotated_face, (IMG_SIZE, IMG_SIZE))

                # # ランドマーク表示を有効にする場合は、以下のブロックのコメントを解除してください
                # r_scale_x = IMG_SIZE / (rx_max_crop - rx_min_crop)
                # r_scale_y = IMG_SIZE / (ry_max_crop - ry_min_crop)
                # for point in ['left_eyebrow', 'right_eyebrow', 'chin', 'nose']:
                #     rx = int((rot_landmarks[point][0] - rx_min_crop) * r_scale_x)
                #     ry = int((rot_landmarks[point][1] - ry_min_crop) * r_scale_y)
                #     if 0 <= rx < IMG_SIZE and 0 <= ry < IMG_SIZE:
                #         cv2.circle(rotated_face_resized, (rx, ry), 3, (0, 255, 0), 1)
                #     else:
                #         logger.warning(f"Invalid landmark position for {point} in {filename} (face_idx: {face_idx}): ({rx}, {ry})")

                rotated_filename = f"{current_face_base_name}{ext}" # ファイル名は変更しない
                rotated_path = os.path.join(rotated_dir, rotated_filename)
                cv2.imwrite(rotated_path, rotated_face_resized)
                logger.info(f"回転画像保存：{rotated_path}")

                processed_face_to_original_map[current_face_base_name] = filename
                at_least_one_face_processed = True

            except Exception as e:
                logger.error(f"顔処理中にエラーが発生しました {filename} (face_idx: {face_idx}): {e}")
                pass

        if not at_least_one_face_processed:
            skip_counters['no_face'] += 1
            logger.info(f"全ての顔処理が失敗したため、オリジナル画像を削除 {img_path}")
            try:
                os.remove(img_path)
                skip_counters['deleted_no_face'] += 1
                logger.info(f"削除：{img_path} (all_faces_failed)")
            except Exception as e:
                logger.error(f"削除エラー {img_path} (all_faces_failed): {e}")

    for reason, count in sorted(skip_counters.items()):
        rate = count / total_images * 100 if total_images > 0 else 0
        logger.info(f"{input_dir}: {reason} {count}/{total_images} ({rate:.1f}%)")

    return processed_face_to_original_map

def find_similar_images(input_dir, processed_face_to_original_map):
    processed_dir = os.path.join(input_dir, "processed")
    resized_dir = os.path.join(input_dir, "resized")
    rotated_dir = os.path.join(input_dir, "rotated")
    bbox_cropped_dir = os.path.join(input_dir, "bbox_cropped")
    bbox_rotated_dir = os.path.join(input_dir, "bbox_rotated")
    deleted_dir = os.path.join(input_dir, "deleted")
    if os.path.exists(deleted_dir):
        shutil.rmtree(deleted_dir)
    os.makedirs(deleted_dir, exist_ok=True)
    logger.info(f"{processed_dir} の類似画像検索開始")
    image_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if os.path.isfile(os.path.join(processed_dir, f))]
    logger.info(f"{os.path.basename(processed_dir)} で {len(image_files)} 画像を検出")

    def compare_images(img_path1, img_path2):
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None or img1.shape != img2.shape:
            logger.error(f"比較失敗: {img_path1} vs {img_path2}")
            return 0.0
        score = ssim(img1, img2)
        logger.info(f"SSIM計算: {img_path1} vs {img_path2}, score={score:.3f}")
        return score

    groups = []
    used_images = set()
    for i, img1_path in enumerate(image_files):
        if img1_path in used_images:
            continue
        current_group = [img1_path]
        for j, img2_path in enumerate(image_files[i+1:], i+1):
            if img2_path in used_images:
                continue
            score = compare_images(img1_path, img2_path)
            if score >= SIMILARITY_THRESHOLD:
                current_group.append(img2_path)
                logger.info(f"類似画像検出: {img1_path} と {img2_path} (SSIM={score:.3f})")
        if len(current_group) > 1:
            groups.append(current_group)
            used_images.update(current_group)
        else:
            used_images.add(img1_path)

    logger.info(f"{len(groups)} グループを検出")
    for group_idx, group in enumerate(groups, 1):
        logger.info(f"グループ {group_idx} 表示開始: {group}")
        group_images = []
        for img_path in group:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                group_images.append(img)
                logger.info(f"表示用画像読み込み: {img_path}")
            else:
                logger.error(f"画像読み込み失敗: {img_path}")
        if not group_images:
            logger.warning(f"グループ {group_idx} に表示可能な画像がありません")
            continue
        max_height = max(img.shape[0] for img in group_images)
        resized_images = [cv2.resize(img, (IMG_SIZE, max_height)) for img in group_images]
        display_image = np.hstack(resized_images)
        window_name = f"Group {group_idx}"
        try:
            cv2.imshow(window_name, display_image)
            logger.info(f"グループ {group_idx} を表示: {window_name}")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(window_name)
            logger.info(f"キー入力: {group_idx} for group {group_idx} {key}")
            if key == 27:
                logger.info("ユーザーにより中断（Escキー）")
                break
        except Exception as e:
            logger.error(f"グループ {group_idx} 表示エラー: {e}")
    cv2.destroyAllWindows()
    logger.info("類似画像グループの表示完了")

    logger.info("類似画像処理開始")
    for group_idx, group in enumerate(groups, 1):
        if len(group) < 2:
            continue
        keep_img_path = group[0]
        logger.info(f"グループ {group_idx}: 保持: {keep_img_path}")
        for img_path in group[1:]:
            logger.info(f"削除対象: {img_path} (processed)")
            try:
                base_name = os.path.splitext(os.path.basename(img_path))[0] # 例: "0952_001_0"
                
                original_filename_with_ext = processed_face_to_original_map.get(base_name)
                if not original_filename_with_ext:
                    logger.warning(f"オリジナルファイル名が見つかりません。スキップ: {base_name}")
                    continue
                _, original_ext = os.path.splitext(original_filename_with_ext)

                files_to_delete = [
                    os.path.join(processed_dir, f"{base_name}.png"), # processed image is always .png
                    os.path.join(resized_dir, f"{base_name}{original_ext}"),
                    os.path.join(rotated_dir, f"{base_name}{original_ext}"),
                    os.path.join(bbox_cropped_dir, f"{base_name}{original_ext}"),
                    os.path.join(bbox_rotated_dir, f"{base_name}{original_ext}"),
                    os.path.join(input_dir, original_filename_with_ext) # Original image
                ]

                for file_path_to_move in files_to_delete:
                    if os.path.exists(file_path_to_move):
                        destination_path = os.path.join(deleted_dir, os.path.basename(file_path_to_move))
                        shutil.move(file_path_to_move, destination_path)
                        logger.info(f"移動: {file_path_to_move} -> {destination_path}")
                    else:
                        logger.info(f"ファイルが見つかりません（既に削除済みか、存在しない）: {file_path_to_move}")
            except Exception as e:
                logger.error(f"処理エラー {img_path}: {e}")

def filter_by_main_person(input_dir, processed_face_to_original_map):
    """
    rotatedフォルダの画像を人物ごとにグループ分けし、最も画像数の多い人物以外の画像を削除する。
    face_recognitionライブラリを使用。
    """
    # 注意: この関数は 'face_recognition' ライブラリを使用します。
    # 事前に 'pip install cmake dlib face_recognition' を実行してインストールが必要です。
    # dlibのインストールに失敗する場合は、Visual Studio C++ Build Toolsのインストールが必要な場合があります。

    logger.info("人物ごとの画像グループ分けとフィルタリングを開始 (face_recognition)")

    # 処理対象の画像はrotatedフォルダから取得
    rotated_dir = os.path.join(input_dir, "rotated")
    if not os.path.exists(rotated_dir) or not os.listdir(rotated_dir):
        logger.warning(f"rotatedディレクトリが見つからないか空です: {rotated_dir}。人物フィルタリングをスキップします。")
        return

    image_files = [f for f in os.listdir(rotated_dir) if os.path.isfile(os.path.join(rotated_dir, f))]
    
    encodings = []
    image_path_list = []

    # 1. 全ての残存画像から顔のエンコーディングを抽出
    logger.info("顔のエンコーディングを抽出中...")
    for filename in image_files:
        img_path = os.path.join(rotated_dir, filename)
        try:
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                encodings.append(face_encodings[0])
                image_path_list.append(img_path)
            else:
                logger.warning(f"face_recognitionが顔を検出できませんでした: {img_path}")
        except Exception as e:
            logger.error(f"face_recognition処理中にエラーが発生しました {img_path}: {e}")
            continue

    if len(encodings) < 2:
        logger.info("クラスタリング対象の画像が2枚未満のため、人物フィルタリングを終了します。")
        return

    encodings_np = np.array(encodings)
    
    # 2. DBSCANでエンコーディングをクラスタリング
    logger.info(f"{len(encodings_np)}個のエンコーディングをクラスタリングします...")
    clt = DBSCAN(metric="euclidean", eps=0.30, min_samples=1)
    clt.fit(encodings_np)
    labels = clt.labels_

    # 3. 最大クラスタ（主要人物）を特定
    cluster_summary = dict(zip(*np.unique(labels, return_counts=True)))
    logger.info(f"クラスタリング結果 (ラベル: 画像数): {cluster_summary}")

    core_labels = labels[labels != -1]
    if len(core_labels) == 0:
        logger.warning("主要な人物クラスタが見つかりませんでした。全ての画像がノイズと判断されたため、フィルタリングをスキップします。")
        return

    unique_core_labels, core_counts = np.unique(core_labels, return_counts=True)
    max_cluster_index = np.argmax(core_counts)
    main_cluster_label = unique_core_labels[max_cluster_index]
    main_cluster_size = core_counts[max_cluster_index]
    
    logger.info(f"主要な人物のクラスタラベル: {main_cluster_label} (画像数: {main_cluster_size})")

    MIN_CLUSTER_SIZE = 3
    MIN_CLUSTER_RATIO = 0.2

    if main_cluster_size < MIN_CLUSTER_SIZE or (main_cluster_size / len(image_files)) < MIN_CLUSTER_RATIO:
        logger.warning(f"最大クラスタのサイズ ({main_cluster_size}) が閾値（{MIN_CLUSTER_SIZE}枚 or 全体の{MIN_CLUSTER_RATIO*100:.0f}%）に満たないため、人物フィルタリングをスキップします。")
        return

    # 4. 削除対象の画像をグループごとに特定し、表示してから削除
    images_to_delete_by_group = defaultdict(list)
    all_images_to_delete_paths = []
    for i, label in enumerate(labels):
        if label != main_cluster_label:
            img_path = image_path_list[i]
            images_to_delete_by_group[label].append(img_path)
            all_images_to_delete_paths.append(img_path)

    if not all_images_to_delete_paths:
        logger.info("削除対象の異人物画像はありません。")
        return

    # グループごとに削除対象の画像を表示
    logger.info(f"{len(all_images_to_delete_paths)}枚の異人物画像を削除前に表示します。")
    MAX_IMAGES_PER_WINDOW = 25  # 5x5 グリッド
    COLS = 5

    should_skip_deletion = False
    for label, paths in sorted(images_to_delete_by_group.items()):
        if should_skip_deletion:
            break

        num_images_in_group = len(paths)
        num_windows = (num_images_in_group + MAX_IMAGES_PER_WINDOW - 1) // MAX_IMAGES_PER_WINDOW

        for part_num in range(num_windows):
            start_index = part_num * MAX_IMAGES_PER_WINDOW
            end_index = start_index + MAX_IMAGES_PER_WINDOW
            chunk_paths = paths[start_index:end_index]

            display_images = []
            for img_path in chunk_paths:
                img = cv2.imread(img_path)
                if img is not None:
                    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    display_images.append(resized_img)
                else:
                    logger.warning(f"表示用画像の読み込みに失敗: {img_path}")
            
            if not display_images:
                continue

            num_images = len(display_images)
            rows = (num_images + COLS - 1) // COLS
            montage = np.zeros((rows * IMG_SIZE, COLS * IMG_SIZE, 3), dtype=np.uint8)

            for i, img in enumerate(display_images):
                row, col = divmod(i, COLS)
                montage[row*IMG_SIZE:(row+1)*IMG_SIZE, col*IMG_SIZE:(col+1)*IMG_SIZE] = img

            window_title = f"Deleted - Group {label}"
            if num_windows > 1:
                window_title += f" (Part {part_num + 1}/{num_windows})"
            
            try:
                cv2.imshow(window_title, montage)
                logger.info(f"{window_title} を表示中。任意のキーを押して次へ。Escで全削除を中止。")
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow(window_title)
                if key == 27: # Escキー
                    logger.info("ユーザーにより表示が中断されました。削除処理をスキップします。")
                    should_skip_deletion = True
                    break
            except cv2.error as e:
                logger.error(f"画像表示エラー: {e}")

    if should_skip_deletion:
        return

    # 削除処理の実行
    deleted_dir = os.path.join(input_dir, "deleted")
    processed_dir = os.path.join(input_dir, "processed")
    resized_dir = os.path.join(input_dir, "resized")
    bbox_cropped_dir = os.path.join(input_dir, "bbox_cropped")
    bbox_rotated_dir = os.path.join(input_dir, "bbox_rotated")

    for img_path_to_delete in all_images_to_delete_paths:
        base_name = os.path.splitext(os.path.basename(img_path_to_delete))[0]
        
        original_filename_with_ext = processed_face_to_original_map.get(base_name)
        if not original_filename_with_ext:
            logger.warning(f"人物フィルタリング: オリジナルファイル名が見つかりません。スキップ: {base_name}")
            continue
        _, original_ext = os.path.splitext(original_filename_with_ext)

        files_to_move = [
            img_path_to_delete,
            os.path.join(processed_dir, f"{base_name}.png"),
            os.path.join(resized_dir, f"{base_name}{original_ext}"),
            os.path.join(bbox_cropped_dir, f"{base_name}{original_ext}"),
            os.path.join(bbox_rotated_dir, f"{base_name}{original_ext}"),
            os.path.join(input_dir, original_filename_with_ext)
        ]

        for file_path in files_to_move:
            if os.path.exists(file_path):
                destination_path = os.path.join(deleted_dir, os.path.basename(file_path))
                try:
                    shutil.move(file_path, destination_path)
                    logger.info(f"人物フィルタリングにより移動: {file_path} -> {destination_path}")
                except Exception as e:
                    logger.error(f"人物フィルタリング中の移動エラー {file_path}: {e}")
            else:
                logger.warning(f"人物フィルタリング: ファイルが見つかりません（移動済みか）: {file_path}")

def filter_by_main_person_cosine(input_dir, processed_face_to_original_map):
    """
    rotatedフォルダの画像を人物ごとにグループ分けし、最も大きいグループの画像のみ残す。
    face_recognitionライブラリとDBSCAN(コサイン類似度)を使用。
    """
    logger.info("人物ごとの画像グループ分けとフィルタリングを開始 (face_recognition, cosine, DBSCAN, largest cluster only)")
    rotated_dir = os.path.join(input_dir, "rotated")
    if not os.path.exists(rotated_dir) or not os.listdir(rotated_dir):
        logger.warning(f"rotatedディレクトリが見つからないか空です: {rotated_dir}。人物フィルタリングをスキップします。")
        return

    # デバッグ用のディレクトリを作成
    debug_dir = os.path.join(input_dir, "debug_person_classification")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)

    image_files = [f for f in os.listdir(rotated_dir) if os.path.isfile(os.path.join(rotated_dir, f))]
    encodings = []
    image_path_list = []

    logger.info("顔のエンコーディングを抽出中...")
    for filename in image_files:
        img_path = os.path.join(rotated_dir, filename)
        try:
            # OpenCVで画像を読み込み、パディングを追加して検出率を向上
            bgr_image = cv2.imread(img_path)
            if bgr_image is None:
                logger.warning(f"画像の読み込みに失敗しました: {img_path}")
                continue

            # 前処理1: パディングを追加 (HOG検出器の性能向上のため)
            padding_ratio = 0.4
            h, w = bgr_image.shape[:2]
            pad_h = int(h * padding_ratio)
            pad_w = int(w * padding_ratio)
            padded_bgr_image = cv2.copyMakeBorder(bgr_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)

            # 前処理2: ノイズリダクション
            denoised_image = cv2.fastNlMeansDenoisingColored(padded_bgr_image, None, 10, 10, 7, 21)
            logger.info(f"  -> ノイズリダクションを適用: {filename}")

            # デバッグ用: 前処理後の画像を保存
            debug_preprocessed_path = os.path.join(debug_dir, f"preprocessed_{filename}")
            cv2.imwrite(debug_preprocessed_path, denoised_image)

            # face_recognitionで使えるようにRGBに変換
            image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)

            # パディングされた画像で顔を検出し、エンコーディングを抽出
            # 高速なHOGモデルを使用。精度が必要な場合は "cnn" に変更（ただし非常に遅い）
            face_locations = face_recognition.face_locations(image_rgb, model="hog")
            if face_locations:
                logger.info(f"  -> {len(face_locations)}個の顔を検出: {filename}")
                # デバッグ用: 検出領域を描画して保存
                debug_detection_image = denoised_image.copy()
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(debug_detection_image, (left, top), (right, bottom), (0, 255, 0), 2)
                debug_detected_path = os.path.join(debug_dir, f"detected_{filename}")
                cv2.imwrite(debug_detected_path, debug_detection_image)

                face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    image_path_list.append(img_path)
            else:
                logger.warning(f"face_recognitionが顔を検出できませんでした: {img_path}")
        except Exception as e:
            logger.error(f"face_recognition処理中にエラーが発生しました {img_path}: {e}")
            continue

    if len(encodings) < 2:
        logger.info("クラスタリング対象の画像が2枚未満のため、人物フィルタリングを終了します。")
        return

    # コサイン類似度を使用。toleranceは0.4を一般的な値として使用。
    # 元スクリプトのmin_samples=1を適用
    TOLERANCE = 0.024
    labels = perform_dbscan_clustering(encodings, metric='cosine', eps=TOLERANCE, min_samples=1)

    # 最大クラスタ（主要人物）を特定
    cluster_summary = dict(zip(*np.unique(labels, return_counts=True)))
    logger.info(f"クラスタリング結果 (ラベル: 画像数): {cluster_summary}")

    # ノイズ(-1)を除いたコアなクラスタを対象にする
    core_labels = labels[labels != -1]
    if len(core_labels) == 0:
        logger.warning("主要な人物クラスタが見つかりませんでした。全ての画像がノイズと判断されたため、全ての画像を削除対象とします。")
        # 主要人物がいないため、全ての画像を削除
        images_to_delete_paths = image_path_list
    else:
        # 最も大きいクラスタを主要人物とする
        unique_core_labels, core_counts = np.unique(core_labels, return_counts=True)
        max_cluster_index = np.argmax(core_counts)
        main_cluster_label = unique_core_labels[max_cluster_index]
        main_cluster_size = core_counts[max_cluster_index]
        logger.info(f"主要な人物のクラスタラベル: {main_cluster_label} (画像数: {main_cluster_size})")

        # 削除対象の画像を特定 (主要人物クラスタ以外)
        images_to_delete_paths = []
        for i, label in enumerate(labels):
            if label != main_cluster_label:
                img_path = image_path_list[i]
                images_to_delete_paths.append(img_path)

    if not images_to_delete_paths:
        logger.info("削除対象の異人物画像はありません (全てが主要人物クラスタに属しています)。")
        return

    # 削除対象の画像の内訳をログに出力
    delete_summary = defaultdict(int)
    for i, label in enumerate(labels):
        if label != main_cluster_label:
            delete_summary[label] += 1
    logger.info(f"削除対象の画像 {len(images_to_delete_paths)}枚の内訳 (ラベル: 画像数): {dict(delete_summary)}")

    # 削除処理
    deleted_dir = os.path.join(input_dir, "deleted")
    processed_dir = os.path.join(input_dir, "processed")
    resized_dir = os.path.join(input_dir, "resized")
    bbox_cropped_dir = os.path.join(input_dir, "bbox_cropped")
    bbox_rotated_dir = os.path.join(input_dir, "bbox_rotated")

    for img_path_to_delete in images_to_delete_paths:
        base_name = os.path.splitext(os.path.basename(img_path_to_delete))[0]
        
        original_filename_with_ext = processed_face_to_original_map.get(base_name)
        if not original_filename_with_ext:
            logger.warning(f"人物フィルタリング: オリジナルファイル名が見つかりません。スキップ: {base_name}")
            continue
        _, original_ext = os.path.splitext(original_filename_with_ext)

        files_to_move = [
            img_path_to_delete,
            os.path.join(processed_dir, f"{base_name}.png"),
            os.path.join(resized_dir, f"{base_name}{original_ext}"),
            os.path.join(bbox_cropped_dir, f"{base_name}{original_ext}"),
            os.path.join(bbox_rotated_dir, f"{base_name}{original_ext}"),
            os.path.join(input_dir, original_filename_with_ext)
        ]

        for file_path in files_to_move:
            if os.path.exists(file_path):
                destination_path = os.path.join(deleted_dir, os.path.basename(file_path))
                try:
                    shutil.move(file_path, destination_path)
                    logger.info(f"人物フィルタリングにより移動: {file_path} -> {destination_path}")
                except Exception as e:
                    logger.error(f"人物フィルタリング中の移動エラー {file_path}: {e}")
            else:
                logger.warning(f"人物フィルタリング: ファイルが見つかりません（移動済みか）: {file_path}")

def cleanup_directories(input_dir):
    """
    OUTPUT_DIR内の 'deleted' フォルダを除くすべてのファイルとフォルダを削除する。
    OUTPUT_DIR内の不要な中間ファイルとフォルダを削除する。
    削除したくないフォルダは、下のリストからコメントアウトしてください。
    """
    logger.info("クリーンアップ開始")
    # 削除対象のサブディレクトリリスト
    dirs_to_delete = [
        "processed",
        # "resized", # 残したい場合はこの行をコメントアウト
        # "rotated", # 残したい場合はこの行をコメントアウト
        "bbox_cropped",
        "bbox_rotated",
    ]

    for dir_name in dirs_to_delete:
        dir_path = os.path.join(input_dir, dir_name)
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"成功的に削除されたディレクトリ: {dir_path}")
        except Exception as e:
            logger.error(f"ディレクトリ削除エラー {dir_path}: {e}")

    # input_dir直下のファイルの削除 (log.txtとdeletedフォルダを除く)
    for item_name in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item_name)
        if os.path.isfile(item_path) and item_name.lower() != 'log.txt':
            try:
                os.remove(item_path)
                logger.info(f"成功的に削除されたファイル: {item_path}")
            except Exception as e:
                logger.error(f"ファイル削除エラー {item_path}: {e}")

def process_images(keyword):
    input_dir = OUTPUT_DIR
    logger.info(f"画像処理開始：{input_dir}")
    processed_face_to_original_map = detect_and_crop_faces(input_dir)
    find_similar_images(input_dir, processed_face_to_original_map)
    # filter_by_main_person(input_dir, processed_face_to_original_map)
    filter_by_main_person_cosine(input_dir, processed_face_to_original_map)
    cleanup_directories(input_dir)
    logger.info(f"画像処理完了：{input_dir}")

def main():
    try:
        logger.info(f"処理開始 for keyword: {KEYWORD}")
        download_images(KEYWORD, MAX_NUM)
        rename_files(KEYWORD)
        consolidate_files()
        process_images(KEYWORD)
        logger.info(f"全処理完了 for keyword: {KEYWORD}")
    except Exception as e:
        logger.error(f"メインエラー: {e}")
        raise
    finally:
        cv2.destroyAllWindows()
        face_detector.close()
        face_mesh.close()

if __name__ == "__main__":
    main()