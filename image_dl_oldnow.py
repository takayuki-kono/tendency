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

KEYWORD = "安藤サクラ"
MAX_NUM = 10
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
                scale_x = IMG_SIZE / (x_max_crop - x_min_crop)
                scale_y = IMG_SIZE / (y_max_crop - y_min_crop)
                for point in ['left_eyebrow', 'right_eyebrow', 'chin', 'nose']:
                    x = int((landmarks[point][0] - x_min_crop) * scale_x)
                    y = int((landmarks[point][1] - y_min_crop) * scale_y)
                    if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
                        cv2.circle(face_image_resized, (x, y), 3, (0, 0, 255), -1)
                    else:
                        logger.warning(f"Invalid landmark position for {point} in {filename} (face_idx: {face_idx}): ({x}, {y})")

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

                r_scale_x = IMG_SIZE / (rx_max_crop - rx_min_crop)
                r_scale_y = IMG_SIZE / (ry_max_crop - ry_min_crop)
                for point in ['left_eyebrow', 'right_eyebrow', 'chin', 'nose']:
                    rx = int((rot_landmarks[point][0] - rx_min_crop) * r_scale_x)
                    ry = int((rot_landmarks[point][1] - ry_min_crop) * r_scale_y)
                    if 0 <= rx < IMG_SIZE and 0 <= ry < IMG_SIZE:
                        cv2.circle(rotated_face_resized, (rx, ry), 3, (0, 255, 0), 1)
                    else:
                        logger.warning(f"Invalid landmark position for {point} in {filename} (face_idx: {face_idx}): ({rx}, {ry})")

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

def cleanup_directories(input_dir):
    # find_similar_imagesで削除対象のファイルは全てdeleted_dirに移動済み
    # ここでは一時ディレクトリを削除するのみ
    # input_dir直下のファイルは、類似画像処理で「保持」されたオリジナル画像なので削除しない
    logger.info("クリーンアップ開始")
    temp_dirs_to_delete = [
        os.path.join(input_dir, "processed"),
        os.path.join(input_dir, "bbox_cropped"),
        os.path.join(input_dir, "bbox_rotated")
    ]
    
    for d in temp_dirs_to_delete:
        try:
            if os.path.exists(d):
                shutil.rmtree(d)
                logger.info(f"成功的に削除された: {d}")
            else:
                logger.warning(f"ディレクトリが存在しません: {d}")
        except Exception as e:
            logger.error(f"クリーンアップエラー {d}: {e}")

def process_images(keyword):
    input_dir = OUTPUT_DIR
    logger.info(f"画像処理開始：{input_dir}")
    processed_face_to_original_map = detect_and_crop_faces(input_dir)
    find_similar_images(input_dir, processed_face_to_original_map)
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