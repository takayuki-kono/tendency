import os
import cv2
import mediapipe as mp
import random
import logging
import shutil
from icrawler.builtin import GoogleImageCrawler

KEYWORD = "安藤サクラ"
MAX_NUM = 100
OUTPUT_DIR = str(random.randint(0, 1000)).zfill(4)
SIMILARITY_THRESHOLD = 2000000
IMG_SIZE = 224

logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)

mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()

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
    if os.path.exists(resized_dir):
        shutil.rmtree(resized_dir)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(resized_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    skip_counters = {'no_face': 0, 'deleted_no_face': 0}
    total_images = 0
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)
        if not results.detections:
            skip_counters['no_face'] += 1
            logger.info(f"顔検出失敗 {filename}")
            try:
                os.remove(img_path)
                skip_counters['deleted_no_face'] += 1
                logger.info(f"削除：{img_path} (no_face)")
            except Exception as e:
                logger.error(f"削除エラー {img_path} (no_face): {e}")
            continue
        h, w = img.shape[:2]
        for face_idx, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            x, y = max(0, min(x, w-1)), max(0, min(y, h-1))
            width, height = min(width, w-x), min(height, h-y)
            face_image = img[y:y+height, x:x+width]
            if face_image is None or face_image.size == 0:
                skip_counters['no_face'] += 1
                logger.info(f"顔領域切り取り失敗 {filename}")
                continue
            face_image_resized = cv2.resize(face_image, (IMG_SIZE, IMG_SIZE))
            base_name, ext = os.path.splitext(filename)
            resized_filename = f"{base_name}_face{face_idx+1}{ext}"
            resized_path = os.path.join(resized_dir, resized_filename)
            cv2.imwrite(resized_path, face_image_resized)
            logger.info(f"透過画像保存：{resized_path}")
            gray = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2GRAY)
            if gray.shape != (IMG_SIZE, IMG_SIZE):
                logger.error(f"無効な画像サイズ: {resized_filename}")
                continue
            processed_filename = f"{base_name}_face{face_idx+1}.png"
            processed_path = os.path.join(processed_dir, processed_filename)
            cv2.imwrite(processed_path, gray)
            logger.info(f"グレースケール画像保存：{processed_path}")
    for reason, count in skip_counters.items():
        rate = count / total_images * 100 if total_images > 0 else 0
        logger.info(f"{input_dir}: {reason} {count}/{total_images} ({rate:.1f}%)")

def find_similar_images(input_dir):
    processed_dir = os.path.join(input_dir, "processed")
    resized_dir = os.path.join(input_dir, "resized")
    logger.info(f"{processed_dir} の類似画像検索開始")
    image_files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if os.path.isfile(os.path.join(processed_dir, f))]
    logger.info(f"{processed_dir} で {len(image_files)} 画像を検出")
    
    def compare_images(img_path1, img_path2):
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None or img1.shape != img2.shape:
            logger.error(f"比較失敗: {img_path1} vs {img_path2}")
            return float('inf')
        diff = cv2.absdiff(img1, img2).sum()
        logger.info(f"差分計算: {img_path1} vs {img_path2}, diff={diff}")
        return diff

    groups = []
    used_images = set()
    for i, img1_path in enumerate(image_files):
        if img1_path in used_images:
            continue
        current_group = [img1_path]
        for j, img2_path in enumerate(image_files[i+1:], i+1):
            if img2_path in used_images:
                continue
            diff = compare_images(img1_path, img2_path)
            if diff <= SIMILARITY_THRESHOLD:
                current_group.append(img2_path)
                logger.info(f"類似画像検出: {img1_path} と {img2_path}")
        if len(current_group) > 1:
            groups.append(current_group)
            used_images.update(current_group)
    
    logger.info(f"{len(groups)} グループを検出")
    for group_idx, group in enumerate(groups, 1):
        logger.info(f"グループ {group_idx} 表示開始")
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
        display_image = cv2.hconcat(resized_images)
        window_name = f"Group {group_idx}"
        try:
            cv2.imshow(window_name, display_image)
            logger.info(f"グループ {group_idx} を表示: {window_name}")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(window_name)
            if key == 27:
                logger.info("ユーザーにより表示中断（Escキー）")
                break
        except Exception as e:
            logger.error(f"グループ {group_idx} 表示エラー: {e}")
    cv2.destroyAllWindows()
    logger.info("類似画像グループの画面表示完了")

    logger.info("類似画像削除処理開始")
    for group_idx, group in enumerate(groups, 1):
        if len(group) < 2:
            continue
        keep_img_path = group[0]
        logger.info(f"グループ {group_idx}: 保持: {keep_img_path}")
        for img_path in group[1:]:
            logger.info(f"削除: {img_path} (processed)")
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    logger.info(f"成功的に削除: {img_path} (processed)")
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                for ext in ['.jpg', '.jpeg', '.png']:
                    resized_path = os.path.join(resized_dir, f"{base_name}{ext}")
                    if os.path.exists(resized_path):
                        os.remove(resized_path)
                        logger.info(f"成功的に削除: {resized_path} (resized)")
            except Exception as e:
                logger.error(f"削除エラー {img_path}: {e}")

def cleanup_directories(input_dir):
    logger.info("クリーンアップ開始")
    resized_dir = os.path.join(input_dir, "resized")
    try:
        if os.path.exists(resized_dir):
            shutil.rmtree(resized_dir)
            logger.info(f"成功的に削除: {resized_dir} (resizedディレクトリ)")
        else:
            logger.warning(f"ディレクトリが存在しません: {resized_dir}")
        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info(f"成功的に削除: {file_path} (元画像)")
    except Exception as e:
        logger.error(f"クリーンアップエラー: {e}")

def process_images(keyword):
    input_dir = OUTPUT_DIR
    logger.info(f"画像処理開始：{input_dir}")
    detect_and_crop_faces(input_dir)
    find_similar_images(input_dir)
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
        face_detection.close()
        selfie_segmentation.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()