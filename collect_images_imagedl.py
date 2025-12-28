import os
import sys
import shutil
import logging
import uuid
from imagedl import imagedl

# --- Configuration ---
KEYWORDS = ["女優 奈緒", "奈緒 女優", "奈緒 俳優"]
TARGET_COUNT_PER_ENGINE = 400
OUTPUT_BASE_DIR = "master_data"
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

ENGINES = [
    'YahooImageClient',
    'BingImageClient',
    'BaiduImageClient',
    'GoogleImageClient'
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "log_imagedl_collection.txt"), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def collect_images():
    unified_name = "奈緒"
    final_dir = os.path.join(OUTPUT_BASE_DIR, unified_name)
    os.makedirs(final_dir, exist_ok=True)

    # 一時保存用のルートディレクトリ
    temp_root = os.path.abspath('temp_imagedl_raw')
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)

    for engine_name in ENGINES:
        logger.info(f"--- Running engine: {engine_name} ---")
        
        for keyword in KEYWORDS:
            logger.info(f"Searching for '{keyword}' on {engine_name}")
            
            try:
                client = imagedl.ImageClient(
                    image_source=engine_name,
                    init_image_client_cfg={'work_dir': temp_root, 'max_retries': 3},
                    search_limits=TARGET_COUNT_PER_ENGINE,
                    num_threadings=10
                )
                
                filters = {'size': 'large'}
                image_infos = client.search(keyword, filters=filters)
                
                if not image_infos:
                    continue
                
                downloaded_infos = client.download(image_infos)
                
                if downloaded_infos:
                    logger.info(f"Moving {len(downloaded_infos)} files to {final_dir}")
                    for info in downloaded_infos:
                        src_path = info['file_path'] # これは .jpg 等の拡張子が付いた完全なパス
                        
                        if os.path.exists(src_path):
                            # 元の拡張子を取得
                            ext = os.path.splitext(src_path)[1]
                            # ユニークなファイル名を生成して衝突を回避
                            # エンジン名 + UUID (短縮) + 拡張子
                            new_filename = f"{engine_name}_{uuid.uuid4().hex[:8]}{ext}"
                            dest_path = os.path.join(final_dir, new_filename)
                            
                            try:
                                shutil.move(src_path, dest_path)
                            except Exception as e:
                                logger.error(f"Failed to move {src_path}: {e}")
                
                # エンジンごとのサブフォルダをクリーンアップ
                # downloaded_infos[0]['work_dir'] は temp_root 内のサブフォルダ
                if downloaded_infos:
                    engine_work_dir = downloaded_infos[0]['work_dir']
                    if os.path.exists(engine_work_dir) and engine_work_dir != temp_root:
                        shutil.rmtree(engine_work_dir)

            except Exception as e:
                logger.error(f"Error with engine {engine_name} for keyword {keyword}: {e}")

    # 最後に残った一時ファイルを削除
    if os.path.exists(temp_root):
        try:
            shutil.rmtree(temp_root)
        except: pass

    logger.info(f"=== Collection Complete. Unified files in {final_dir} ===")

if __name__ == "__main__":
    collect_images()
