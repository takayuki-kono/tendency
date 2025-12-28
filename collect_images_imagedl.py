import os
import sys
import shutil
import logging
from imagedl import imagedl

# --- Configuration ---
KEYWORDS = ["奈緒"]
TARGET_COUNT_PER_ENGINE = 500  # 各エンジンあたりの目標枚数
OUTPUT_BASE_DIR = "master_data"
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 使用するエンジン
# Bing: 安定・高画質, Baidu: 大量, Google: 網羅的
ENGINES = [
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
    for keyword in KEYWORDS:
        logger.info(f"=== Starting collection for: {keyword} ===")
        
        # 最終的な保存先
        final_dir = os.path.join(OUTPUT_BASE_DIR, keyword)
        os.makedirs(final_dir, exist_ok=True)
        
        for engine_name in ENGINES:
            logger.info(f"Running engine: {engine_name}")
            
            try:
                # imagedlのクライアント初期化
                # work_dirを指定して、エンジンごとの一時フォルダに保存させる
                client = imagedl.ImageClient(
                    image_source=engine_name,
                    init_image_client_cfg={'work_dir': 'temp_imagedl_raw', 'max_retries': 3},
                    search_limits=TARGET_COUNT_PER_ENGINE,
                    num_threadings=10
                )
                
                # 検索実行（高画質フィルタを適用）
                # size='large' は Bing, Baidu, Google で共通のフィルタオプション
                filters = {'size': 'large'}
                image_infos = client.search(keyword, filters=filters)
                
                if not image_infos:
                    logger.warning(f"No images found by {engine_name}")
                    continue
                
                # ダウンロード実行
                downloaded_infos = client.download(image_infos)
                logger.info(f"Downloaded {len(downloaded_infos)} images from {engine_name}")
                
                # ダウンロードされたファイルを master_data に移動
                if downloaded_infos:
                    for info in downloaded_infos:
                        src_path = info['file_path']
                        if os.path.exists(src_path):
                            filename = os.path.basename(src_path)
                            # ファイル名衝突回避のためエンジン名を付与
                            dest_filename = f"{engine_name}_{filename}"
                            dest_path = os.path.join(final_dir, dest_filename)
                            
                            shutil.move(src_path, dest_path)
                
                # 一時フォルダのクリーンアップ（エンジンごとの結果フォルダ）
                engine_output_dir = downloaded_infos[0]['work_dir']
                if os.path.exists(engine_output_dir):
                    shutil.rmtree(engine_output_dir)

            except Exception as e:
                logger.error(f"Error with engine {engine_name}: {e}")

        logger.info(f"=== Finished collection for: {keyword}. Files are in {final_dir} ===")

    # 全体の一時フォルダを削除
    if os.path.exists('temp_imagedl_raw'):
        shutil.rmtree('temp_imagedl_raw')

if __name__ == "__main__":
    collect_images()
