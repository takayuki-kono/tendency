import os
import sys
import shutil
import logging
from imagedl import imagedl

# --- Configuration ---
# キーワードをより具体的に。バリエーションを持たせることで重複を避けつつ枚数を稼ぐ
KEYWORDS = ["女優 奈緒", "奈緒 女優", "奈緒 俳優"]
TARGET_COUNT_PER_ENGINE = 400  # 1エンジンあたり
OUTPUT_BASE_DIR = "master_data"
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 使用するエンジン
# Yahoo: 日本のコンテンツに最強
# Bing: 安定・高画質
# Baidu: 中国経由のレア画像・高画質写真が多い
# Google: 網羅性
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
    # 最終的な保存先（奈緒として統一）
    unified_name = "奈緒"
    final_dir = os.path.join(OUTPUT_BASE_DIR, unified_name)
    os.makedirs(final_dir, exist_ok=True)

    collected_urls = set()

    for engine_name in ENGINES:
        logger.info(f"--- Running engine: {engine_name} ---")
        
        for keyword in KEYWORDS:
            logger.info(f"Searching for '{keyword}' on {engine_name}")
            
            try:
                client = imagedl.ImageClient(
                    image_source=engine_name,
                    init_image_client_cfg={'work_dir': 'temp_imagedl_raw', 'max_retries': 3},
                    search_limits=TARGET_COUNT_PER_ENGINE,
                    num_threadings=10
                )
                
                # 高画質フィルタ
                filters = {'size': 'large'}
                image_infos = client.search(keyword, filters=filters)
                
                if not image_infos:
                    continue
                
                # ダウンロード
                downloaded_infos = client.download(image_infos)
                
                if downloaded_infos:
                    for info in downloaded_infos:
                        src_path = info['file_path']
                        if os.path.exists(src_path):
                            filename = os.path.basename(src_path)
                            # 重複・衝突回避
                            dest_filename = f"{engine_name}_{filename}"
                            dest_path = os.path.join(final_dir, dest_filename)
                            
                            if not os.path.exists(dest_path):
                                shutil.move(src_path, dest_path)
                            else:
                                os.remove(src_path) # 重複は削除
                
                # 一時フォルダのクリーンアップ
                engine_output_dir = downloaded_infos[0]['work_dir']
                if os.path.exists(engine_output_dir):
                    shutil.rmtree(engine_output_dir)

            except Exception as e:
                logger.error(f"Error with engine {engine_name} for keyword {keyword}: {e}")

    # 全体の一時フォルダを削除
    if os.path.exists('temp_imagedl_raw'):
        shutil.rmtree('temp_imagedl_raw')

    logger.info(f"=== Collection Complete. Unified files in {final_dir} ===")

if __name__ == "__main__":
    collect_images()