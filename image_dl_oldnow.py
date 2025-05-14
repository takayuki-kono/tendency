import os
import shutil
import logging
from icrawler.builtin import GoogleImageCrawler

# 検索キーワードと最大ダウンロード数を定義
KEYWORD = "賀来千香子"
MAX_NUM = 100

# ログ設定
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

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
        (keyword, keyword),  # キーワード
        (f"{keyword} 過去", f"{keyword}_過去"),  # キーワード 過去
        (f"{keyword} 現在", f"{keyword}_現在")  # キーワード 現在
    ]
    
    for search_keyword, storage_dir in search_terms:
        # フォルダが存在する場合、削除して再作成
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
        os.makedirs(storage_dir, exist_ok=True)
        
        logging.info(f"Starting download for keyword: {search_keyword}, storage: {storage_dir}")
        print(f"Downloading images for: {search_keyword}")
        
        crawler = setup_crawler(storage_dir)
        crawler.crawl(keyword=search_keyword, max_num=max_num)
        
        # ダウンロードされた画像数を確認
        downloaded_files = [f for f in os.listdir(storage_dir) if os.path.isfile(os.path.join(storage_dir, f))]
        logging.info(f"Downloaded {len(downloaded_files)} images for {search_keyword}")
        print(f"Downloaded {len(downloaded_files)} images for {search_keyword}")

def rename_files(keyword):
    """各フォルダ内のファイル名を親フォルダ名に基づいてリネーム"""
    folders = [
        keyword,  # キーワード
        f"{keyword}_過去",  # キーワード 過去
        f"{keyword}_現在"   # キーワード 現在
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            logging.warning(f"Folder {folder} does not exist, skipping rename")
            print(f"Folder {folder} does not exist, skipping rename")
            continue
        
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            old_path = os.path.join(folder, file)
            # 新しいファイル名：親フォルダ名_元のファイル名
            new_filename = f"{folder}_{file}"
            new_path = os.path.join(folder, new_filename)
            
            try:
                os.rename(old_path, new_path)
                logging.info(f"Renamed {old_path} to {new_path}")
                print(f"Renamed {old_path} to {new_path}")
            except Exception as e:
                logging.error(f"Error renaming {old_path} to {new_path}: {e}")
                print(f"Error renaming {old_path} to {new_path}: {e}")

def consolidate_files(keyword, output_dir=None):
    """全フォルダのファイルを1つのフォルダに統合"""
    if output_dir is None:
        output_dir = f"{keyword}_all"
    
    # 統合フォルダを作成（存在する場合は削除して再作成）
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    folders = [
        keyword,  # キーワード
        f"{keyword}_過去",  # キーワード 過去
        f"{keyword}_現在"   # キーワード 現在
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            logging.warning(f"Folder {folder} does not exist, skipping consolidation")
            print(f"Folder {folder} does not exist, skipping consolidation")
            continue
        
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            src_path = os.path.join(folder, file)
            dst_path = os.path.join(output_dir, file)
            
            try:
                shutil.move(src_path, dst_path)
                logging.info(f"Moved {src_path} to {dst_path}")
                print(f"Moved {src_path} to {dst_path}")
            except Exception as e:
                logging.error(f"Error moving {src_path} to {dst_path}: {e}")
                print(f"Error moving {src_path} to {dst_path}: {e}")
    
    # 空のフォルダを削除
    for folder in folders:
        if os.path.exists(folder) and not os.listdir(folder):
            shutil.rmtree(folder)
            logging.info(f"Removed empty folder {folder}")
            print(f"Removed empty folder {folder}")

# 処理の実行
logging.info(f"Starting image download for keyword: {KEYWORD}")
print(f"Starting image download for keyword: {KEYWORD}")

# 1. 3つのキーワードで画像ダウンロード
download_images(KEYWORD, MAX_NUM)

# 2. ファイル名をリネーム
rename_files(KEYWORD)

# 3. ファイルを統合フォルダに移動
consolidate_files(KEYWORD)

logging.info(f"Completed processing for keyword: {KEYWORD}")
print(f"Completed processing for keyword: {KEYWORD}")