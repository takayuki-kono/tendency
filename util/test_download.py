from icrawler.builtin import GoogleImageCrawler
import os
import shutil

test_dir = "test_download_dir"
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

os.makedirs(test_dir, exist_ok=True)

print("Testing icrawler download...")
crawler = GoogleImageCrawler(storage={'root_dir': test_dir})

try:
    crawler.crawl(keyword="広瀬すず", max_num=5)
    print(f"\nDownload complete. Files in {test_dir}:")
    files = os.listdir(test_dir)
    print(f"Downloaded {len(files)} files")
    for f in files[:5]:
        print(f"  - {f}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
