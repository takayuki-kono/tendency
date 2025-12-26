from icrawler.builtin import BingImageCrawler
import os
import shutil

test_dir = "test_bing_100"
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

os.makedirs(test_dir, exist_ok=True)

print("Testing Bing with MAX_NUM=100...")
crawler = BingImageCrawler(storage={'root_dir': test_dir})

try:
    crawler.crawl(keyword="広瀬すず", max_num=100)
    files = os.listdir(test_dir)
    print(f"\nResult: Downloaded {len(files)} files out of 100 requested")

    # 検索バリエーションもテスト
    for variant in ["広瀬すず 正面", "広瀬すず 顔"]:
        var_dir = f"test_bing_{variant.replace(' ', '_')}"
        if os.path.exists(var_dir):
            shutil.rmtree(var_dir)
        os.makedirs(var_dir, exist_ok=True)

        print(f"\nTesting: {variant}")
        crawler2 = BingImageCrawler(storage={'root_dir': var_dir})
        crawler2.crawl(keyword=variant, max_num=100)
        files2 = os.listdir(var_dir)
        print(f"  Downloaded {len(files2)} files")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
