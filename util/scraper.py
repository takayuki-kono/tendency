import os
import time
import json
import requests
import hashlib
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright

# --- Configuration ---
BASE_KEYWORD = "奈緒"
OUTPUT_DIR = "scraper_output"
TARGET_COUNT = 1000
CONCURRENCY = 20  # ダウンロード並列数（Bingの直リンクなら速いので増やせます）

# 高画質のみを狙うためのフィルタ
# filterui:imagesize-large -> 大きい画像
# filterui:face-face -> 顔（任意だが、今回は人物なので有効かも。一旦外して量重視）
BING_FILTERS = "+filterui:imagesize-large"

# 検索ワード拡張
SEARCH_MODIFIERS = [
    "", " 高画質", " インタビュー", " ドラマ", " 映画", 
    " CM", " 衣装", " 髪型", " 雑誌", " イベント", 
    " 笑顔", " 横顔", " 真顔", " 写真集", " 制作発表",
    " actress", " portrait", " close up", " japanese",
    " 2024", " 2023", " 2022", " 昔", " デビュー"
]

def download_image(url, save_dir, headers):
    try:
        # タイムアウトを短めに設定して回転率を上げる
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            # コンテンツタイプチェック
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                return False

            # URLハッシュでファイル名生成
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
            
            # 拡張子推定
            ext = ".jpg"
            if "png" in content_type: ext = ".png"
            elif "webp" in content_type: ext = ".webp"
            elif "jpeg" in content_type: ext = ".jpg"
            
            save_path = os.path.join(save_dir, f"{url_hash}{ext}")
            
            if os.path.exists(save_path):
                return False # 重複スキップ

            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        return False
    return False

def scrape_bing_high_res():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    collected_urls = set()
    total_downloaded = 0

    with sync_playwright() as p:
        print("Launching Browser...")
        browser = p.chromium.launch(headless=False) # 動作確認のため画面表示
        page = browser.new_page()

        for modifier in SEARCH_MODIFIERS:
            if total_downloaded >= TARGET_COUNT:
                break
            
            query = f"{BASE_KEYWORD} {modifier}".strip()
            print(f"\n--- Searching: {query} (Total: {total_downloaded}) ---")
            
            # Bing画像検索URL (Large size filter applied)
            url = f"https://www.bing.com/images/search?q={query}&qft={BING_FILTERS}&form=HDRSC2"
            
            try:
                page.goto(url)
                time.sleep(2)

                # 無限スクロールでURLを収集
                prev_count = 0
                max_scroll_attempts = 10 # 1キーワードあたりのスクロール回数制限
                
                for _ in range(max_scroll_attempts):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(2)
                    
                    # 'm' 属性を持つリンクを探す (Bingの仕様: m="{murl:..., ...}")
                    # a.iusc が画像リンクのクラス
                    elements = page.query_selector_all("a.iusc")
                    current_count = len(elements)
                    
                    if current_count == prev_count:
                        # 「もっと見る」ボタンがあるか確認
                        try:
                            btn = page.query_selector(".btn_seemore")
                            if btn: 
                                btn.click()
                                time.sleep(2)
                            else:
                                break # これ以上読み込めない
                        except:
                            break
                    
                    prev_count = current_count
                    print(f"  Loaded {current_count} thumbnails...")
                    if current_count > 300: # 1キーワードあたり300枚くらいで次へ
                        break

                # URL抽出
                elements = page.query_selector_all("a.iusc")
                page_urls = []
                
                print(f"  Extracting high-res URLs from {len(elements)} elements...")
                for el in elements:
                    try:
                        m_attr = el.get_attribute("m")
                        if m_attr:
                            # JSONパースして murl (Media URL) を取得
                            m_json = json.loads(m_attr)
                            murl = m_json.get("murl")
                            if murl and murl.startswith("http") and murl not in collected_urls:
                                collected_urls.add(murl)
                                page_urls.append(murl)
                    except:
                        continue
                
                print(f"  Found {len(page_urls)} unique high-res URLs. Downloading...")

                # 並列ダウンロード実行
                with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
                    }
                    futures = [
                        executor.submit(download_image, u, OUTPUT_DIR, headers) 
                        for u in page_urls
                    ]
                    
                    for future in futures:
                        if future.result():
                            total_downloaded += 1
                            if total_downloaded >= TARGET_COUNT:
                                break
                                
            except Exception as e:
                print(f"Error searching {query}: {e}")

        browser.close()

    print(f"\nCompleted! Total downloaded: {total_downloaded}")

if __name__ == "__main__":
    scrape_bing_high_res()