# [廃止] Google画像検索スクレイピング機能（Selenium版）

**注意: 本機能は 2025-12-28 に廃止されました。**
理由: Google側の仕様変更によるスクレイピングの不安定化（セレクタの頻繁な変更、無限スクロールの動作不良など）のため。
代替策: Bing画像検索（icrawler）の検索キーワードを拡張し、同等の収集量を確保する方針に変更されました。

---

## (旧) 変更概要
`components/part1_setup.py` に、SeleniumとHeadless Chromeを使用した高度なGoogle画像検索クローラーを実装しました。
初期の `requests` 版では取得数に限りがあったため、無限スクロールに対応したSelenium版に移行し、より多くの画像を収集可能にしました。また、低品質画像のフィルタリング機能も備えています。

## 実装詳細 (削除済み)
### 1. 新規関数: `scrape_google_images` (Selenium版)
- **ライブラリ**: `selenium`, `webdriver-manager`, `Pillow` (PIL)
- **ターゲット**: Google画像検索 (`tbm=isch`)
- **動作**:
  1. Headless ChromeブラウザでGoogle画像検索結果ページを開く。
  2. **無限スクロール**: ページ最下部までスクロールし、「もっと見る」ボタンがあればクリックして画像をロードし続ける。
  3. **画像抽出**: 収集可能なすべての画像要素 (`img`) を特定。
  4. **品質チェック**: 以下の条件に該当する画像はスキップ（フィルタリング）。
     - ファイルサイズ < 5KB
     - 解像度 (幅 or 高さ) < 100px
     - ロゴやファビコンなどの不要画像パターン
  5. ファイル名: `google_{count:04d}.jpg`

### 2. インテグレーション: `download_images`
- Bing収集プロセスと並行して実行され、同一の保存ディレクトリに格納します。
- `consolidate_files` プロセスでBing画像とともに統合・リネームされます。

## 目的
データ収集量の最大化と品質の底上げ。
`requests` では初期ロード分しか取得できなかった問題を、ブラウザ自動操作（Selenium）により解決し、1キーワードあたり100枚以上の収集を目指す。

## 必要な環境
- Google Chrome ブラウザがインストールされていること。
- `pip install selenium webdriver-manager pillow`

## 影響範囲
- `download_and_filter_faces.py` (実行時間が増加するが、収集枚数が大幅に増える)
- `components/part1_setup.py`

## Author
Gemini (2025-12-28 Updated)