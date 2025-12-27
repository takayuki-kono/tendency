# Google画像検索スクレイピング機能の追加

## 変更概要
`components/part1_setup.py` に、Google画像検索結果を高速に収集する機能を追加しました。
従来の `BingImageCrawler` (icrawler) に加えて、`requests` と `BeautifulSoup` を使用した独自のスクレーパーを実装しました。

## 実装詳細
### 1. 新規関数: `scrape_google_images`
- **ライブラリ**: `requests`, `BeautifulSoup`, `re`
- **ターゲット**: Google画像検索 (`tbm=isch`)
- **動作**:
  1. 指定キーワードで検索リクエストを送信
  2. HTML内の `<img>` タグから画像URL (`src` または `data-src`) を抽出
  3. **品質チェック**: 以下の条件を満たす画像は低品質としてスキップ
     - ファイルサイズ < 5KB
     - 解像度 (幅 or 高さ) < 150px
  4. ロゴ画像などを除外し、サムネイル/画像を直接ダウンロード
  5. ファイル名: `google_{count:04d}.jpg`

### 2. 変更点: `download_images`
- 既存の Bing 収集プロセスの後に、新設の `scrape_google_images` を呼び出すように変更。
- 同一の保存ディレクトリを使用するが、ファイル名プレフィックスが異なるため衝突しない。
- 後の `consolidate_files` プロセスで自動的に統合・リネームされる。

## 目的
データ収集量の増加と、ソースの多角化によるデータセットの質の向上。
icrawlerのGoogle全収集機能が動作不安定なため、軽量かつ高速な「初期結果収集」に特化させた。

## 影響範囲
- `download_and_filter_faces.py` (間接的影響: 収集枚数が増える)
- `components/part1_setup.py`

## Author
Gemini (2025-12-27)
