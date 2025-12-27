# データ収集・前処理ドキュメント (Stage 1 & 2)

本ドキュメントでは、画像の収集から学習用データセットの生成までのプロセス（Stage 1〜2）について詳述します。

## 概要フロー
1. **収集**: `download_and_filter_faces.py` (Bing/Google)
2. **整理**: `reorganize_by_label.py` (ラベル付け)
3. **分割**: `create_person_split.py` (Train/Val/Test)
4. **前処理**: `preprocess_multitask.py` (詳細フィルタリング)

---

## 1. 画像収集 & 初期フィルタリング (Stage 1)
**スクリプト**: `download_and_filter_faces.py`

### 処理詳細
このスクリプトは、以下のコンポーネントを順次呼び出して実行します。

#### A. ダウンロード & クロップ (`components/part1_setup.py`)
- **ソース**:
    1. **Bing Image Search** (`icrawler`): メイン収集元。
    2. **Google Image Search** (`requests` + `BeautifulSoup`): 高速・軽量な追加収集（初期結果のみ）。
- **共通処理**:
    - **顔検出**: InsightFace を使用。
    - **回転補正**: ランドマークに基づき正立させる。
    - **クロップ**: 顔領域（眉〜顎）を切り出し、224x224px にリサイズ。
    - **保存先**: `master_data/{人物名}/rotated/`

#### B. 類似画像除去 (`components/part2a_similarity.py`)
- **手法**: InsightFaceのFace Embeddingを抽出し、DBSCANクラスタリングで重複（距離が0に近いもの）を検出。
- **動作**: 重複画像のうち1枚を残して他を削除（または退避）。

#### C. 外れ値除去 (`components/part2b_filter.py`)
- **手法**: Embedding空間でのクラスタリングを行い、最大クラスタ（本人である可能性が最も高い集団）のみを残す。
- **目的**: 検索ノイズ（同姓同名の別人や、誤って検出された背景の顔など）を除去。

---

## 2. データセット構成 (Stage 2a)

### ラベル整理 (`reorganize_by_label.py`)
- 人物ごとのフォルダを、ターゲットラベル（例: `TaskA_Label`）ごとのフォルダ構造に再配置します。
- `master_data/{ラベル}/{人物名}/`

### データ分割 (`create_person_split.py`)
- **方針**: リークを防ぐため、**「人物単位」**で分割します。同じ人物の画像がTrainとTestに混ざることはありません。
- **ルール**:
    - クラス内に3人以上 → Train, Val, Test に分散
    - クラス内に2人 → Train, Val
    - クラス内に1人 → Trainのみ（警告対象）

---

## 3. 詳細前処理 & データセット生成 (Stage 2b)
**スクリプト**: `preprocess_multitask.py`

### 目的
学習に適した「綺麗な顔画像」のみを選別し、最終的なデータセット `preprocessed_multitask/` を生成します。

### フィルタリング項目
以下の指標に基づき、閾値（パーセンタイル等）でフィルタリングします。

1. **Sharpness (鮮明度)**:
   - Laplacian Variance を使用して画像のブレ・ボケを検知。
   - デフォルト: 下位5% (`--sharpness_percentile_low 5`) を除外。
   
2. **Pitch (顔の上下向き)**:
   - 正面を向いていない画像を除外。

3. **Symmetry (左右対称性)**:
   - 極端な横顔や照明ムラを除外。

4. **Eyebrow-Eye Distance**:
   - 表情の崩れ（目瞑りや驚き顔の一部）を検知。

### 出力
- `preprocessed_multitask/train/`
- `preprocessed_multitask/validation/`
- `preprocessed_multitask/test/`
- `preprocessed_multitask/stats.json` (統計情報)

## 関連ドキュメント
- `docs/add_google_scraping.md`: Google収集機能の詳細仕様
