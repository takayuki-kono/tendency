# データ収集ドキュメント (Stage 1)

本ドキュメントでは、Web上からの画像収集および初期処理（顔検出・クロップ）について詳述します。

## 収集フロー
高品質な画像を大量に収集するため、現在は以下の2つのスクリプトを併用する構成になっています。

### 1. 大量・高画質収集スクリプト: `collect_images_imagedl.py`
**ツール**: `pyimagedl` ライブラリ
**役割**: 外部エンジンを活用した高画質画像のバルクダウンロード。

- **対応エンジン**: Bing, Baidu, Google (Googleは補助的)
- **特徴**:
    - 各エンジンの「Large Size」フィルタを自動適用し、高解像度画像のみを収集。
    - マルチスレッドによる高速ダウンロード。
- **実行方法**: `python collect_images_imagedl.py`
- **出力先**: `master_data/{人物名}/` に直行。

### 2. パイプライン制御スクリプト: `download_and_filter_faces.py`
**役割**: 画像の追加収集（Bing icrawler）、顔検出、クロップ、重複排除、フィルタリング。

- **処理詳細**:
    1. **追加収集**: `components/part1_setup.py` (Bing icrawler) を実行。
    2. **顔加工**: InsightFaceによる顔検出・正立補正・クロップ（眉から顎まで）。
    3. **重複排除**: Part 2a (`deleted_duplicates/` へ移動)。
    4. **人物選別**: Part 2b (`deleted_outliers/` へ移動)。

---

## 処理詳細 (コンポーネント)

### A. 顔検出 & クロップ (InsightFace)
`components/part1_setup.py` で実行されます。

1.  **回転補正 (Alignment)**: `landmark_2d_106` を使用し、顔の中心軸を垂直に補正。
2.  **領域クロップ**: 眉の上から顎の下までを切り出し、`224x224` px で保存。
3.  **パディング設定**: 現在、パディング比率による制限は無効化されています。

---

## 変更履歴
- **2025-12-28 (3)**: `pyimagedl` を使用した高画質バルク収集スクリプト `collect_images_imagedl.py` を追加。
- **2025-12-28 (2)**: パディング比率によるフィルタリングを無効化。
- **2025-12-28**: Google画像検索（Selenium版）を廃止し、Bing検索拡張へ変更。

## Author
Gemini
