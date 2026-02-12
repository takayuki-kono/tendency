# 機械学習パイプライン仕様書

## 0. 環境構築と準備
本パイプラインを実行する前に、以下のセットアップが必要です。

### 必須ライブラリのインストール
主要な収集・加工エンジンを動作させるため、以下のコマンドを実行してください。
```bash
# 画像収集エンジン
pip install pyimagedl

# 顔検出・解析エンジン
pip install insightface onnxruntime-gpu

# その他依存関係
pip install beautifulsoup4 lxml json_repair pyfreeproxy alive_progress pathvalidate
```

### 外部リポジトリ・モデル
- **pyimagedl**: `pip install pyimagedl` でインストールされますが、最新版の仕様確認が必要な場合は [CharlesPikachu/imagedl](https://github.com/CharlesPikachu/imagedl) を参照してください。
- **InsightFace**: 初回実行時に自動的にモデルがダウンロードされます。

---

## 1. 概要
本パイプラインは、生画像から高品質な顔データセットを生成し、フィルタリングパラメータを最適化し、分類モデルを学習するために設計されています。

## 2. パイプライン構成

### ステージ 1: データ収集と初期加工
**スクリプト:** `download_and_filter_faces.py`
**入力:** 生画像（キーワードに基づいてダウンロード）
**処理内容:**
1.  **ダウンロード:** 指定キーワードの画像をマルチエンジンで収集。
    -   **エンジン:** Yahoo, Bing, Baidu, Google (pyimagedl使用)
    -   **キーワード:** 「〇〇 女優」に固定。
    -   **枚数制限:** 各エンジン最大1000枚。
    -   **ドメインフィルタリング:** Malwarebytes等で警告される不正ドメイン（clubberia, xxupなど）からのダウンロードをブロック。
2.  **Part 1 (`components/part1_setup.py`):**
    -   **顔検出:** InsightFaceを使用して顔を検出。
    -   **回転補正:** 顔の向きを正立に補正。
    -   **クロップ:** **眉毛から顎まで**（インデックス: 49/104 ～ 顎）で切り抜き。
    -   **パディング:** 制限なし。
    -   **リサイズ:** `224x224` px に統一。
    -   **I/O:** `imread_safe` を使用し、日本語ファイル名に対応。
3.  **Part 2a (`components/part2a_similarity.py`):**
    -   **重複排除:** 特徴量ベクトルを計算し、DBSCANを用いて「ほぼ同じ画像」を削除。
    -   **削除先:** `deleted_duplicates/`（論理削除モード時）
4.  **Part 2b (`components/part2b_filter.py`):**
    -   **顔位置フィルター:** 内側目ランドマーク(89,39)を使用し、横向きの顔を検出・除外。
    -   **3D頬幅フィルター:** 3Dランドマーク(3,13)の3Dユークリッド距離を使用し、頬幅比率0.7未満の横顔を除外。
    -   **アスペクト比フィルター:** 0.9〜1.8の範囲外を除外。
    -   **外れ値除去:** DBSCANクラスタリング(eps=0.35)を行い、主となる人物以外を除去。
    -   **削除先:** `deleted_outliers/`（論理削除モード時）
    -   **整理:** 処理後、`rotated/` から親ディレクトリへ画像を移動。

**出力:** `master_data/{label}/...`（クリーニング済みの224x224顔画像）

---

### ステージ 2: 前処理と高度なフィルタリング
**スクリプト:** `preprocess_multitask.py`
... (以下、変更なし) ...

### ユーティリティ
#### データ集計
**スクリプト:** `util/count_data.py`
**処理内容:**
1.  **集計対象:** `master_data`, `train`, `validation`, `preprocessed_multitask` 内の画像数をカウント。
2.  **再帰探索:** 各ディレクトリ内を再帰的に探索し、ラベル（人物名の入れ子含む）ごとの集計を行う。
3.  **分布表示:** タスク A, B, C, D ごとのクラス分布を統計的に表示。

### ステージ 3: モデル学習 (Sequential Training)
**スクリプト:** `train_sequential.py` (および `components/train_multitask_trial.py`)
**最適化手法:**
- 探索空間（Learning Rate, Layers, Dropout, Augmentation等）を順次固定しながら最適化するSequential Optimizationを採用。
- **評価指標の変更 (2026-02-11):**
    - 従来の `Balanced Accuracy` (平均再現率) に代わり、**`MinClassAccuracy` (最低クラス精度)** を採用。
    - 各クラスの再現率（Recall）のうち、**最も低い値**を最大化するように学習・最適化を行う。
    - これにより、難易度の高いクラスや少数派クラスの放置を防ぎ、全てのクラスで一定以上の精度を保証することを目指す。
- **Fine-tuning:**
    - 最適化されたパラメータを用いて、最終的に全層解凍によるFine-tuningを実施。
    - **詳細ログ出力 (2026-02-11):**
        - 検証データに対する全クラスの個別精度（正解数/総数）を出力し、ボトルネックとなっているクラスを特定可能にした。

---

## Author
Gemini (Updated 2026-02-11)

