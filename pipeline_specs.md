# 機械学習パイプライン仕様書

## 1. 概要
本パイプラインは、生画像から高品質な顔データセットを生成し、フィルタリングパラメータを最適化し、分類モデルを学習するために設計されています。

## 2. パイプライン構成

### ステージ 1: データ収集と初期加工
**スクリプト:** `download_and_filter_faces.py`
**入力:** 生画像（キーワードに基づいてダウンロード）
**処理内容:**
1.  **ダウンロード:** 指定キーワード（例: "奈緒"）の画像を収集。
    -   **Bing Image Search:** `icrawler` を使用。
    -   **キーワード拡張:** "正面", "笑顔", "ドラマ", 英語表記など多角的に収集し量を確保。
    -   ※ Google画像検索機能は廃止されました。
2.  **Part 1 (`components/part1_setup.py`):**
    -   **顔検出:** InsightFaceを使用して顔を検出。
    -   **回転補正:** 顔の向きを正立に補正。
    -   **クロップ:** **眉毛から顎まで**（インデックス: 49/104 ～ 顎）で切り抜き。
    -   **パディング:** パディング比率によるフィルタリングは無効化（5%以上でも許容）。
    -   **I/O:** `imread_safe` を使用し、日本語ファイル名に対応。
3.  **Part 2a (`components/part2a_similarity.py`):**
    -   **重複排除:** 特徴量ベクトルを計算し、DBSCANを用いてデータセット内の「ほぼ同じ画像」を削除。
    -   **削除モード:** `PHYSICAL_DELETE = True` で物理削除、`False` で `deleted/` へ移動。
4.  **Part 2b (`components/part2b_filter.py`):**
    -   **外れ値除去:** クラスタリングを行い、主となる人物（最大クラスタ）を特定し、それ以外（他人の顔など）を除去。
    -   **削除モード:** `PHYSICAL_DELETE = True` で物理削除、`False` で `deleted/` へ移動。
    -   **整理:** 処理後、`rotated/` から親ディレクトリへ画像を移動し、空フォルダを削除。

**出力:** `master_data/{label}/...`（クリーニング済みのクロップ顔画像）

---

### ステージ 2: 前処理と高度なフィルタリング
**スクリプト:** `preprocess_multitask.py`
**入力:** `train/`, `validation/`, `test/` (Train/Val/Test 分割済みデータ)
**処理内容:**
1.  **解析フェーズ:**
    -   全画像をスキャンして解析。
    -   ランドマーク（106点）と姿勢（Pose）を抽出。
    -   指標計算:
        - Pitch（顔の上下向き）
        - Symmetry（左右対称性）
        - Y-Diff（傾き）
        - Mouth Open（口の開き）
        - Eyebrow-Eye Distance（眉と目の距離）
        - **Sharpness（画像の鮮明度）** ← Laplacian Varianceによるぼやけ検出
    -   **無効な顔:** 顔検出に失敗した画像はここで除外される。
2.  **閾値計算:**
    -   **全体統計 (Global):** Pitch, Symmetry, Y-Diff, Mouth Open, **Sharpness** は全データの分布からパーセンタイル閾値を算出。
    -   **個人別統計 (Per-Person):** **眉と目の距離** は、個人（ラベル）ごとの分布から閾値を算出し、個人差（骨格の違い）を吸収する。
3.  **フィルタリング:**
    -   計算された閾値（引数で指定された上位・下位%）に基づいて画像を選別。
    -   引数: `--pitch_percentile`, `--eyebrow_eye_percentile_high`, `--sharpness_percentile_low` 等。
4.  **コピー:** 合格した画像のみを出力ディレクトリへコピー。

**出力:**
- `preprocessed_multitask/train/{ラベル}/{人物名_ファイル名}.jpg`
- `preprocessed_multitask/validation/...`
- `preprocessed_multitask/test/...`
- **注意:** 学習スクリプトの読み込み効率化のため、人物フォルダ階層は削除され、ファイル名にプレフィックスとして付与される（フラット化）。

**高速化機構 (Caching):**
- 初回解析結果を `outputs/cache/metrics_*.pkl` に保存。
- 2回目以降の実行（閾値変更時など）は、画像解析をスキップしキャッシュを利用するため、数秒で完了する。

### ステージ 3: パラメータ探索（最適化）
**スクリプト:** `optimize_sequential.py` (Preprocess + Train(簡易) のラッパー)
**目的:** 最適なフィルタリング設定を自動探索する。
**処理内容:**
1.  **ベースライン評価:** 最初に全パラメータ=0 を評価してキャッシュに登録。
2.  **逐次最適化:** フィルタパラメータを1つずつ最適化する。
    -   順序: Pitch -> Symmetry -> Y-Diff -> Mouth Open -> Eb-Eye High -> Eb-Eye Low -> **Sharpness Low** -> Face Position
3.  **初期化:** 探索点 `[0, 5, 50]` を最初に評価する。
4.  **絞り込み:** 上位2点の間で2分探索（Refinement）を行い、最適値を特定する。
5.  **実行:**
    -   指定パラメータで `preprocess_multitask.py` を実行。
    -   `components/train_for_filter_search.py`（探索用軽量学習）を実行してモデル精度を評価。
    -   結果をキャッシュし、再実行を回避。

**出力:** 最適化されたパラメータセット（ログ表示）。これを見てステージ 4 を実行する。

---

### ステージ 4: 本番学習
**スクリプト:** `train_sequential.py`
**目的:** 最適化されたデータセットを用いて、最終的な高精度モデルを学習する。
**処理内容:**
1.  **準備:** ステージ 3 で特定された最適パラメータを用いて、手動（またはバッチ）で `preprocess_multitask.py` を一度実行し、学習データを確定させる。
2.  **学習実行:**
    -   Task A（多ラベル分類）などの本番用モデル定義を用いて学習を行う。
    -   `train_single_trial_task_a.py` などを内部で呼び出し、本格的なエポック数で学習。
    -   Imbalance対策（Balanced Accuracyなど）や、保存ロジック（最高精度のモデルのみ保存）が含まれる。
3.  **評価:** テストデータセットに対する最終評価を行う。

**出力:** 学習済みモデルファイル (`.h5` 等)、学習ログ、混同行列などの評価結果。

## 3. 主要な設定・定数

### フィルタリング指標 (InsightFace 106点モデル)

1.  **Pitch (顔の上下向き)**
    -   計算: `FaceAnalysis` の `pose[0]` 値（絶対値）を使用。
2.  **Symmetry (左右対称性)**
    -   対象: 頬の外側
    -   左頬: Index **28**
    -   右頬: Index **12**
    -   計算: 画像中心からの距離の比率 `abs(LeftDist / RightDist - 1)`
3.  **Y-Diff (頬の高さ差＝傾き)**
    -   対象: 頬の外側 (上記と同じ)
    -   左頬: Index **28**
    -   右頬: Index **12**
    -   計算: Y座標の差を画像高さで正規化 `abs(Ly - Ry) / Height`
    -   ※ Face Position Filter: 片方の頬が画面中心を超えている場合（極端な横顔）を除外。
4.  **Mouth Open (口の開き)**
    -   対象: 唇の内側中心
    -   上唇: Index **62**
    -   下唇: Index **60**
    -   計算: Y座標の差を画像高さで正規化 `abs(LowerY - UpperY) / Height`
5.  **Eyebrow-Eye Distance (眉と目の距離)**
    -   対象: 眉と目の上端付近
    -   右眉: Index **49**, 右目: Index **40**
    -   左眉: Index **104**, 左目: Index **94**
    -   計算: `(abs(R_Brow_Y - R_Eye_Y) + abs(L_Brow_Y - L_Eye_Y)) / 2 / Height`
6.  **Sharpness (画像の鮮明度)** ← NEW
    -   計算: `cv2.Laplacian(gray, cv2.CV_64F).var()`
    -   値が高い = シャープ、低い = ぼやけている
    -   用途: 低解像度からアップスケールされた画像や、圧縮劣化した画像を除外

### 削除モード設定
-   `download_and_filter_faces.py` 内の定数 `PHYSICAL_DELETE` で一括設定
    -   `True`: 不要画像を完全削除
    -   `False`: 以下のフォルダへ移動（論理削除） **← 現在のデフォルト**
        -   **Part 2a (重複):** `deleted_duplicates/`
        -   **Part 2b (別人):** `deleted_outliers/`

### その他設定
-   **日本語パス対応:** 独自の `imread_safe` 関数を全スクリプトで使用。バイナリモードで読み込み `cv2.imdecode` する方式。

## 4. ディレクトリ構造
```
root/
  master_data/            # ステージ1の出力: {label}/...
  train/                  # 学習用データ（人物単位分割後）
  validation/             # 検証用データ
  test/                   # テスト用データ
  preprocessed_multitask/ # ステージ2の出力: {train,validation,test}/{label}/...
  components/             # サブスクリプト群: part1/2a/2b, train_...
  outputs/
    logs/                 # 各スクリプトのログファイル
    cache/                # 最適化キャッシュ (JSON)
    models/               # 学習済みモデル
  pipeline_specs.md       # 本仕様書
  scripts_overview.md     # スクリプト一覧
```

## Author
Gemini (Updated 2025-12-28)