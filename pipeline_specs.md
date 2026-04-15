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
2.  **Part 1 (`components/part1_setup.py`):** ※2パス方式
    -   **パス1: ダウンロード & 顔サイズ収集**
        -   全画像をダウンロード後、InsightFaceで顔を検出しbbox幅（face_size）を記録。
        -   画像はstaging領域に一時保持。
    -   **パス2: 正規化 & 加工**
        -   全画像のface_sizeから75パーセンタイル（上位25%境界）を算出。
        -   **Face Size正規化:** face_size > 75pctの画像は、元画像を `scale = 75pct / face_size` で縮小してから処理。(2026-02-12追加)
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
    -   **外れ値除去:** DBSCANクラスタリングを行い、**上位2人物（上位2クラスタ）**を残し、それ以外を除去。
    -   **人物フォルダ分け:** 残したクラスタは `person_clusters/person_1/`, `person_clusters/person_2/` に分けて保存。
    -   **削除先:** `deleted_outliers/`（論理削除モード時）
    -   **整理:** `rotated/` は空になるため削除する（残す画像は `person_clusters/` 側に移動済み）。

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
- 探索空間（Model, Layers, Dropout, Augmentation等）を順次固定しながら最適化するSequential Optimizationを採用。
- **評価指標の変更 (2026-02-11):**
    - 従来の `Balanced Accuracy` (平均再現率) に代わり、**`MinClassAccuracy` (最低クラス精度)** を採用。
    - 各クラスの再現率（Recall）のうち、**最も低い値**を最大化するように学習・最適化を行う。
    - これにより、難易度の高いクラスや少数派クラスの放置を防ぎ、全てのクラスで一定以上の精度を保証することを目指す。
- **学習率キャリブレーション (2026-02-14):**
    - `train_multitask_trial.py` に `--auto_lr_target_epoch` オプションあり（スタンドアロン用）。
    - `BEST_EPOCH: N` を標準出力し、キャリブレーション時に利用可能。
    - **Step 1 (初期LR):** グリッドサーチを廃止し、`calibrate_base_lr` を採用。
        - 10 epoch の学習を繰り返し、Best epochが epoch 5 に来るようLRを調整し、それをベースのLRとして採用する。
        - キャリブレーション時はEarly Stoppingを無効化し、必ず全Epoch学習して真のBest Epochを特定する。
    - **Step 3.5 (Fine-Tuning LR):** Fine-tuning前にLRキャリブレーションを実施。
        - Target Epochを **10〜15の範囲で探索（6点）** し、最も検証精度が高かったLRを採用する。（`search_ft_lr_by_targets`）
        - `unfreeze_layers=60` を暫定値として使用。
    - **Step 4.5-4.7 (FT後の再最適化):**
        - unfreeze_layers確定後、必要に応じてFT LRを再探索（Target=10〜15）。
        - FT条件下でdropout, head_dropout, weight_decayを再最適化。
        - 正則化パラメータ変更後に Final FT LR Calibration (Target 10〜15探索) を実施。
- **Fine-tuning:**
    - 最適化されたパラメータを用いて、最終的に全層解凍によるFine-tuningを実施。
    - **FT LRの外部制御 (2026-02-14):** `train_multitask_trial.py` のPhase 2 LRハードコード(`/100`)を廃止。`--learning_rate` をそのまま使用し、`train_sequential.py` のLRキャリブレーションで最適値を決定。
    - Phase 2から `FT_BEST_EPOCH: N` を出力し、FTキャリブレーション時に利用。
    - **詳細ログ出力 (2026-02-11):**
        - 検証データに対する全クラスの個別精度（正解数/総数）を出力し、ボトルネックとなっているクラスを特定可能にした。


### ステージ 4: フィルタリングパラメータ最適化
**スクリプト:** `optimize_sequential.py` (NN版), `optimize_svm_sequential.py` (SVM版)
**最適化手法:**
- **LRキャリブレーション (Step -1):** (2026-02-14 / 2026-02-14更新)
    - 最適化開始前に、フィルタなしデータで20 epochの学習を最大5回繰り返す。
    - Best epochが中間（epoch 10）に来るようLRを二分探索で調整。
    - 調整式: `new_lr = current_lr × (best_epoch / 10)`
    - 収束条件は `BestEpoch==10` を厳密採用（許容誤差0）。一致しない場合は最大5回まで調整を継続（LR変動幅による早期終了は行わない）。
    - **Early Stopping無効化:** キャリブレーション時はEarly Stoppingを無効化し、必ず全Epoch学習して真のBest Epochを特定する。
    - 各試行の候補から「epoch10への距離最小（同距離ならスコア高い方）」を最終採用。
    - 得られた `calibrated_base_lr` を全後続trialで使用。
    - キャリブレーション最終結果をB0のベースラインスコアとして流用（重複排除）。
    - **2段階スケーリング (Two-Stage LR Scaling):** (2026-02-15更新)
        - `calibrate_lr_scaling.py` で2つの指数 `exp1`, `exp2` を独立に最適化。
      ### 2.2 LRスケーリングロジック（適応型）

- **目的**: フィルタリングによるデータ量の減少に応じて、学習率を自動調整する。
- **計算式**:
  - `relative_ratio = ratio / base_ratio`
  - `exponent = max(0.65, relative_ratio ** 0.66)` (動的指数)
  - `adjusted_lr = base_lr * (relative_ratio ** exponent)`
- **キャリブレーション (`calibrate_lr_scaling.py`)**:
  - 各フィルタレベル (25%, 50%, 75%) で学習を実行。
  - 評価指標: `abs(21 - score * 30) + distance` (Target Score 0.7 & Epoch 10, 低いほど良い)
  - 各レベルの最適Expを探索・分析し、動的指数の決定に利用する。
  - `optimize_sequential.py` は `base_lr` を設定ファイルから読み込み、`exponent` は上記数式で決定する。
    - 学習率スケジューラは前半5epochを固定LRとし、6epoch目からCosine Decayを開始。
    - 各trial: 20 epoch、Fine-tuning Off で評価。
    - **学習延長 (Conditional Extension):** 評価スコアが極端に低い場合(<0.5)、または学習の最終エポックがベストスコアだった場合、未収束と判断してLRを最小値(初期値の5%)に落とし、精度が下がるまで最大20エポック学習を延長する (2026-02-20 更新)。
- **Phase 1 - 独立パラメータ評価:**
    - 各パラメータを個別に評価（初期探索点: 0, 25, 50, 75）。
    - スコア上昇が見られた場合、**「精度スコア(score)単体の Top 2」**および**「精度上昇効率(efficiency)の Top 2」**の双方について中間点を設ける二分探索を実施。評価した中間点がどちらかの上位2位（Top2）に入り続ける限り探索を継続（どちらのTop2からも外れれば終了）。
    - 全探索点から、ベースラインからの「精度向上分」と「フィルタリング枚数」を計測し、**効率 (Efficiency) = 精度向上 / (フィルタリング枚数 + 1)** を計算。
    - 単体でベストスコアを出したパラメータ (Single Best) および最も効率の良いパラメータを記録。
- **Phase 2 - 効率ベースの貪欲法 (Efficiency-Based Greedy Integration):**
    - 効率が高い順にパラメータをソートし、貪欲法的に統合。精度が下がった時点で統合を停止。
    - **Grayscaleは含めず**、フィルタパラメータのみで実行。
    - **適応型加重平均 LR Exponent (2026-02-20 変更):** 複数のフィルタが同時に適用される場合、各フィルタの強度（例: pitch=5 の 5）を重みとし、パラメータ固有の `exp` を用いた加重平均 (`sum(exp * power) / sum(power)`) を計算し、その時点の動的 `exponent` として採用する。
- **Final Selection:**
    - Greedy Integration結果、Single Best、Original(全パラメータ適用)を比較し、最高スコアの戦略を選択。
- **Grayscale Test (最終ベスト決定後):**
    - Final Selectionで決定したベスト構成に対してのみ、Color vs Grayscaleを比較。
    - **Grayscaleは最適化プロセスに統合せず、最終結果に対する後処理として評価する。** (2026-02-12 変更)

---

## Author
Gemini (Updated 2026-02-15)

