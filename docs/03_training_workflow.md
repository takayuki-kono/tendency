# 学習ワークフローと最適化戦略 (Training Workflow & Optimization Strategy)

本ドキュメントでは、モデルの学習パイプラインおよび前処理フィルタの最適化戦略について、実装レベルの詳細仕様を記述します。

---

## 1. フィルタ最適化 (Preprocessing Tuning)

本プロジェクトでは、学習データの品質を高めるため、データ取得後・学習前に「フィルタ閾値の最適化」を行います。
これには **SVMベース（推奨）** と **CNNベース** の2つの手法が実装されていますが、主にSVMベースを使用します。

### 手法A: SVMベース最適化 (高速・推奨)
- **スクリプト**: `optimize_svm_sequential.py`
- **目的**: 高速に最適なフィルタパラメータを探索する。
- **仕組み**:
    1.  **特徴抽出**: 全画像に対し **InsightFace (ArcFace)** を適用し、512次元のEmbeddingベクトルを抽出。
        - キャッシュ: `outputs/cache/embeddings/` (ファイルハッシュで管理)
    2.  **分類器**: 軽量な **SVM (RBFカーネル)** を使用。データ量が少なくても安定しやすい。
    3.  **評価指標**: **Balanced Accuracy** (クラス不均衡を考慮した正解率)。
        - 評価データ: `validation` フォルダのデータを使用 (Train/Val固定分割)。
        - ※以前はCross-Validationでしたが、個人識別によるデータリーク（スコア99%張り付き）を防ぐため固定分割に戻しました。

#### 最適化フロー詳細
1.  **ベースライン評価**: 全フィルタ無効 (0%) でのスコアを計測。
2.  **独立パラメータ探索 (Independent Optimization)**:
    - 各パラメータ（Pitch, Sym, Y-Diff等）を個別に変化させ、最適値を探索。
    - **探索点**: `[0, 25, 50]` (パーセンタイル: 上位N%をカット)。
    - **詳細探索 (Refinement)**: 最も良かった2点の中間値（二分探索）を追加評価し、極値を特定。
3.  **全体スケーリング (Global Scaling)**:
    - 上記で得られた「各パラメータの比率」を保ったまま、強度を一律に調整。
    - **係数**: `[1.0, 0.5, 0.25]` (例: 係数0.5なら、各カット率を半分にする)。
4.  **結果出力**: 最適化されたパラメータを用いた `preprocess_multitask.py` の実行コマンドを表示。

### 手法B: CNNベース最適化 (高精度・低速)
- **スクリプト**: `optimize_sequential.py`
- **仕組み**: 実際にCNN (`EfficientNetV2`) を数Epoch学習させて評価する。
- **特徴**: SVMよりも最終的なタスクに近いが、1試行に時間がかかる（数分/回）。
- **モデル選択ステップ**:
    - 最初に `EfficientNetV2B0` と `EfficientNetV2S` を比較し、勝った方を採用するロジックが含まれる。

---

## 2. 前処理とフィルタリング仕様 (`preprocess_multitask.py`)

最適化されたパラメータを受け取り、実際に画像をフィルタリング（選別）して `preprocessed_multitask` フォルダに出力します。

### フィルタ指標の定義
| 指標名 | 引数名 | 定義・計算式 | 目的 |
| :--- | :--- | :--- | :--- |
| **Pitch** | `--pitch_percentile` | `abs(face.pose[0])` | 顔の上下の角度。俯き/煽りを排除。 |
| **Symmetry** | `--symmetry_percentile` | `abs(face_center_x - image_center_x)` | 顔の中心が画像中心からどれだけズレているか (左右対称性)。 |
| **Y-Diff** | `--y_diff_percentile` | `abs(left_eye_y - right_eye_y)` | 顔の傾き (Roll)。首を傾げている画像を排除。 |
| **Mouth Open** | `--mouth_open_percentile` | `abs(lower_lip_y - upper_lip_y)` | 口の開き具合。 |
| **Eb-Eye Dist** | `--eyebrow_eye_...` | `(R_brow_eye + L_brow_eye) / 2` | 眉と目の距離。表情（驚き・険しさ）や骨格特徴。 |
| **Sharpness** | `--sharpness_percentile_low` | `Laplacian(gray).var()` | 画像の鮮明度（ピンボケ判定）。 |
| **Aspect Ratio** | `--aspect_ratio_cutoff` | `box_height / box_width` | 顔検出枠の縦横比。極端に細長い/潰れた誤検出を排除。 |

### フィルタリングロジック
1.  **パーセンタイル計算**:
    - 全画像（Validなもの）の指標分布を計算。
    - 指定されたパーセンタイル（例: 10%）に基づき、閾値を決定（例: 上位10%をカットする閾値）。
    - *Eb-Eye Distのみ*: **個人ごと** に分布を取り、その中での外れ値をカット（個人差を吸収するため）。他は全体分布を使用。
2.  **判定**:
    - 各画像について、全指標が閾値内であれば「採用」。一つでも超えれば「不採用（スキップ）」。
3.  **アンダーサンプリング**:
    - クラス間のデータ数差を埋めるため、最も少ないクラス（または平均）に合わせて多いクラスのデータをランダムに間引く処理。

---

## 3. 本番学習パイプライン (`train_sequential.py`)

前処理済みデータを用いて、最終的なマルチタスクモデルを学習します。

### 学習フェーズ (`train_single_trial_task_a.py` 内部)
1.  **Phase 1: Warmup (転移学習)**
    - **対象**: `EfficientNetV2` のバックボーンを凍結 (Freeze)。Head層（全結合層）のみ学習。
    - **設定**: LR = `1e-3`, Epochs = 10 (EarlyStoppingあり)。
    - **目的**: ランダム初期化されたHead層を、バックボーンの特徴量に馴染ませる。いきなり全層学習するとバックボーンが壊れるのを防ぐ。
2.  **Phase 2: Fine-tuning (微調整)**
    - **対象**: バックボーンの上位層を解凍 (Unfreeze)。全層（または一部）を学習。
    - **設定**: LR = `1e-5` (低学習率), Epochs = 30。
    - **目的**: タスク特有の特徴をバックボーンに学習させる。

### モデル設定
- **アーキテクチャ**: `EfficientNetV2B0` (デフォルト) または `V2S`。
- **入力サイズ**: 224 x 224 px
- **出力 (マルチタスク)**:
    - Task A: `Dense(2, activation='softmax')` (傾向: a, b)
    - Task B: `Dense(2, ...)` (d, e)
    - ...
- **損失関数**: `SparseCategoricalCrossentropy`
- **評価指標**: `BalancedSparseCategoricalAccuracy` (自作指標。データ不均衡に頑健)

### ラベル定義
- **ソース**: `components/train_for_filter_search.py`
- **Task A**: `['a', 'b']` (フォルダ名の1文字目)
- ※ データセットのフォルダ名（例: `adgi/女優A`）から自動的にパースされます。

---

## 4. ディレクトリ構造

```
d:\tendency\
├── master_data/           # [入力] ダウンロードされた生データ (女優名フォルダ)
├── preprocessed_multitask/# [中間] フィルタリング済み学習データ
│   ├── train/             # train_sequential.py はここを見る
│   │   ├── adgi/          # ラベルフォルダ
│   │   └── ...
│   └── validation/
├── outputs/               # [出力]
│   ├── logs/              # 実行ログ
│   ├── cache/             # 最適化キャッシュ (削除推奨: filter_svm_opt_cache.json)
│   └── models/            # 学習済みモデル (.keras)
└── components/            # 内部スクリプト群
    ├── train_svm.py       # SVM学習用
    ├── part1_setup.py     # ダウンロード・クロップ
    └── ...
```

---

## 5. コマンドリファレンス

### 1. データの準備
```cmd
# キャッシュ削除 (データ変更時必須)
del outputs\cache\filter_svm_opt_cache.json

# ダウンロード・更新 (master_dataへ)
.venv_windows_gpu\Scripts\python.exe download_and_filter_faces.py

# ラベルフォルダへの整理 (master_data直下 -> master_data/ラベル/)
.venv_windows_gpu\Scripts\python.exe util/reorganize_by_label.py

# Train/Validation 分割 (画像数が多い順に上位50%をTrainへ)
.venv_windows_gpu\Scripts\python.exe util/create_person_split.py
```

### 2. フィルタ最適化
```cmd
# SVM版 (推奨)
.venv_windows_gpu\Scripts\python.exe optimize_svm_sequential.py
```

### 3. 本番学習
```cmd
# 前処理 (最適化で出たコマンドをコピペ、または手動指定)
.venv_windows_gpu\Scripts\python.exe preprocess_multitask.py --out_dir preprocessed_multitask ...

# 学習開始
.venv_windows_gpu\Scripts\python.exe train_sequential.py
```
