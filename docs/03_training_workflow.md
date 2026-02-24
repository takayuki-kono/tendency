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
    1.  **SVM HP探索**: C/gamma の候補からベストなSVMパラメータを選択。
    2.  **特徴抽出**: 全画像に対し **InsightFace (ArcFace)** を適用し、512次元のEmbeddingベクトルを抽出。
        - キャッシュ: `outputs/cache/embeddings/` (ファイルハッシュで管理)
    3.  **分類器**: 軽量な **SVM (RBFカーネル)** を使用。データ量が少なくても安定しやすい。
    4.  **評価指標**: **MinClassAccuracy** (最弱クラスの精度)。
        - 評価データ: `validation` フォルダのデータを使用 (Train/Val固定分割)。

#### 最適化フロー詳細
1.  **ベースライン評価**: 全フィルタ無効 (0%) でのスコアを計測。
2.  **独立パラメータ探索 (Independent Optimization)**:
    - 各パラメータ（Pitch, Sym, Y-Diff等）を個別に変化させ、最適値を探索。
    - **探索点**: `[0, 25, 50]` (パーセンタイル: 上位N%をカット)。
    - Sharpnessは **Low** (ボケ除去) と **High** (ノイズ除去) の2方向を探索。
    - **詳細探索 (Refinement)**: 探索されたスコア上位2点間の差が1%以下になるまで、再帰的に中間点を探索（二分探索）。これにより**1%単位の精度**で最適パラメータを特定する。
    - **選択ロジック (Selection)**: 各パラメータに対し「最高スコア(Best Score)の設定」と「最高効率(Best Efficiency)の設定」の2つを候補として保持する。
3.  **Greedy Integration**:
    - 上記で得られた各候補（Best Score設定、Best Efficiency設定）を、効率が高い順にソートしてGreedyに追加試行を行う。
    - **採用基準**: スコアが上昇し、かつ効率が良い設定を優先的に採用していく。
    - 上記で得られた「各パラメータの比率」を保ったまま、強度を一律に調整。
    - **係数**: `[1.0, 0.5, 0.25]` (例: 係数0.5なら、各カット率を半分にする)。
    - スコアが低下するパラメータはスキップし、残りの候補も継続して試す（途中で打ち切らない）。
4.  **結果出力**: 最適化されたパラメータを用いた `preprocess_multitask.py` の実行コマンドを表示。

### 手法B: CNNベース最適化 (高精度・低速)
- **スクリプト**: `optimize_sequential.py`
- **仕組み**: 実際にCNN (`EfficientNetV2`) を数Epoch学習させて評価する。
- **特徴**: SVMよりも最終的なタスクに近いが、1試行に時間がかかる（数分/回）。
- **探索設定**:
    - **探索点**: `[0, 2, 5, 25, 50]` (初期探索) -> 1%刻みの二分探索 (Refinement)。
- **Phase 1 結果サマリー**: 全パラメータの独立探索完了後に「精度上昇効率一覧」をログ出力。各パラメータの Best Score 候補と Best Efficiency 候補をテーブル形式で表示し、全体の Overall Best Score / Best Efficiency も出力する。
- **出力アーティファクト**:
    - `outputs/logs/sequential_opt_log_YYYYMMDD_HHMMSS.txt`: 実行ログ（タイムスタンプ付きで履歴保持）。
    - `outputs/optimization_analysis.json`: 最適化プロセスの全候補、Greedy統合の履歴、最終パラメータを含む詳細ログ。
    - `run_optimized_preprocess.bat`: 最適化されたパラメータを適用するための実行バッチファイル。
- **モデル選択ステップ**:
    - 最初に `EfficientNetV2B0` と `EfficientNetV2S` を比較し、勝った方を採用するロジックが含まれる。
- **LR自動調整リトライ** (全スクリプト共通: `optimize_sequential.py`, `train_sequential.py`。両者で条件・定数を同一にしている):
    - 各トレーニング実行後に `BEST_EPOCH` を確認し、終了条件を満たさなければ再調整（最大3回）。
    - **終了条件（キャリブレーションと同じ）**: (1) best_epoch 11～19 **かつ** last_epoch_accu≠best（差が LR_LAST_ACCU_EPS=0.01 以上）→ 調整終了 (2) **11≤best_epoch≤15 かつ last_epoch_accu < trial_score**（ピーク後下降）→ 再調整せず終了 (3) 試行回数が LR_MAX_ADJUSTMENTS に達した → 終了
    - **再調整条件**: best_epoch <= 10、best_epoch == 最終epoch、または last_accu ≈ best_accu（差 < 0.01、plateau）
    - **調整計算（時間軸）**: `new_lr = current_lr * best_epoch / target_epoch`（scale は 0.3～3.0 にクリップ）
    - 全試行中の最高スコアの結果を採用する。
- **Phase 1 タイブレーカー**:
    - Phase 1完了後、同じベストスコアを出した複数の候補値があるパラメータを検出。
    - 該当キャッシュを削除して再評価し、勝者を決定。


### 学習率の動的スケーリング (Dynamic LR Scaling)
前処理フィルタによりデータ量が減少した場合、学習率を以下の多項式曲線に基づいて自動調整します。
- **目的**: データ数が少ない場合（フィルタで厳しく選別した場合）に適した学習率へ動的に補正する。
- **計算式** (2026-02-18更新):
  - `exponent`: `outputs/lr_scaling_config.json` に保存された値を使用（パラメータごとに個別最適化済み）。
  - Exp Range: `0.15` ~ `1.0` (Binary Search)
  - `adjusted_lr = base_lr * (relative_ratio ** exponent)`
  - データ残存率 (`ratio`) が閾値（0.5）以上 (`High Ratio`) の場合は `exp1`、未満 (`Low Ratio`) の場合は `exp2` を適用。
- **Base LR決定ロジック**:
  - **目的**: **Validation Score 最大化** を最優先指標とする。
  - **ターゲットEpoch**: **10** (2026-02-19変更)。
   - **手順**:
     1. Epoch 10 に収束するLR (`lr_10`) を特定し、ベースLRとして採用。
  - **キャリブレーションの打ち切り**: (1) **11≤best_epoch≤19 かつ last_epoch_accu≠best**（差が LR_LAST_ACCU_EPS=0.01 以上で「≠best」）→ 終了 (2) **試行回数が LR_MAX_ADJUSTMENTS に達した** → 終了。run_calibration_trial は last_epoch_accu も返す。
- **キャリブレーション設定**:
  - **学習率減衰特性 (Exponent) の探索**:
    - 探索範囲: `0.15` ～ `1.0`
    - 固定探索点: `[0.25, 0.5, 0.75, 1.0]`
    - 主要パラメータ（Y-Diff, Sharpness, Symmetry等）ごとに個別の最適値を算出し、最適化時に使い分ける。

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
| **Sharpness(L)**| `--sharpness_percentile_low` | `Laplacian(gray).var()` | 画像の鮮明度（下位カット）。ピンボケ画像を除外。 |
| **Sharpness(H)**| `--sharpness_percentile_high`| `Laplacian(gray).var()` | 画像の鮮明度（上位カット）。高周波ノイズの強い画像を除外。 |
| **Aspect Ratio** | `--aspect_ratio_cutoff` | `box_height / box_width` | 顔検出枠の縦横比。極端に細長い/潰れた誤検出を排除。 |

### フィルタリングロジック
1.  **パーセンタイル計算**:
    - 全画像（Validなもの）の指標分布を計算。
    - 指定されたパーセンタイル（例: 10%）に基づき、閾値を決定（例: 上位10%をカットする閾値）。
    - **グループ化**: ソースの**ディレクトリパス**（例: `a/森口瑤子`）単位。眉-目距離(Eb-Eye)のパーセンタイルとアンダーサンプリングはこの単位で行うため、個人（1フォルダ＝1人）ごとに「その人の上位○%」が残る。他指標（pitch/symmetry/sharpness等）の閾値は従来どおり全体分布で計算。※ファイル名に個人名が付いていなくても、フォルダ構造で個人が分かれていれば個人単位になる。
2.  **判定**:
    - 各画像について、全指標が閾値内であれば「採用」。一つでも超えれば「不採用（スキップ）」。
3.  **アンダーサンプリング**:
    - クラス間のデータ数差を埋めるため、最も少ないクラス（または平均）に合わせて多いクラスのデータをランダムに間引く処理。

### preprocess_multitask.py の詳細

**入力**: `train/`, `validation/`, `test/`（各ディレクトリ直下に `{タスク}/{人物名}/` または `{タスク}/{サブフォルダ}/` のような相対パスで画像が並んでいる想定）。

**処理フロー**:
1. **スキャン**: 各ソースを `os.walk` で走査。画像拡張子 `.jpg/.png/.jpeg/.bmp` のファイルを列挙。各ファイルの「グループキー」= そのファイルがあるディレクトリの相対パス（例: `a/森口瑤子`）。
2. **顔分析（キャッシュあり）**: `outputs/cache/metrics_<hash>.pkl` をキー（ソース名+ファイル数）で参照。キャッシュがなければ InsightFace で全画像を解析し、顔検出・106ランドマークを取得。検出失敗・読込失敗は `valid=False` で除外。
3. **閾値計算**:
   - **グローバル閾値**: 全 valid 画像の指標分布から算出。`pitch_percentile=10` なら「上位10%を落とす」ので `th_pitch = percentile(pitch, 90)`。同様に symmetry, y_diff, mouth_open, sharpness(低/高), face_size(低/高), aspect_ratio(両側), retouching, mask, glasses を計算。
   - **個人閾値**: グループ（ディレクトリパス）ごとに **眉-目距離 (eb_eye_dist)** だけ、そのグループ内の分布で `eyebrow_eye_percentile_low` / `eyebrow_eye_percentile_high` の閾値を計算。
4. **判定**: 各画像について、上記の全閾値と比較。**一つでも閾値を超えたらスキップ**（採用されない）。スキップ理由は `pitch_global`, `symmetry_global`, `eb_eye_low_personal`, `undersampling` などでログに集計される。
5. **アンダーサンプリング**: グループごとの採用枚数の**平均** `target_count = mean(counts)` を算出。あるグループの採用枚数が `target_count` を超えていれば、そのグループ内でランダムシャッフルしたうえで先頭 `target_count` 枚だけ残し、残りはスキップ（`skip_reasons['undersampling']`）。train/validation/test いずれも同じロジック（`skip_undersampling` は通常 False）。
6. **コピー**: 採用された画像を `out_dir/train/`, `out_dir/validation/`, `out_dir/test/` にコピー。出力ファイル名は `rel = os.path.relpath(src, src_root)` の `parts` の先頭をディレクトリ、`parts[1:]` を `_` で連結した名前（例: `a/森口瑤子/foo.jpg` → `out_dir/train/a/森口瑤子_foo.jpg`）。`--grayscale` 指定時はグレースケール変換してから保存。

**指標の定義（実装準拠）**:

| 指標 | 計算 | 閾値の向き（percentile>0 のとき） |
|------|------|-----------------------------------|
| Pitch | `abs(face.pose[0])`（InsightFace） | 値 **>** 閾値 → 除外（上向き/下向きが大きい） |
| Symmetry | `abs(face_center_x - image_center_x)`（両目中心と画像中心の差） | 値 **>** 閾値 → 除外（横顔寄り） |
| Y-Diff | `abs(left_eye_y - right_eye_y)` | 値 **>** 閾値 → 除外（首の傾き） |
| Mouth Open | `abs(lower_lip_y - upper_lip_y)`（ランドマーク） | 値 **>** 閾値 → 除外（口開きすぎ） |
| Eb-Eye Dist | 左右の (眉Y−目Y) の平均（ピクセル） | **個人内**で 値 **>** th_high または 値 **<** th_low → 除外 |
| Sharpness | `cv2.Laplacian(gray).var()` | 値 **<** th_low → 除外（ボケ）、値 **>** th_high → 除外（ノイズ/過シャープ） |
| Face Size | ファイル名の `_sz(\d+)` から取得（0 の場合は対象外） | 値 **<** th_low または **>** th_high → 除外 |
| Aspect Ratio | 顔検出枠の `height/width` | 値 **<** th_low または **>** th_high → 除外 |
| Retouching | 肌領域の Sobel 高周波成分の平均（低いほど加工疑い） | 値 **<** 閾値 → 除外 |
| Mask | 上顔/下顔の肌色比率から算出（高いほどマスク疑い） | 値 **>** 閾値 → 除外 |
| Glasses | 目周辺エッジと額エッジの比（高いほど眼鏡疑い） | 値 **>** 閾値 → 除外 |

**引数（0＝フィルタ無効）**: `--pitch_percentile`, `--symmetry_percentile`, `--y_diff_percentile`, `--mouth_open_percentile`, `--eyebrow_eye_percentile_low` / `--eyebrow_eye_percentile_high`, `--sharpness_percentile_low` / `--sharpness_percentile_high`, `--face_size_percentile_low` / `--face_size_percentile_high`, `--aspect_ratio_cutoff`, `--retouching_percentile`, `--mask_percentile`, `--glasses_percentile`, `--grayscale`。  
**出力**: `preprocessed_multitask/train/`, `preprocessed_multitask/validation/`, `preprocessed_multitask/test/`。これが `train_sequential.py` の直接の入力。

---

## 3. 本番学習パイプライン (`train_sequential.py`)

前処理済みデータを用いて、最終的なマルチタスクモデルを学習します。

### 学習フェーズ (`train_multitask_trial.py` 内部)
1.  **Phase 1: Warmup (転移学習)**
    - **対象**: `EfficientNetV2` のバックボーンを凍結 (Freeze)。Head層（全結合層）のみ学習。
    - **設定**: 目標 LR = `warmup_lr`（キャリブレーション済み）, Epochs = `warmup_epochs`（デフォルト5）。
    - **LR スケジュール**: Warmup 中は **LinearWarmupScheduler** により、Epoch 1 で `warmup_lr/n`、Epoch n で `warmup_lr` まで線形に増加（本来の warmup: 徐々に LR を上げる）。
    - **目的**: ランダム初期化されたHead層を、バックボーンの特徴量に馴染ませる。
    - **条件**: `warmup_lr > 0` かつ `fine_tune=True` の場合のみFT前に実行。
    - **ベスト重みの復元**: Warmup 中に val min_class_accuracy が最良だったエポックの重みを保存し、Warmup 終了後に復元してから FT に進む（epoch 2 が best など道中が良い場合に対応）。
2.  **Phase 2: Fine-tuning (微調整)**
    - **対象**: バックボーンの上位層を解凍 (Unfreeze)。`--unfreeze_layers` で層数指定可能 (デフォルト40)。
    - **設定**: LR = `1e-5` (低学習率), Epochs = 50。
    - **目的**: タスク特有の特徴をバックボーンに学習させる。
    - `train_sequential.py` で解凍層数 (20/40/60/全層) を探索して最適値を選択。

### モデル設定
- **アーキテクチャ**: `EfficientNetV2B0` (デフォルト) または `V2S`。
- **入力サイズ**: 224 x 224 px
- **出力 (マルチタスク)**:
    - Task A: `Dense(2, activation='softmax')` (傾向: a, b)
    - Task B: `Dense(2, ...)` (d, e)
    - ...
- **損失関数**: `SparseCategoricalCrossentropy` (Mixup/LabelSmoothing使用時は `CategoricalCrossentropy`)
- **評価指標**: `MinClassAccuracy` (最弱クラスの精度、全タスク平均)
- **EarlyStopping**: 全タスク平均MinClassAccuracyが改善しなくなったらpatience回猶予後に停止 (損失関数ではなく精度ベース)
- **学習率スケジュール**: Polynomial Decay (常に有効)
    - **開始条件**: **常に有効** (学習開始時から減衰を適用)。
    - **減衰計算**: 全エポック数に対する進捗 `progress` を基に `1.0 - progress` で線形に減衰させる。
    - **最低LR**: `initial_lr × 0.05` (ゼロにはしない)。
- **条件付きEpoch拡張 (Conditional Extension)**:
    - ベストエポックが最終エポック、または最終エポックのスコアがベストスコアと同等の場合、または `Balanced Accuracy < 0.5` の場合、追加学習モードに入る。
    - **学習率**: `min_lr` (initial_lr × 0.05) 固定。
    - **終了条件**: 精度（平均MinClassAccuracy）が下がったら即停止、または最大20エポック追加。
- **Weight Decay**: Dense層の `kernel_regularizer=l2(wd)` で実装 (AdamW不要)
- **Mixup**: Beta(α, α) 分布からサンプリング (Gamma分布2つから構築)
- **検証データ**: Mixup/Label Smoothingは適用しない (生データで評価)

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
