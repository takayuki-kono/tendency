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
    - **探索点**: `[0, 5, 25, 50, 75]` (初期探索) -> 1%刻みの二分探索 (Refinement)。
- **Phase 1 結果サマリー**: 全パラメータの独立探索完了後に「精度上昇効率一覧」をログ出力。各パラメータの Best Score 候補と Best Efficiency 候補をテーブル形式で表示し、全体の Overall Best Score / Best Efficiency も出力する。
- **出力アーティファクト**:
    - `outputs/logs/sequential_opt_log_YYYYMMDD_HHMMSS.txt`: 実行ログ（タイムスタンプ付きで履歴保持）。
    - `outputs/optimization_analysis.json`: 最適化プロセスの全候補、Greedy統合の履歴、最終パラメータを含む詳細ログ。
    - `run_optimized_preprocess.bat`: 最適化されたパラメータを適用するための実行バッチファイル。
- **Step 0（モデル＋head LR）**:
    - `MODEL_NAME_CANDIDATES`（`model_architecture.py`）の**各** `model_name` に対し、フィルタなし前処理の上で **`calibrate_base_lr`（`target_best_epoch=13` 他、他軸と同ロジック）**を実行し、キャリブ終了時の Val スコアが最大のモデルを採択。
    - 採択モデル専用の `CALIBRATED_BASE_LR` を以降の `run_trial`（フィルタ軸最適化）の基準 LR とし、`best_train_params.json` の `model_name` / `learning_rate_head` を更新。
- **LR自動調整リトライ** (全スクリプト共通: `optimize_sequential.py`, `train_sequential.py`。両者で条件・定数を同一にしている):
    - 各トレーニング実行後に `BEST_EPOCH` を確認し、終了条件を満たさなければ再調整（`LR_MAX_ADJUSTMENTS=6` まで。合計試行は `range(LR_MAX_ADJUSTMENTS+1)` により **最大 7 回**）。
    - **終了条件**（`components/lr_adjustment.py` で共通化）: (1) best_epoch が LR_ACCEPTABLE_MIN～LR_ACCEPTABLE_MAX(11～15) **かつ** last_epoch_accu≠best（差≥0.01）→ 調整終了 (2) 同範囲 **かつ** last_epoch_accu < trial_score（ピーク後下降）→ 再調整せず終了 (3) 試行回数が LR_MAX_ADJUSTMENTS に達した → 終了
    - **再調整条件**: best_epoch <= 10、best_epoch == 最終epoch、または last_accu ≈ best_accu（差 < 0.01、plateau）
    - **調整計算（時間軸）**: `ratio = best_epoch / target_epoch`（**比の上下限なし**）で `new_lr = current_lr * ratio`（実 LR は `clip_learning_rate_for_training` の絶対域。auto_lr の sqrt スケールは別）
    - 全試行中の最高スコアの結果を採用する。
- **Phase 1 タイブレーカー**:
    - Phase 1完了後、同じベストスコアを出した複数の候補値があるパラメータを検出。
    - 該当キャッシュを削除して再評価し、勝者を決定。
- **Validation クラス最小サンプル数ガード**:
    - `run_trial` は preprocess 直後に `preprocessed_multitask/validation/` を走査し、各タスク×各クラスの画像枚数の最小値を取得する（実装: `_min_val_class_count`）。
    - 最小値が `MIN_VAL_PER_CLASS`（既定 20）未満の候補は、学習せず `(0.0, total_images, filtered_count, val_min_cnt)` の 4-tuple をキャッシュに保存して即リターン（採点失敗扱い）。
    - キャッシュ値の互換仕様: 4-tuple（新形式: `val_min_cnt` 記録あり）と 3-tuple（旧形式: `val_min_cnt` 未記録）を両対応。
    - **キャッシュヒット経路のガード**:
        - 4-tuple: 記録された `val_min_cnt` を直接参照し、`< MIN_VAL_PER_CLASS` なら `score=0.0` に上書きして返す（`INVALIDATED (val undersize)` を WARNING 出力）。
        - 3-tuple: `val_min_cnt` 不明のため `saved_images = total - filtered` のヒューリスティックで推定。`saved < LEGACY_CACHE_MIN_SAVED`（既定 `MIN_VAL_PER_CLASS × 10 = 200`）なら同様に `score=0.0` に上書き（`INVALIDATED (legacy 3-tuple, saved=... )` を WARNING 出力）。
    - 目的: フィルタを強くしすぎて validation が激減すると、小標本ノイズで偶然の高精度（min class acc）が出て採用される問題を排除する。キャッシュヒットでも最新ガードが適用されるため、旧ラン由来のノイズエントリが `Selected` に昇格する事故を防ぐ。


### 学習率の動的スケーリング (Dynamic LR Scaling)
前処理フィルタによりデータ量が減少した場合、学習率を以下の多項式曲線に基づいて自動調整します。
- **目的**: データ数が少ない場合（フィルタで厳しく選別した場合）に適した学習率へ動的に補正する。
- **計算式** (2026-02-18更新):
  - `exponent`: `outputs/lr_scaling_config.json` に保存された値を使用（パラメータごとに個別最適化済み）。
  - Exp Range: `0.15` ~ `1.0` (Binary Search)
  - `adjusted_lr = base_lr * (relative_ratio ** exponent)`
  - 単一の exponent で `adjusted_lr = base_lr * (relative_ratio ** exponent)`。パラメータ別に individual_exponents が設定されていればその重み付き平均、なければデフォルト exponent を使用。

### Auto LR (steps_per_epoch ベース sqrt スケーリング)
`components/train_multitask_trial.py` 内部の補助スケーリング。`--auto_lr_target_epoch > 0` が渡された実行で発動し、学習画像枚数から算出した `steps_per_epoch` に対して sqrt ベースで LR を補正する。
- **計算式**: `lr_scale = sqrt(REFERENCE_STEPS_PER_EPOCH / steps_per_epoch)`（`REFERENCE_STEPS_PER_EPOCH = 20.0`、画像約 640 枚 / batch 32 で `scale=1.0`）。
- **クランプ**: 2026-04-25 に `max(0.3, min(lr_scale, 3.0))` を**撤廃**。LR 調整 ratio 側と同様、振動ガードは LR 再調整ループ／calibrate の反転検知 dampening に一任する設計に統一。
- **Base LR決定ロジック**:
  - **目的**: **target_best_epoch=13 への収束** を最優先指標とする。同率なら score で比較。
  - **ターゲットEpoch**: **13**（LR_TARGET_EPOCH に合わせて train/optimize 共通）。
   - **手順**:
     1. Epoch 13 に収束するLRを特定し、ベースLRとして採用。
  - **キャリブレーションの打ち切り**（`lr_calibration_should_stop`、run_trial の終了条件と同一）: (1) **11≤best_epoch≤15 かつ last_epoch_accu≠best**（差≥0.01）→ 終了 (2) **11≤best_epoch≤15 かつ last_accu < best**（ピーク後下降）→ 終了 (3) **試行回数が LR_MAX_ADJUSTMENTS に達した** → 終了。run_calibration_trial は last_epoch_accu も返す。
  - **候補採点ルール**（2026-04-22 修正）: 最終選択は `(distance, -score, ...)` タプルで比較する。すなわち `target_best_epoch` からの距離（`abs(best_epoch - target)`）が最小の iteration を優先採用し、距離同率のときのみ score（`MinClassAcc` など）の大きい方を採用する。
     - 旧仕様（`(-score, distance, ...)` = score 最優先）だと、noisy val 環境で target から離れた iteration が score 偶発で採用されるため、target=13 に寄せるループ自体が無意味化していた問題を修正。
     - `train_sequential.py` では `calibrate_base_lr(score_priority=False)` を全呼び出しで使用（デフォルト）。`score_priority=True` の旧挙動は互換のため残存するが実使用しない。
  - **探索方式**（2026-04-22 統一）: `optimize_sequential.py` / `train_sequential.py` の `calibrate_base_lr` は、両境界 `lr_low` / `lr_high` を記憶する **log 空間二分探索** で統一。
     - `best_epoch < target_min` → `lr_high = current_lr`（上限更新）。
     - `best_epoch >= target_min`（帯内含む） → `lr_low = current_lr`（下限更新）。
     - 両境界が揃ったら次 LR = `sqrt(lr_low * lr_high)`（幾何平均 = log 空間の中点）。LR は乗算スケールで効くため算術平均より幾何平均の方が対称で収束が速い。
     - 片側のみのとき: `scale = compute_lr_adjustment_ratio(...)`（`best_epoch/target_mid`）で `new_lr = current_lr * scale`（比のクランプなし）。
     - `LR_MAX_ADJUSTMENTS=6` → 最大 7 trial（0..6）。反対側境界を踏むまでの試行余裕を確保。
     - 収束判定: `|new_lr - current_lr| / current_lr < 0.02` なら以降の trial を打ち切り、最良候補を採用。
  - **LR スロット分離**（2026-04-22 明確化）: `best_train_params.json` の `learning_rate_head` / `learning_rate_nohead` / `learning_rate_ft` はそれぞれ head-only / body(backbone) / FT 用の別の条件で乖離する LR を保持する設計。head calibration（optimize_sequential の run_trial, および train_sequential Step 1/1.2）の結果は `learning_rate_head` のみを更新し、`learning_rate_nohead` には書き込まない（head LR を body 側にミラーすると body/FT 引き継ぎが狂うため）。読み取り側 `_get_head_lr_from_best` も `(learning_rate_head, warmup_lr, learning_rate)` の順で head 優先に並べ、`learning_rate_nohead` は head の fallback 候補に入れない。
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
5. **アンダーサンプリング** (2026-04-25 更新): グループごとの採用枚数のうち **2 番目に多いグループの採用枚数** を `target_count` とする（`counts` を降順ソートして `counts[1]` を採用。グループが 1 つしかない場合は `counts[0]` = 切らない）。あるグループの採用枚数が `target_count` を超えていれば、そのグループ内でランダムシャッフルしたうえで先頭 `target_count` 枚だけ残し、残りはスキップ（`skip_reasons['undersampling']`）。train/validation/test いずれも同じロジック（`skip_undersampling` は通常 False）。
    - **旧仕様との差分**: 以前は `target_count = int(mean(counts))` で平均まで切っていたため、最多 1 人に引きずられて中位以下のグループまで削られる副作用があった。新仕様では**最多の 1 人だけ**が 2 位に合わせて切り詰められ、他のグループは原則そのまま残る。
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
0.  **Head Carryover (Phase 0 ベスト重みロード)**
    - `--init_weights_path <path>` が指定され、該当ファイルが存在する場合、`create_model` 直後に `model.load_weights(path, by_name=True, skip_mismatch=True)` で重みをロードする。
    - 用途: `train_sequential.py` の Step 3.9 で保存した **Phase 0（head-only）ベスト重み**を、FT 開始時の初期重みとして引き継ぐ（head carryover）。これにより FT の冒頭で head が再初期化されず、従来の warmup フェーズが不要になる。
1.  **Phase 1: Warmup (転移学習)**
    - **対象**: `EfficientNetV2` のバックボーンを凍結 (Freeze)。Head層（全結合層）のみ学習。
    - **設定**: 目標 LR = `warmup_lr`（キャリブレーション済み）, Epochs = `warmup_epochs`（デフォルト5）。
    - **LR スケジュール**: Warmup 中は **LinearWarmupScheduler** により、Epoch 1 で `warmup_lr/n`、Epoch n で `warmup_lr` まで線形に増加（本来の warmup: 徐々に LR を上げる）。
    - **目的**: ランダム初期化されたHead層を、バックボーンの特徴量に馴染ませる。
    - **条件**: `warmup_lr > 0` かつ `fine_tune=True` の場合のみFT前に実行。
    - **ベスト重みの復元**: Warmup 中に val min_class_accuracy が最良だったエポックの重みを保存し、Warmup 終了後に復元してから FT に進む（epoch 2 が best など道中が良い場合に対応）。
    - **現在の運用**: Step 3.9 の head carryover が成功している場合は `warmup_lr=0` を渡して本フェーズをスキップする（`train_sequential.py` 側で設定）。保存ファイルが無いときのフォールバックとしてのみ Warmup が走る。
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
    - **patience 算出（2026-04-26）**: `create_callbacks` 内で `auto_lr_target_epoch`（= `target_epoch`）が正のとき **`max(3, target_epoch // 2) * 2`**、未使用（0）のとき **10**（従来の半分目安×2、および従来 5 の2倍）。`AccuracyEarlyStopping` クラス既定も 10。
    - **ベスト重み復元**（2026-04-25 修正）: `AccuracyEarlyStopping` コールバックは patience 超過時だけでなく、**学習終了時 (`on_train_end`) にも必ず best epoch の重みを復元**する。これにより 20 epoch 走り切って自然終了したケースでも、学習後の `model.predict(val_ds)` に基づく `[Detailed Class Accuracy]` 出力（`Class 'a': ...` 等）が best epoch の重みで計算され、`FINAL_VAL_ACCURACY` と per-class 内訳の基準がずれない。修正前は last epoch の重みで per-class が出力され、`Class 'z': 0.5918` 等と last 値になる不整合があった（例: `BestEpoch=16/20, Score=0.6020` なのに per-class min=0.5918）。
- **学習率スケジュール**: Polynomial Decay (常に有効)
    - **開始条件**: **常に有効** (学習開始時から減衰を適用)。
    - **減衰計算**: 全エポック数に対する進捗 `progress` を基に `1.0 - progress` で線形に減衰させる。
    - **最低LR（スケジュール上）**: `initial_lr × 0.05` (ゼロにはしない)。
    - **絶対域（2026-04-26）**: オプティマイザに入る各エポックの LR は `components/lr_adjustment.py` の **`LR_TRAIN_ABSOLUTE_MIN=1e-7`** ～ **`LR_TRAIN_ABSOLUTE_MAX=0.1`** に `clip_learning_rate_for_training` で収める。極小は `.8f` ログで 0 に見えて実質停止するのを防ぎ、極大は設定外れの保険。学習率確定直後（head/FT いずれも）・`ConditionalLearningRateScheduler` / `LinearWarmupScheduler` 各エポック・延長学習の `extension_lr` に適用。
    - **import 注意（2026-04-26 修正）**: `train_multitask_trial.py` は `components/` 配下でサブプロセス起動されるため、同ディレクトリの `lr_adjustment` は **`from lr_adjustment import ...`** とする。`from components.lr_adjustment ...` だと実行時 `sys.path` によっては **解決失敗**し、キャリブの `FINAL_VAL_ACCURACY` 抽出不能（score=0.0）になる。
- **条件付きEpoch拡張 (Conditional Extension)**:
    - **発動条件（2026-04-25）**: ベストエポックが最終エポック、**または**最終エポックのスコアがベストスコアと**同一**（`is_best_at_last`）の場合のみ、追加学習モードに入る。旧仕様の「score が閾値未満（例: 0.5 未満）」単独での発動は**廃止**（低スコアでも中盤で既にピークが取れていれば延長は無駄になるため）。
    - **学習率**: `min_lr` (initial_lr × 0.05) 固定。
    - **終了条件**: 精度が**厳密に下がったときのみ**停止（plateau＝同じのときは継続）。
    - **延長上限**: 1epochずつ継続し、下がらない限り続行（安全上限あり）。
    - 発動を防ぐには `train_multitask_trial.py` に `--no_extension` を付与する（CLI 単体実行・スクリプト起動形態によっては常時付与も可）。
- **Weight Decay**: Dense層の `kernel_regularizer=l2(wd)` で実装 (AdamW不要)
- **Mixup**: Beta(α, α) 分布からサンプリング (Gamma分布2つから構築)
- **検証データ**: Mixup/Label Smoothingは適用しない (生データで評価)

### `outputs/best_train_params.json` の学習率の扱い
- `train_sequential.py` の最終採用ハイパラを保存する。
- LRは **head学習（凍結 / `fine_tune=False`）用** と **FT（`fine_tune=True`）用** を分離して保持する。
    - **`learning_rate_nohead`**: headなし（凍結 / `fine_tune=False`）で使うLR（次回の凍結キャリブレーションの初期値にも使う）
        - 互換のため旧キー **`learning_rate_head`** も当面は読み書きする（非推奨）。
    - **`learning_rate_ft`**: FT本番で使うLR（次回のFTキャリブレーションの初期値にも使う）
    - **互換用**: 既存キーの `learning_rate` は主にFT側のLRとして残す（古いファイル読み込み時の互換のため）
    - **更新元**:
        - `optimize_sequential.py` は、LRキャリブレーションで `CALIBRATED_BASE_LR`（base_lr）が確定したタイミングで `learning_rate_nohead`（互換: `learning_rate_head`）を上書き更新する（次回のoptimize開始時の初期値に反映させるため）。
- **`train_multitask_trial.py` へ転送しないメタ情報キー**:
    - `learning_rate_nohead` / `learning_rate_head` / `learning_rate_ft` は `train_sequential.py` / `optimize_sequential.py` 側のメタ情報であり、`components/train_multitask_trial.py` の argparse には存在しない。
    - そのため `train_sequential.py` の `run_trial` / `run_calibration_trial`、および `optimize_sequential.py` の `run_trial` / `run_calibration_trial` でサブプロセス起動コマンドを組み立てる際、これらのキーは `_skip_keys` で除外する。
    - 実際に `train_multitask_trial.py` に渡すLRは、いずれの呼び出し側でも明示的に `--learning_rate <value>` として付与する。

### `train_sequential.py` の主要ステップ
- **`model_name` の扱い（2026-04-28 更新）**: `outputs/best_train_params.json` に `model_name` があっても `train_sequential` は **Step 1.1（バックボーン比較）をスキップしない**（**現在の前処理データ**で毎回確定）。**`optimize_sequential.py` Step 0** は `MODEL_NAME_CANDIDATES` ごとに **LR キャリブ**してから最高スコアの `best_model` を選び、同 JSON に `model_name` / `learning_rate_head` を書き戻す。`train_sequential` では 1.1 を再確定する想定。
- **Step 1: Base LR Calibration** — `target_best_epoch=13` を狙う head LR を、**`components/model_architecture.py` の `LRCALIB_BASE_BACKBONE`（既定 `EfficientNetV2B0`）固定**でキャリブ（保存 JSON の `learning_rate_head` 等は `_get_head_lr_from_best` の**初期候補**にのみ用い、バックボーン切替は 1.1 以降）。
- **Step 1.1: Model Architecture** — 毎回: `MODEL_NAME_CANDIDATES`（`model_architecture.py`、例: EfficientNet B0/S、ResNet50/101V2、ConvNeXtSmall、MobileNetV3Large、**`ConvNeXtV2Tiny` / `ConvNeXtV2Small`**）から `model_name` を比較確定。Keras 標準にない **ConvNeXt V2** は `components/third_party/convnext_tf`（[zibbini](https://github.com/zibbini/convnext-v2_tensorflow) 由来、MIT）の **`convnextv2_tiny`** および、同梱に **`convnextv2_small` が無い**ため **`convnextv2_nano` を `ConvNeXtV2Small` 名で探索**（`components/zibbini_v2_models.py` の `ZIBBINI_V2_BUILDERS`、`weights=None` 想定）。**MobileNetV4** 相当は **MobileNetV3Large** を代用。`optimize_param` に **`head_only_tie_log` を渡す**。
- **Step 1.2: Head LR Re-calibration** — **1.1** で `model_name != LRCALIB_BASE_BACKBONE` のときだけ、確定したバックボーンで head LR を再キャリブ（Step1 は基準バックボーンのみのため）。基準と同じ `model_name` ならスキップ。
- **Step 1.5 / 2 / 3** — `weight_decay` / 構造（layers/units/dropout）/ データ拡張・正則化（shift 含む）を順次最適化（head-only）。**`optimize_param` にはすべて `head_only_tie_log` を渡す**（同点は 3.8）。候補ごとに最高スコアを比較し、**同点が複数**なら候補先頭を仮採用して次軸へ。
- **Step 3.8: 同点解消** — 上記で記録した「同最高スコアの候補群」が存在する各パラメータについて、**以降の軸を確定した `current_params` を引き継ぎ、同点候補同士だけ `run_trial` し直し**採択（`resolve_tie_breaks`）。`width_shift_range` / `height_shift_range` 連動分は特殊キー `_shift_coupled` として扱う。再採点は**記録順**（先に出た同点軸から解消し、`current_params` を更新しながら次へ）。**同点の定義**は `train_sequential._is_same_score` で、**検証スコア差が 0.01 未満**なら同一扱い。
- **Step 3.8 を 3.9 前に置く理由 / unfreeze 後について** — Step 3.9 は「同点解消**後**の hparams」で `best_head.weights.h5` を再学習する。同点解消を unfreeze 確定**後**（FT 条件）だけに回すと、3.9 は仮採択のまま保存し、**後段で hparam（例: dropout）が差し替わる**と head 重みと不整合になる（その場合は **3.9 を掛け直す** or **旧 4.6 相当**を FT 下に別枠で置く、が要る）。現状は **hparam 確定 → 3.9 で head 1 本**の順序を保つ。FT 中の最適点が凍結時の同点と違う可能性はあるが、それを **unfreeze 後だけ**で扱うのは 4.6/追加試行系の責務と分ける。
- **Step 3.9: Best Head Weights 再学習＆保存** — Step 3 / 3.8 で確定した best params 構成で head-only を再学習し、ベスト epoch の重みを `outputs/best_head_weights/best_head.weights.h5` に保存する。FT フェーズで `--init_weights_path` 経由で初期値として読み込む（head carryover）。
- **Step 3.5: FT LR Calibration** — `init_weights_path` に Step 3.9 の保存ファイルを設定。保存に成功している場合は `warmup_lr=0` / `warmup_epochs=0` として warmup フェーズをスキップし、head carryover された初期重みから直接 FT に入る（保存失敗時のみ従来の warmup にフォールバック）。
- **Step 4 / 4.5** — `unfreeze_layers` 最適化 → 暫定 `unfreeze≠60` のとき FT LR 再 Cal。（旧 **Step 4.6** 廃止: 凍結フェーズ＋3.8 に一本化。）
- **Best-of-N（先）** — 複数 **seed** で各 1 回フル FT（LR は Step 4.5 まで）し、**ベスト seed** を採択。直後、**4.7 の探索で `model_seed{seed}.keras` が上書きされる前に** `best_sequential_model.keras` へ **Best-of-N 時点の**重みをコピー（**重み＝4.5 時点 LR、JSON の `learning_rate_ft`＝4.7 採用 LR**と必ずしも一致しない点に注意）。
- **Step 4.7** — `seed` 固定のまま `search_ft_lr_by_targets` で **target epoch 帯 `[min(targets), max(targets)]`（既定 10〜15）に対し `calibrate_base_lr` を 1 回だけ**実行し、**最良 Val の LR** と **採用スコア**（`final_ft_score`）を確定（旧: 10..15 を各 `target_best_epoch=t` で別キャリブし毎回 `initial_lr` に戻していた）。追加の**最終 `run_trial` 1 本は行わない**。Best-of-N スコア（pre-4.7 LR）と 4.7 最良はログで併記。
- **掃除** — `model_seed{42..44}.keras` を削除。

### Head carryover に使う weights ファイル
- **パス**: `outputs/best_head_weights/best_head.weights.h5`
- **書き手**: `components/train_multitask_trial.py` の `BestHeadWeightsSaver` コールバック（head-only 学習中、全タスク平均 `val_min_class_accuracy` のベスト更新時に保存）＋延長学習中にベスト更新があった場合も保存。
- **読み手**: `components/train_multitask_trial.py` 本体。`--init_weights_path` 指定時に `model.load_weights(path, by_name=True, skip_mismatch=True)` で読み込む。
- **追加 CLI 引数**（`components/train_multitask_trial.py`）:
    - `--init_weights_path <path>`: 指定された .weights.h5 をロード（by_name, skip_mismatch）。存在しなければ WARNING で継続。
    - `--save_best_head_weights_path <path>`: head-only 学習中のみ、ベスト重みをこのパスに保存。FT 時（`--fine_tune True`）には無視。

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
