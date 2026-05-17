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
    - **眉-目パーセンタイル**: 個人フォルダ内の相対カットにつながるため、`optimize_svm_sequential.py` の自動探索からは除外（preprocess 呼び出しは high/low を `0` 固定）。本番データ作成で眉-目を使うときは、`preprocess_multitask.py` を直接叩いて指定する。

### 手法B: CNNベース最適化 (高精度・低速)
- **スクリプト**: `optimize_sequential.py`
- **仕組み**: 実際にCNN (`EfficientNetV2`) を数Epoch学習させて評価する。
- **特徴**: SVMよりも最終的なタスクに近いが、1試行に時間がかかる（数分/回）。
- **探索設定**:
    - **探索点**: `[0, 5, 25, 50]`（昇順の粗探索）。**`p>0` の試行で score が連続 2 回 `p=0` の score を下回ったら、それより大きいパーセンタイルは粗探索しない**（早期終了。1 回だけ下回った場合は次点まで試す）。粗探索後、ベースライン超の改善があり **評価済み点が2点以上** あれば、従来どおり Score/Efficiency 各 Top2 間の中間点を 1% 刻みで追う Refinement。
    - **眉-目パーセンタイル**（`--eyebrow_eye_percentile_*`）: **探索対象外**（試行では常に 0）。`preprocess_multitask.py` 単体実行時のみ手動で指定可能。
    - **平均明度**（`--mean_brightness_percentile_low`）: グローバル分布で **暗い順の下位 X%** を除外。Phase 1 で `mean_brightness_low` として他軸と同様に粗探索＋Refinement。
- **Phase 2 greedy（固定閾値統合）**: Phase 1 の各試行後に書かれる `preprocessed_multitask/filter_threshold_manifest.json` の **train** スプリット `global_numeric_thresholds` から、当該軸の **実際に使った実数閾値**を読み取る。greedy では複数軸を足し込むとき、グローバル軸については **パーセンタイル同時指定ではなく** `preprocess_multitask.py` の `--*_threshold`（当該軸の `--*_percentile` は 0）で前処理する。マニフェスト欠損時は警告のうえ **パーセンタイルにフォールバック**。LR スケーリングの加重指数計算では、固定閾値で採用した軸に **名目 50%** 相当を載せる（実装定数 `_FIXED_AXIS_NOMINAL_PERCENTILE_FOR_LR`）。最終出力の `final_params` に `filter_fixed_thresholds` が付く場合は、`run_optimized_preprocess.bat` も `--*_threshold` を含む。
- **Phase 1 結果サマリー**: 全パラメータの独立探索完了後に「精度上昇効率一覧」をログ出力。各パラメータの Best Score 候補と Best Efficiency 候補をテーブル形式で表示し、全体の Overall Best Score / Best Efficiency も出力する。
- **出力アーティファクト**:
    - `outputs/logs/sequential_opt_log_YYYYMMDD_HHMMSS.txt`: 実行ログ（タイムスタンプ付きで履歴保持）。
    - `outputs/optimization_analysis.json`: 最適化プロセスの全候補、Greedy統合の履歴、最終パラメータを含む詳細ログ。
    - `run_optimized_preprocess.bat`: 最適化されたパラメータを適用するための実行バッチファイル。**preprocess 引数は `run_trial` と同一関数 `_preprocess_multitask_argv_tail` で組み立て**（ログ表示・BAT・実際の試行でズレないようにする）。
- **Step 0（モデル＋head LR）**:
    - `MODEL_NAME_CANDIDATES`（`model_architecture.py`）の**各** `model_name` に対し、フィルタなし前処理の上で **`calibrate_base_lr`（`target_best_epoch=13` 他、他軸と同ロジック）**を実行し、キャリブ終了時の Val スコアが最大のモデルを採択。
    - **`calibrate_base_lr` の試行回数上限**は `LR_CALIBRATION_MAX_ITERATIONS`（既定 **10**、`components/lr_adjustment.py`）。`run_trial` 内の LR 再調整は従来どおり `LR_MAX_ADJUSTMENTS`（最大 7 回）で別定数。
    - 採択モデル専用の `CALIBRATED_BASE_LR` を以降の `run_trial`（フィルタ軸最適化）の基準 LR とし、`best_train_params.json` の `model_name` / **`learning_rate`** / **`lr_calib_context`**（head・train+val 枚数・`base_lr`）を更新。Step 0 の各モデルキャリブでは **保存 `lr_calib_context` と（モデル・枚数・head）が一致する候補だけ** 保存 `base_lr` を `initial_lr` に、それ以外は `LR_CALIBRATION_INITIAL`（0.01）。
    - **Step 0 も `filter_opt_cache.json` に記録**（`@calib…`＝各試行 `run_calibration_trial` の結果、`@calfull…`＝モデル単位の `calibrate_base_lr` 完走結果）。データ枚数が変わると既存どおりキャッシュ全消去。`best_train_params` のキャリブ対象ハイパラが変わればダイジェストでミスヒット。アルゴリズム更新時はコード内 `_CALIB_CACHE_VERSION` を上げる。
- **LR自動調整リトライ** (全スクリプト共通: `optimize_sequential.py`, `train_sequential.py`。両者で条件・定数を同一にしている):
    - 各トレーニング実行後に `BEST_EPOCH`（FT 時は `FT_BEST_EPOCH`）を確認し、終了条件を満たさなければ再調整（`LR_MAX_ADJUSTMENTS=6` まで。合計試行は `range(LR_MAX_ADJUSTMENTS+1)` により **最大 7 回**）。
    - **早期終了**（`lr_calibration_should_stop`）: best_epoch が LR_ACCEPTABLE_MIN～LR_ACCEPTABLE_MAX(11～15) **かつ** last_epoch_accu≠best（差≥0.01）→ 以降の LR 探索を打ち切り。
    - **試行回数上限**: `for adj_iter in range(LR_MAX_ADJUSTMENTS+1)` で、最後の試行の直後は次 LR を計算せず終了（最大 7 試行）。
    - **次 LR の決定**（`calibrate_base_lr` と同一の `lr_bisect_update_bounds_and_next_raw`）: `target_min = target_max = LR_TARGET_EPOCH`（既定 13）。`best_epoch` がターゲットより早い試行で `lr_high`、遅い試行で `lr_low` を更新。**両境界が揃えば** 次 LR = `sqrt(lr_low * lr_high)`（ログでは **`[LR sweep] 次 lr 案: 幾何平均`**）。**片側のみ**は `new_lr = current_lr × (best_epoch / target_mid)`（比の上下限なし）。**相対変化判定**: clip 前の `new_lr` と現在 LR との相対差が `LR_CALIB_MIN_RELATIVE_CHANGE`（既定 5%）未満なら打ち止め。**ターゲット方向打ち止め**（`run_trial` の sweep のみ）: 直前 sub-run との比で `best_epoch` が `LR_TARGET_EPOCH` に近づく方向へ **連続 `LR_SWEEP_MAX_CONSECUTIVE_NO_TARGET_PROGRESS` 回（既定 2）動かなければ**終了（`best_epoch_moved_toward_lr_target`）。実際に optimizer に載せる値は `clip_learning_rate_for_training`。
    - 全試行中の **最高検証スコア**の出力（ログ・キャッシュ・staging keras）を採用する。
    - **ログの読み方（`train_sequential` / `optimize_sequential` の `run_trial`）**: ブロック先頭の **`run_trial 文脈:`** が当該 1 本の目的（例: `optimize_param axis='mask' value=37`、Phase2 greedy 等）。LR 再調整付き試行では **`[LR sweep]`** が sub-run ごとの `train_lr`・直後の val/best_epoch・**終了理由**（満足条件 / 試行上限 / 相対変化 / **ターゲット方向に2回連続動かず**）、最後に **まとめ**（採用 Val・最良試行の lr・二分境界）を出す。キャッシュヒット時は学習をスキップするため `[LR sweep]` は出ない。
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

### フィルタ閾値マニフェスト（推論・サービスデプロイ）
- **`preprocess_multitask.py`** は各実行で、パーセンタイルから **実際に判定に使った実数閾値** を JSON に書き出す。既定ファイルは **`--out_dir` 直下の** `filter_threshold_manifest.json`（train / validation / test を 1 ファイルの `splits` にまとめる）。
- 内容は **グローバル閾値**（pitch / symmetry 等）に加え、眉-目距離について **ラベル（フォルダ階層由来の `label`）ごと**の閾値 `per_label_eyebrow_thresholds`。単一画像の合否判定では、サービス側で同じパイプライン計測値を用意し、この JSON の不等号条件に沿って照合する（学習時と異なるソース集合ではパーセンタイルのみでは再現不能なため）。
- **スプリット間の注意**: train / validation / test はそれぞれ **当該スプリットの valid 顔集合** で閾値を再計算する。推論を学習画像と同一基準に揃える場合は通常 **`splits.train`** を参照する。
- **アンダーサンプリング**: JSON の閾値だけでは再現しない（枚数キャップ・シャッフルあり）。マニフェストの `notes` にも記載。
- **`face_size`（`_sz<px>`）** と **`face_roll_deg_abs`（`_rz<ミリ度>`）**: `components/part1_setup.py` が保存時に **バウンディングボックス幅** を `_sz`、**回転補正に使った in-plane 角**（度×1000 の符号付き整数、例 `_rz-3250` = −3.25°）を `_rz` としてファイル名に埋め込む。`preprocess_multitask.py` は `_rz` から絶対度数へ復元する。**`_rz` が無いデータ**（正立化済み切り出しのみ等）は **元画像に対する補正角は復元不能**のため **`face_roll_deg_abs = 0`** とする。**`--rotation_percentile`**（既定0）が正のとき、当該メトリクスで pitch と同様に上位 X％を閾値で落とす。`filter_threshold_manifest.json` の `face_roll_abs_deg_upper_reject_if_strictly_greater` に実数閾値を記録。
- **`components/train_multitask_trial.py`** が FT 済み `.keras` または `--export_model_path` を保存したとき、`preprocessed_multitask/filter_threshold_manifest.json` を **モデルと同じディレクトリ**へ `filter_threshold_manifest.json` としてコピーする（ファイルが無い場合は WARN のみ）。
- **`train_sequential.py`** が **`best_sequential_model.keras`**（最終ステップの成果物）または **`best_sequential_pipeline_best.keras`**（試行横断の Val 最大が更新されたとき）をコピーしたときも同様にマニフェストを付ける。

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
  - **キャリブレーションの打ち切り**: **`lr_calibration_should_stop`（各 trial 直後）**。`target_best_epoch` が **タプル帯 `(lo, hi)`** のときは **`lo≤best_epoch≤hi`**（`cal_epochs` で上限クランプ）かつ **last_epoch_accu≠best**（差≥`LR_LAST_ACCU_EPS`）。**単一ターゲットまたは None** のときは従来どおり **11〜15**（`LR_ACCEPTABLE_*`）。**そのほか** — **`calibrate_base_lr` が `LR_CALIBRATION_MAX_ITERATIONS`（既定10）到達**、または **LR 相対変化 `< LR_CALIB_MIN_RELATIVE_CHANGE`（既定 0.05＝5%）**。run_calibration_trial は last_epoch_accu も返す。
  - **候補採点ルール**（2026-04-22 修正）: 最終選択は `(distance, -score, ...)` タプルで比較する。すなわち `target_best_epoch` からの距離（`abs(best_epoch - target)`）が最小の iteration を優先採用し、距離同率のときのみ score（`MinClassAcc` など）の大きい方を採用する。
     - 旧仕様（`(-score, distance, ...)` = score 最優先）だと、noisy val 環境で target から離れた iteration が score 偶発で採用されるため、target=13 に寄せるループ自体が無意味化していた問題を修正。
     - `train_sequential.py` では `calibrate_base_lr(score_priority=False)` を全呼び出しで使用（デフォルト）。`score_priority=True` の旧挙動は互換のため残存するが実使用しない。
  - **探索方式**（2026-04-22 統一、2026-05-12 帯モード修正）: `optimize_sequential.py` / `train_sequential.py` の `calibrate_base_lr` は **log 空間二分**（両境界あり）と **片側比** で統一。
     - `best_epoch < target_min` → `lr_high = current_lr`（LR高すぎ側の最小試行）。
     - `best_epoch > target_max` → `lr_low = current_lr`（LR低すぎ側の最大試行）。
     - **`target_min ≤ best_epoch ≤ target_max`（帯内含む単一点 target では best_epoch がターゲットと一致する場合）では境界を更新しない**（旧実装では帯内含む試行まで `lr_low` に入り二分が歪む問題があった）。
     - 両境界が揃ったら次 LR = `sqrt(lr_low * lr_high)`（幾何平均 = log 空間の中点）。LR は乗算スケールで効くため算術平均より幾何平均の方が対称で収束が速い。
     - 片側のみのとき: `scale = compute_lr_adjustment_ratio(...)`（`best_epoch/target_mid`）で `new_lr = current_lr * scale`（比のクランプなし）。実装は **`lr_bisect_update_bounds_and_next_raw`** に集約。
     - `LR_CALIBRATION_MAX_ITERATIONS=10`（`lr_adjustment.py`）→ `calibrate_base_lr` は最大 10 trial。`run_trial` 側の LR 再調整は **同じ境界・幾何／ratio ロジック**だが試行上限のみ `LR_MAX_ADJUSTMENTS=6`（最大 7 trial）。
     - 収束判定: `|new_lr - current_lr| / current_lr < LR_CALIB_MIN_RELATIVE_CHANGE`（`lr_adjustment.py`、既定 **0.05**＝5% 未満）なら以降の trial を打ち切り、最良候補を採用。
  - **LR の一本化（2026-04-25）**: `best_train_params.json` に記録する教師あり LR は **`learning_rate` のみ**（終端フェーズで実際に使う値）。旧キー `learning_rate_head` / `learning_rate_ft` / `learning_rate_nohead` は保存時に削除され、読み込み互換のため `_skip_keys` でサブプロセスに転送しないだけ残す。
  - **JSON のみフィールド**: `finish_mode`・`score_step_3_*`・`lr_step_3_5_ft_calib_*`・`lr_calib_context` 等は **記録・復元用のみ**であり、`train_multitask_trial.py` には渡さない（`components/lr_adjustment.py` の **`TRAIN_MULTITASK_CLI_EXCLUDE_KEYS`** を `train_sequential` / `optimize_sequential` が subprocess 構築時に適用）。
  - **`lr_calib_context` と `calibrate_base_lr` の initial_lr（2026-04-25）**: JSON に **`lr_calib_context`**（`model_name`, `data_file_count`, `mode`=`head`|`ft`, `base_lr`）を保存する。**いずれかが変われば**（モデル・データ枚数・head-only vs FT）次回の `calibrate_base_lr` は **`LR_CALIBRATION_INITIAL`（0.01）から**。**三つとも前回保存と一致**すれば（同一ラン内は「直前のキャリブ結果」、別ランはディスクの `lr_calib_context`）**保存 `base_lr` を initial_lr にして再キャリブ**。`train_sequential` のデータ枚数は **`preprocessed_multitask/train` のファイル数**、`optimize_sequential` の Step 0 は **`train`+`validation` のファイル数**（各スクリプトのキャッシュキーと一致）。**`train_sequential.run_trial` の開始 LR も同じ `resolve_calib_initial_lr` で決める**（`params['learning_rate']` だけではモデル切替時に B0 用 LR が残るため）。
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
| **Mean brightness (L)** | `--mean_brightness_percentile_low` | 全画像 `mean(cv2.cvtColor(BGR2GRAY))`（0–255） | 平均輝度の下位から除外（暗い画像を落とす）。sharpness と同じ全画素グレー平均。 |
| **Aspect Ratio** | `--aspect_ratio_cutoff` | `box_height / box_width` | 顔検出枠の縦横比。極端に細長い/潰れた誤検出を排除。 |

### フィルタリングロジック
1.  **パーセンタイル計算**:
    - 全画像（Validなもの）の指標分布を計算。
    - 指定されたパーセンタイル（例: 10%）に基づき、閾値を決定（例: 上位10%をカットする閾値）。
    - **グループ化**: ソースの**ディレクトリパス**（例: `a/森口瑤子`）単位。眉-目距離(Eb-Eye)のパーセンタイルとアンダーサンプリングはこの単位で行うため、個人（1フォルダ＝1人）ごとに「その人の上位○%」が残る。他指標（pitch/symmetry/sharpness/mean_brightness 等）の閾値は従来どおり全体分布で計算。※ファイル名に個人名が付いていなくても、フォルダ構造で個人が分かれていれば個人単位になる。
2.  **判定**:
    - 各画像について、全指標が閾値内であれば「採用」。一つでも超えれば「不採用（スキップ）」。
3.  **アンダーサンプリング**（`preprocess_multitask.py` の詳細は下記）:
    - **第 1 段**: フィルタ直後の残件で **クラス内**（同一クラスに属する全フルラベル＝人物フォルダ）について、各バケツを **そのクラス内で 2 番目に多い枚数**まで間引く（`undersampling_post_filter`）。train/validation/test は各スプリットで同じロジック。
    - **第 2 段**: 第 1 段のあと、**クラス間**で各クラスの合計枚数を **最少クラスに揃える**（`undersampling_class_balance`）。`--skip_class_balance` で無効化。
    - **第 3 段**: 第 2 段のあと（`--skip_class_balance` 時は第 1 段直後）、**再びクラス内2位上限**を適用（落とし枚数も `undersampling_post_filter` に加算）。

### preprocess_multitask.py の詳細

**入力**: `train/`, `validation/`, `test/`（各ディレクトリ直下に `{タスク}/{人物名}/` または `{タスク}/{サブフォルダ}/` のような相対パスで画像が並んでいる想定）。

**処理フロー**:
1. **スキャン**: 各ソースを `os.walk` で走査。画像拡張子 `.jpg/.png/.jpeg/.bmp` のファイルを列挙。各ファイルの「グループキー」= そのファイルがあるディレクトリの相対パス（例: `a/森口瑤子`）。
2. **顔分析（キャッシュあり）**: `outputs/cache/metrics_<hash>.pkl` をキー（ソース名+ファイル数）で参照。キャッシュがなければ InsightFace で全画像を解析し、顔検出・106ランドマークを取得。検出失敗・読込失敗は `valid=False` で除外。
3. **閾値計算**:
   - **グローバル閾値**: 全 valid 画像の指標分布から算出。`pitch_percentile=10` なら「上位10%を落とす」ので `th_pitch = percentile(pitch, 90)`。同様に symmetry, y_diff, mouth_open, sharpness(低/高), mean_brightness(低、暗い順に下位カット), face_size(低/高), aspect_ratio(両側), retouching, mask, glasses を計算。
   - **個人閾値**: グループ（ディレクトリパス）ごとに **眉-目距離 (eb_eye_dist)** だけ、そのグループ内の分布で `eyebrow_eye_percentile_low` / `eyebrow_eye_percentile_high` の閾値を計算。
4. **判定**: 各画像について、上記の全閾値と比較。**一つでも閾値を超えたらスキップ**（採用されない）。スキップ理由は `pitch_global`, `symmetry_global`, `mean_brightness_low_global`, `eb_eye_low_personal`, `undersampling_post_filter`, `undersampling_class_balance` などでログに集計される。
5. **アンダーサンプリング（三段）**:
    - **第 1 段（クラス内・フォルダバケツ）**: フィルタ直後の残件で、クラスキー（相対パス先頭セグメント）ごとにフルラベル（フォルダ単位）別の枚数を集め、**そのクラス内で 2 番目に多いバケツの枚数**を上限として各バケツを切り詰める（`skip_reasons['undersampling_post_filter']`）。
    - **第 2 段（クラス間）**: 第 1 段の**あと**、クラスキーごとの**合計枚数**を出し、**最少クラスと同じ合計**になるよう多いクラスからランダムに削る（`skip_reasons['undersampling_class_balance']`）。正の枚数のクラスが2つ以上のときのみ。`--skip_class_balance` で無効化。
    - **第 3 段（クラス内・再適用）**: 第 2 段の**あと**に第 1 段と同型のクラス内2位上限を再度適用（`undersampling_post_filter` に加算）。`skip_class_balance` のときは第 1 段直後に実行（第 2 段を挟まないため多くの場合ほぼ無変化）。
    - **`skip_undersampling`**: True のとき第 1〜第 3 段とも行わない。train/validation/test は同じ実装（通常 False）。
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
| Mean brightness | 全画像グレースケールの画素平均（0–255） | 値 **<** th_low → 除外（`mean_brightness_percentile_low`：暗い順の下位%） |
| Face Size | ファイル名の `_sz(\d+)` から取得（0 の場合は対象外） | 値 **<** th_low または **>** th_high → 除外 |
| Aspect Ratio | 顔検出枠の `height/width` | 値 **<** th_low または **>** th_high → 除外 |
| Retouching | 肌領域の Sobel 高周波成分の平均（低いほど加工疑い） | 値 **<** 閾値 → 除外 |
| Mask | 上顔/下顔の肌色比率から算出（高いほどマスク疑い） | 値 **>** 閾値 → 除外 |
| Glasses | 目周辺エッジと額エッジの比（高いほど眼鏡疑い） | 値 **>** 閾値 → 除外 |

**引数（0＝フィルタ無効）**: `--pitch_percentile`, `--symmetry_percentile`, `--y_diff_percentile`, `--mouth_open_percentile`, `--eyebrow_eye_percentile_low` / `--eyebrow_eye_percentile_high`, `--sharpness_percentile_low` / `--sharpness_percentile_high`, `--mean_brightness_percentile_low`, `--face_size_percentile_low` / `--face_size_percentile_high`, `--aspect_ratio_cutoff`, `--retouching_percentile`, `--mask_percentile`, `--glasses_percentile`, `--grayscale`, `--skip_class_balance`（**中段**のクラス間均衡のみスキップ。第1・第3段のクラス内2位は実行）。  
**出力**: `preprocessed_multitask/train/`, `preprocessed_multitask/validation/`, `preprocessed_multitask/test/`。これが `train_sequential.py` の直接の入力。

---

## 3. 本番学習パイプライン (`train_sequential.py`)

前処理済みデータを用いて、最終的なマルチタスクモデルを学習します。

- **モデルファイル（2 系統）**:
    - **`outputs/models/best_sequential_pipeline_best.keras`**: `run_trial`（逐次最適化の各試行・Best-of-N の各 seed・Step 4.7 のフル FT など **いずれかの試行**）および Step 3.9 の head 保存で、**FINAL_VAL_ACCURACY がそのラン内でこれまで観測した最大を更新したとき**に **`_promote_best_sequential_model`** によりコピーする。**試行ごとにハイパラが異なり得る**（探索中に出た「とにかく数字が良かった」重み）。
    - **`outputs/models/best_sequential_model.keras`**: **`best_train_params.json` と整合する最終成果物**。Best-of-N の勝ち seed の `model_seed{best_seed}.keras` をコピーし、FT ルートでは Step 4.7 のフル FT が BoN を上回ったときのみ再更新。**推論・`analyze_errors.py` の既定パスはこちら**（最後に確定した設定に対応したモデル）。
    - 短い LR キャリブ用 `run_calibration_trial` は従来どおり上記いずれの「本番コピー」対象外（モデル未エクスポート）。

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
    - **絶対域（2026-04-26）**: オプティマイザに入る各エポックの LR は `components/lr_adjustment.py` の **`LR_TRAIN_ABSOLUTE_MIN=1e-7`** ～ **`LR_TRAIN_ABSOLUTE_MAX=0.1`** に `clip_learning_rate_for_training` で収める。極小は `.8f` ログで 0 に見えて実質停止するのを防ぎ、極大は設定外れの保険。学習率確定直後（head/FT いずれも）・`ConditionalLearningRateScheduler` / `LinearWarmupScheduler` 各エポックに適用。
    - **import 注意（2026-04-26 修正）**: `train_multitask_trial.py` は `components/` 配下でサブプロセス起動されるため、同ディレクトリの `lr_adjustment` は **`from lr_adjustment import ...`** とする。`from components.lr_adjustment ...` だと実行時 `sys.path` によっては **解決失敗**し、キャリブの `FINAL_VAL_ACCURACY` 抽出不能（score=0.0）になる。
- **条件付きEpoch拡張 (Conditional Extension)**（2026-05-03 廃止）: 旧実装では最終 epoch がベストのとき追加の 1 epoch 学習を繰り返していたが、**本リポジトリでは削除済み**（LR 再調整・`calibrate_base_lr` で代替）。
- **Weight Decay**: Dense層の `kernel_regularizer=l2(wd)` で実装 (AdamW不要)
- **Mixup**: Beta(α, α) 分布からサンプリング (Gamma分布2つから構築)
- **検証データ**: データ拡張・Mixup は適用しない。**`CategoricalCrossentropy` を使う設定**（`mixup_alpha>0` または `label_smoothing>0`）では、損失と整合するためラベルのみ **one-hot** に変換する（学習側と同じクラス次元）。

### `outputs/best_train_params.json` の学習率の扱い
- `train_sequential.py` の最終採用ハイパラを保存する。
- **`learning_rate` のみ**を保存する（パイプライン終了時点で採用した LR）。
- **`lr_calib_context`**: 上記 LR をキャリブしたときの **`model_name` / `data_file_count` / `mode`（`head`|`ft`）/ `base_lr`**。次回 `calibrate_base_lr` の `initial_lr` はこの三つが一致するとき保存 `base_lr`、変化時は `LR_CALIBRATION_INITIAL`（0.01）。
- **旧キー**（`learning_rate_head` / `learning_rate_ft` / `learning_rate_nohead`）は今後書き込まず、読み込んだ古いファイルに残っていても `_skip_keys` で転送のみ除外する。
- **`train_multitask_trial.py` へ転送しないメタ情報キー**:
    - 上記旧キーは `train_sequential.py` / `optimize_sequential.py` 側のメタ情報であり、`components/train_multitask_trial.py` の argparse には存在しない。
    - そのため `train_sequential.py` の `run_trial` / `run_calibration_trial`、および `optimize_sequential.py` の `run_trial` / `run_calibration_trial` でサブプロセス起動コマンドを組み立てる際、これらのキーは `_skip_keys` で除外する。
    - 実際に `train_multitask_trial.py` に渡すLRは、いずれの呼び出し側でも明示的に `--learning_rate <value>` として付与する。

### `train_sequential.py` の主要ステップ
- **このスクリプトの `run_trial` ログ**: 試行開始直後に **`run_trial 文脈:`**（例: `optimize_param axis='model_name' value='ResNet101V2'`）。LR 自動調整が動くと **`[LR sweep]`** で sub-run・終了理由・**まとめ**が出る。**条件・ログ形式は `optimize_sequential.py` と同一**で、上記 §1 手法B の **LR自動調整リトライ** → **ログの読み方**に準ずる。**戻り値**は **`(val_score, memorized_best_lr)`**（キャッシュは `[score, lr]` 形式。旧キャッシュはスコアのみ）。
- **`model_name` の扱い（2026-04-28 更新）**: `outputs/best_train_params.json` に `model_name` があっても `train_sequential` は **Step 1.1（バックボーン比較）をスキップしない**（**現在の前処理データ**で毎回確定）。**`optimize_sequential.py` Step 0** は `MODEL_NAME_CANDIDATES` ごとに **LR キャリブ**してから最高スコアの `best_model` を選び、同 JSON に `model_name` / **`learning_rate`** を書き戻す。`train_sequential` では 1.1 を再確定する想定。
- **Step 1.1: Model Architecture** — 毎回: `MODEL_NAME_CANDIDATES`（`model_architecture.py`、例: EfficientNet B0/S、ResNet50/101V2、MobileNetV3Large）から `model_name` を比較確定。**MobileNetV4** 相当は **MobileNetV3Large** を代用。`optimize_param` に **`head_only_tie_log` を渡す**。各候補の `run_trial` は `(model, data_file_count, head|ft)` に応じた開始 LR を `resolve_calib_initial_lr` で決め、**LR sweep 内で最良 Val だった `train_lr` をその候補の記憶 LR**として保持する。採択モデルについてはその **記憶 LR（`clip_learning_rate_for_training` 後）を `learning_rate` / `lr_calib_context` の head `base_lr` に採用**する。旧 **Step 1.2（採択後の `calibrate_base_lr`）は廃止**（1.1 との LR 探索重複を避ける）。**キャッシュ**は `train_opt_cache` の各キーについて `[score, memorized_lr]` 形式（従来の数値のみの値は旧形式として解釈し、記憶 LR 欠損時は `initial_lr_for_calibrate` にフォールバック）。
- **Step 1.25: Optimizer** — **`model_name` と head LR（1.1 記憶値）確定後**に、`optimizer` を `optimize_param(['adam','adamw','sgd'])` で head-only 比較。**アーキテクチャや LR が未確定の段階で先に決めることはしない**（同じ LR でも Adam/SGD で最適域が異なるため、キャリブはモデル単位でもオプティマイザ単位での再キャリブはコスト過大になりがち）。
- **`weight_decay` と正則化の役割分担**（`components/train_multitask_trial.py`）:
  - **`adamw`かつ環境が AdamW に対応**し **`weight_decay > 0`**: アダプティブ用の **`AdamW(weight_decay=...)`** のみ。**Dense の kernel L2 は付けず**ダブりを避ける。
  - **`adam` / `sgd`、または前述 AdamW 未対応のフォールバック**: **`weight_decay > 0` は Dense 層の `kernel_regularizer=l2(weight_decay)`**（従来と同様）。AdamW フォールバック時も同様に L2 で近似。
- **Step 1.5 / 2 / 3** — `weight_decay` / 構造（layers/units/dropout）/ データ拡張・正則化（shift 含む）を順次最適化（head-only）。**`optimize_param` にはすべて `head_only_tie_log` を渡す**（同点は 3.8）。候補ごとに最高スコアを比較し、**同点が複数**なら候補先頭を仮採用して次軸へ。
- **Step 3.8: 同点解消** — 上記で記録した「同最高スコアの候補群」が存在する各パラメータについて、**以降の軸を確定した `current_params` を引き継ぎ、同点候補同士だけ `run_trial` し直し**採択（`resolve_tie_breaks`）。`width_shift_range` / `height_shift_range` 連動分は特殊キー `_shift_coupled` として扱う。再採点は**記録順**（先に出た同点軸から解消し、`current_params` を更新しながら次へ）。**`model_name` 同点**の解消後は、勝者候補の **`run_trial` 記憶 LR** で `learning_rate` を更新する。**同点の定義**は `train_sequential._is_same_score` で、**検証スコア差が 0.01 未満**なら同一扱い。
- **Step 3.8 を 3.9 前に置く理由 / unfreeze 後について** — Step 3.9 は「同点解消**後**の hparams」で `best_head.weights.h5` を再学習する。同点解消を unfreeze 確定**後**（FT 条件）だけに回すと、3.9 は仮採択のまま保存し、**後段で hparam（例: dropout）が差し替わる**と head 重みと不整合になる（その場合は **3.9 を掛け直す** or **旧 4.6 相当**を FT 下に別枠で置く、が要る）。現状は **hparam 確定 → 3.9 で head 1 本**の順序を保つ。FT 中の最適点が凍結時の同点と違う可能性はあるが、それを **unfreeze 後だけ**で扱うのは 4.6/追加試行系の責務と分ける。
- **Step 3.9: Best Head Weights 再学習＆保存** — Step 3 / 3.8 で確定した best params 構成で head-only を再学習し、ベスト epoch の重みを `outputs/best_head_weights/best_head.weights.h5` に保存する。あわせて同一ベスト重みをロードした **完全モデル**を `outputs/best_head_weights/best_head_only.keras` に別保存する（推論・検証用。FT の head carryover は従来どおり `.weights.h5` のみ使用）。FT フェーズで `--init_weights_path` 経由で初期値として読み込む（head carryover）。
- **Step 3.5: FT LR Calibration** — **常に 2 系統**で `calibrate_base_lr`（同一 `initial_lr`）を実行する: **(A)** Step 3.9 の `best_head.weights.h5` を `init_weights_path` に載せ `warmup_lr=0` / `warmup_epochs=0`（ファイルが無いときは (A) をスキップ）; **(B)** `init_weights_path` 空・`warmup_lr=head_lr`・`warmup_epochs=5`。**(A)(B) の採用 `FINAL_VAL_ACCURACY` が高い方**の calibrated LR と init/warmup 設定を以降の FT に採用（同点 `_is_same_score` は carryover 側を優先）。終了時の **勝者スコア**を **`score_step_3_5_ft_calib`** とし、head-only フェーズベストとの比較（`head_finish`）に使う。詳細スコア・LR は `best_train_params.json` の `score_step_3_5_ft_calib_carry` / `score_step_3_5_ft_calib_warmup`・`ft_calib_carryover_selected` 等を参照。
- **分岐: FT キャリブが head-only フェーズベスト未満のとき** — **Step 4 / 4.5 / Step 4.7 をスキップ**。`fine_tune=False`・`learning_rate=head_lr`・`init_weights_path=best_head.weights.h5`（無ければ WARNING）で **Best-of-N のみ**（各 run に `--export_model_path=outputs/models/model_seed{seed}.keras`）。採用 seed のモデルを **`best_sequential_model.keras`**（最終成果物）にコピー。**パイプライン横断で高かった試行**は引き続き **`best_sequential_pipeline_best.keras`** に蓄積される。`outputs/best_train_params.json` に `finish_mode=head_only`、`score_step_3_9_head` / **`score_step_3_5_ft_calib`**（Step 3.5 勝者） / `score_head_only_phase_best` / **`ft_calib_carryover_selected`** / `score_step_3_5_ft_calib_carry` / `score_step_3_5_ft_calib_warmup` 等を記録。
- **Step 4 / 4.5** — **上記分岐で head-only 仕上げの場合はスキップ。** それ以外（**FT キャリブが head-only フェーズベスト以上**、すなわち `score_3_5 >= score_head_only_phase_best`）では、従来どおり `unfreeze_layers` 最適化 → 暫定 `unfreeze≠60` のとき FT LR 再 Cal。（旧 **Step 4.6** 廃止: 凍結フェーズ＋3.8 に一本化。）
- **Best-of-N（先）** — **head-only 仕上げ**のときは複数 seed で **head-only**（上記）。**FT 継続**のときは複数 **seed** で各 1 回フル FT（LR は Step 4.5 まで）し、**ベスト seed** を採択。その時点の `model_seed{best_seed}.keras` を **`best_sequential_model.keras`（最終成果物）に仮コピー**（Step 4.7 より前）。
- **Step 4.7** — **head-only 仕上げのときはスキップ。** **FT 継続**のとき、`seed` 固定のまま `search_ft_lr_by_targets` が **ターゲット 12 / 13 / 14 の 3 回**だけ **`calibrate_base_lr(..., target_best_epoch=t)` を同一 Step 内で連続実行**。**一気通貫**: 最初だけ Step 4.7 用に解決した `initial_lr` から入り、**2 本目以降は直前ターゲットのキャリブ採用 LR を次の `initial_lr` に載せ替える**。この **3 本の検証スコア**を比べ **最高のものを採用**（`_is_same_score` 同点群内では、|`chosen_best_epoch`−t| が小さい方、さらに同率では **t が大きい方**）。**`final_ft_score` > Best-of-N** のときだけ `final_lr` でフル FT（`FINAL_EPOCHS`・同 seed）の `run_trial` を 1 本走らせ `best_sequential_model.keras` を更新。**`log_result` の終端スコアはそのフル run**（キャリブ値より僅かに異なり得る）。**それ以外**は Best-of-N のコピーを維持。**実行直前に `outputs/cache/train_opt_cache.json` を削除**。
- **掃除** — `model_seed{42..44}.keras` を削除（4.7 export 実行後も含め、ループ末尾で削除）。

### Head carryover に使う weights ファイル
- **パス（重み・FT 初期値）**: `outputs/best_head_weights/best_head.weights.h5`
- **パス（head-only 完全モデル）**: `outputs/best_head_weights/best_head_only.keras`
- **書き手**: `components/train_multitask_trial.py` の `BestHeadWeightsSaver` コールバック（head-only 学習中、全タスク平均 `val_min_class_accuracy` のベスト更新時に保存）。
- **読み手**: `components/train_multitask_trial.py` 本体。`--init_weights_path` 指定時に `model.load_weights(path, by_name=True, skip_mismatch=True)` で読み込む。
- **追加 CLI 引数**（`components/train_multitask_trial.py`）:
    - `--init_weights_path <path>`: 指定された .weights.h5 をロード（by_name, skip_mismatch）。存在しなければ WARNING で継続。
    - `--save_best_head_weights_path <path>`: head-only 学習中のみ、ベスト重みをこのパスに保存。FT 時（`--fine_tune True`）には無視。
    - `--save_best_head_model_path <path>`: head-only 学習**終了後**、ベスト重みをロードしたうえで完全モデル（`.keras`）をこのパスに保存。FT 時には無視。
    - `--export_model_path <path>`: head-only でも **完全モデル**を指定パスへ保存（`train_sequential` の head-only 仕上げ Best-of-N で使用）。FT 時は従来どおり `fine_tune` 側の `model_seed{seed}.keras` 保存が優先され、本引数は通常未使用。

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

### エラー分析 (`util/analyze_errors.py`)
- クラスフォルダ配下は **再帰的に**画像を収集する（例: `train/a/橋本環奈/person_clusters/person_1/*.jpg`）。
- **人物別精度**（コンソール `Per-Person Accuracy` / `report.json` の `per_person_accuracy`）は、各画像について **クラス名 / クラス直下の最初のパス要素**をキーに集計する（上例では `a/橋本環奈`。`person_clusters` 以下は同一人物にまとまる）。クラス直下にだけ画像がある場合は `a/__class_root__`。

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
