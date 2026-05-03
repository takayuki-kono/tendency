import shutil
import subprocess
import sys
import re
import os
import logging
import json
import hashlib
import math
import winsound
import time

# --- 設定 ---
PYTHON_EXEC = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
DATA_SOURCE_DIR = "preprocessed_multitask/train" # ファイル数カウント用
SINGLE_TASK_MODE = True # フォルダ名＝クラス名の場合はTrue, ファイル名パースの場合はFalse

# 出力ディレクトリ
LOG_DIR = "outputs/logs"
CACHE_DIR = "outputs/cache"
MODEL_DIR = "outputs/models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "train_opt_cache.json")
BEST_PARAMS_FILE = "outputs/best_train_params.json"
# Phase 0（head-only）のベスト重みを保存するパス。FT 側の head carryover で初期値として使う。
BEST_HEAD_WEIGHTS_DIR = "outputs/best_head_weights"
BEST_HEAD_WEIGHTS_PATH = os.path.join(BEST_HEAD_WEIGHTS_DIR, "best_head.weights.h5")
# Step 3.9: head-only の完全モデル（FT 用 .h5 とは別ファイル。推論・検証用）
BEST_HEAD_MODEL_PATH = os.path.join(BEST_HEAD_WEIGHTS_DIR, "best_head_only.keras")
os.makedirs(BEST_HEAD_WEIGHTS_DIR, exist_ok=True)

if not os.path.exists(PYTHON_EXEC): PYTHON_EXEC = "python"

# ログ設定
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_filename = f"sequential_train_opt_log_{timestamp}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from components.lr_adjustment import (
    LR_TARGET_EPOCH, LR_ACCEPTABLE_MIN, LR_ACCEPTABLE_MAX,
    LR_MAX_ADJUSTMENTS, LR_CALIBRATION_MAX_ITERATIONS,
    LR_CALIB_CONTEXT_JSON_KEY,
    LR_LAST_ACCU_EPS,
    compute_lr_adjustment_ratio, lr_adjustment_decision, lr_calibration_should_stop,
    lr_calib_mode_from_fine_tune,
    parse_lr_calib_context,
    resolve_calib_initial_lr,
    make_lr_calib_context,
)
from components.model_architecture import LRCALIB_BASE_BACKBONE, MODEL_NAME_CANDIDATES

# main() がセットする。run_trial は (model_name, data_file_count, head|ft) に整合した開始 LR をここから解決する。
_LR_CALIB_STATE: dict = {'persisted_ctx': None, 'last_ctx': None}

# FT 以前の head-only 試行（run_trial / head キャリブ / Step3.9）で観測した FINAL_VAL の最大値
_HEAD_ONLY_PHASE_BEST: dict = {'score': -1.0}


def _reset_head_only_phase_best() -> None:
    _HEAD_ONLY_PHASE_BEST['score'] = -1.0


def _maybe_record_head_only_phase_best(params: dict, score: float) -> None:
    """fine_tune=False のときだけ score を FT 以前フェーズのベスト候補として記録する。"""
    if lr_calib_mode_from_fine_tune(params.get('fine_tune', 'False')) != 'head':
        return
    try:
        s = float(score)
    except (TypeError, ValueError):
        return
    if s > _HEAD_ONLY_PHASE_BEST['score']:
        _HEAD_ONLY_PHASE_BEST['score'] = s


def _log_subprocess_line(line: str, tag: str = "[Train]") -> None:
    """子プロセスの 1 行を親ログへ。学習行に加え、重み取得の Downloading / 例外・エラー行も通す。"""
    if any(
        kw in line
        for kw in (
            "Epoch ",
            "FINAL_VAL_ACCURACY",
            "TASK_",
            "MinClassAcc",
            "Avg=",
            "BEST_EPOCH",
            "FT_BEST_EPOCH",
        )
    ):
        logger.info(f"  {tag} {line}")
        return
    if any(
        k in line
        for k in (
            "Traceback",
            "Error:",
            "Exception:",
            "ResourceExhausted",
            "OutOfMemoryError",
            "Downloading data from",
            "Killed",
            "Aborted",
            "ModuleNotFoundError",
            "ImportError:",
            "OOM when",
        )
    ):
        logger.info(f"  {tag} {line}")


def _log_subprocess_fail(tag: str, returncode: int, lines: list) -> None:
    blob = "\n".join(lines)
    tail = blob[-8000:] if len(blob) > 8000 else blob
    logger.error(
        f"{tag} 異常終了 (returncode={returncode})。子プロセス出力末尾(最大8000字):\n{tail}"
    )


def count_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        count += len(files)
    return count


def train_sequential_data_file_count() -> int:
    """LR キャリブ文脈・キャッシュキーと同じ: preprocessed_multitask/train のファイル数。"""
    return count_files(DATA_SOURCE_DIR)


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

def run_trial(params):
    """
    指定されたパラメータで学習を実行し、検証精度を返す
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating Params: {params}")
    
    # キャッシュ確認
    # params辞書をソートして文字列化（キー順依存防止）
    params_str = json.dumps(params, sort_keys=True)
    file_count = count_files(DATA_SOURCE_DIR)
    
    # ハッシュ化してキーにする（長いので）
    key_src = f"{params_str}_count={file_count}"
    cache_key = hashlib.md5(key_src.encode('utf-8')).hexdigest()
    
    cache = load_cache()
    if cache_key in cache:
        sc = cache[cache_key]
        _maybe_record_head_only_phase_best(params, sc)
        logger.info(f"Cache Hit! Skipping execution. Val Accuracy: {sc}")
        logger.info(f"Original Params: {params}")
        logger.info(f"{'='*50}")
        return sc

    logger.info(f"Cache Miss. Running process... (File Count: {file_count})")
    logger.info(f"{'='*50}")

    try:
        # 開始 LR: params['learning_rate'] は「直前に確定したモデル用」が残りやすい（例: Step1.1 で model だけ差し替え）。
        # model / データ枚数 / head|ft が last または persisted と一致するときだけ base_lr を引き継ぎ、異なれば resolve が 0.01 側へ寄せる。
        mn = params.get('model_name')
        mode = lr_calib_mode_from_fine_tune(params.get('fine_tune', 'False'))
        n_fc = train_sequential_data_file_count()
        start_lr, lr_resolve_note = resolve_calib_initial_lr(
            mn, n_fc, mode,
            last_ctx=_LR_CALIB_STATE.get('last_ctx'),
            persisted_ctx=_LR_CALIB_STATE.get('persisted_ctx'),
        )
        logger.info(f"  run_trial start LR: {lr_resolve_note}")
        logger.info(
            f"  → using LR={start_lr:.8g} (model={mn}, data_file_count={n_fc}, mode={mode}); "
            f"params['learning_rate']={params.get('learning_rate')} は参照しない"
        )
        current_lr = start_lr
        best_trial_score = -1.0
        best_trial_output = None
        training_epochs = int(params.get('epochs', 20))
        
        for adj_iter in range(LR_MAX_ADJUSTMENTS + 1):
            trial_params = params.copy()
            trial_params['learning_rate'] = current_lr
            
            if adj_iter > 0:
                logger.info(f"  [LR Adjust #{adj_iter}] LR={current_lr:.8f}...")
            
            cmd = [PYTHON_EXEC, "components/train_multitask_trial.py"]
            # learning_rate_nohead/head/ft は train_sequential 側のメタ情報で train_multitask_trial には存在しない引数
            _skip_keys = {'learning_rate_nohead', 'learning_rate_head', 'learning_rate_ft'}
            for key, value in trial_params.items():
                if key in _skip_keys:
                    continue
                cmd.extend([f"--{key}", str(value)])
            cmd.extend(["--single_task_mode", str(SINGLE_TASK_MODE)])

            # Popenでリアルタイム出力
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace'
            )
            
            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                _log_subprocess_line(line)
            
            process.wait()
            
            if process.returncode != 0:
                _log_subprocess_fail("train_multitask_trial", process.returncode, output_lines)
                return 0.0

            full_output = "\n".join(output_lines)
            
            # スコア抽出
            match_score = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
            trial_score = float(match_score.group(1)) if match_score else 0.0
            
            # BEST_EPOCH抽出
            is_ft = str(trial_params.get('fine_tune', 'False')).lower() == 'true'
            if is_ft:
                match_epoch = re.search(r"FT_BEST_EPOCH:\s*(\d+)", full_output)
            else:
                match_epoch = re.search(r"BEST_EPOCH:\s*(\d+)", full_output)
            best_epoch = int(match_epoch.group(1)) if match_epoch else training_epochs
            
            # 各エポックのMinClassAccスコアを抽出してlast epoch accuを取得
            epoch_scores = re.findall(r"MinClassAcc=([\d.]+)", full_output)
            last_epoch_accu = float(epoch_scores[-1]) if epoch_scores else 0.0
            
            # ベストスコア更新
            if trial_score > best_trial_score:
                best_trial_score = trial_score
                best_trial_output = full_output
            
            logger.info(f"  BestEpoch={best_epoch}/{training_epochs}, Score={trial_score:.4f}, LastEpochAccu={last_epoch_accu:.4f}")
            should_exit, log_msg, need_adjust, effective_epoch = lr_adjustment_decision(
                best_epoch, last_epoch_accu, trial_score, training_epochs
            )
            if should_exit and log_msg:
                logger.info(log_msg)
                break
            if need_adjust and adj_iter < LR_MAX_ADJUSTMENTS and effective_epoch is not None:
                ratio = compute_lr_adjustment_ratio(effective_epoch, target_epoch=LR_TARGET_EPOCH, total_epochs=training_epochs)
                current_lr *= ratio
                logger.info(f"  [LR Adjust] effective_epoch={effective_epoch} -> ratio={ratio:.4f} -> LR={current_lr:.8f}")
            else:
                break
        
        # ベスト結果を使用
        full_output = best_trial_output if best_trial_output else "\n".join(output_lines)
        score = best_trial_score

        # スコア抽出 (Task A)
        match_a = re.search(r"FINAL_VAL_ACCURACY:\s*(\d+\.\d+)", full_output)
        
        # 他のタスクも抽出してログに出す
        for char_code in range(ord('A'), ord('Z') + 1):
            task_label = chr(char_code)
            match_task = re.search(rf"TASK_{task_label}_ACCURACY:\s*(\d+\.\d+)", full_output)
            if match_task:
                logger.info(f"Task {task_label} Accuracy: {float(match_task.group(1))}")

        # 詳細なクラス別精度をログに転記
        details_match = re.search(r"--- Task [A-Z] Details.*", full_output, re.DOTALL)
        if details_match:
            logger.info("\n[Detailed Class Accuracy]")
            logger.info(details_match.group(0).strip())

        if match_a:
            score = max(score, float(match_a.group(1)))
            logger.info(f"Result: Val Accuracy = {score}")
            
            # キャッシュ保存
            cache = load_cache()
            cache[cache_key] = score
            save_cache(cache)

            _maybe_record_head_only_phase_best(params, score)
            return score
        else:
            logger.error("Score not found in output.")
            return 0.0

    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return 0.0


def _is_same_score(a: float, b: float, eps: float = 0.01) -> bool:
    """検証スコア同士: 差が eps 未満なら同点扱い（揺れを1本化）。"""
    return abs(float(a) - float(b)) < eps


def resolve_tie_breaks(current_params: dict, tie_log: list) -> dict:
    """
    逐次最適化で同最高スコアが複数あったパラメータを、
    他パラが確定した current_params を引き継ぎ、同候補同士だけ再 run_trial して決着する。

    tie_log: (target_name, [同点候補…]) または ('_shift_coupled', [w/h 共通候補…])。記録順に処理。
    """
    if not tie_log:
        return current_params
    p = current_params
    for item in tie_log:
        if not item or len(item) < 2:
            continue
        if item[0] == "_shift_coupled":
            _, tied = item
            if len(tied) <= 1:
                continue
            label = "shift (width / height 同値)"
        else:
            target_name, tied = item[0], item[1]
            if len(tied) <= 1:
                continue
            label = target_name
        logger.info(f"\n>>> Tie-break: {label} (candidates: {tied}) <<<")
        sub = {}
        for v in tied:
            trial = p.copy()
            if item[0] == "_shift_coupled":
                trial["width_shift_range"] = v
                trial["height_shift_range"] = v
            else:
                trial[item[0]] = v
            sub[v] = run_trial(trial)
        m = max(sub.values()) if sub else -1.0
        at_max = [v for v in tied if _is_same_score(sub.get(v, -1.0), m)]
        best_v = at_max[0] if at_max else tied[0]
        if item[0] == "_shift_coupled":
            p["width_shift_range"] = best_v
            p["height_shift_range"] = best_v
        else:
            p[item[0]] = best_v
        logger.info(f"Tie-break winner: {label} = {best_v!r} (Score: {sub[best_v]:.4f})")
    return p


def optimize_param(target_name, candidates, current_params, tie_log=None):
    """
    1 つのパラメータを候補の中から最適化。同最高スコアが複数なら仮に候補先頭を採用し、
    tie_log へ登録。Step 3 完了後の resolve_tie_breaks で全パラ確定後に再採点。
    """
    logger.info(f"\n>>> Optimizing {target_name} (Candidates: {candidates}) <<<")
    
    scores = {}
    for val in candidates:
        params = current_params.copy()
        params[target_name] = val
        score = run_trial(params)
        scores[val] = score

    m = max(scores.values()) if scores else -1.0
    tied = [c for c in candidates if _is_same_score(scores[c], m)]
    best_val = tied[0]
    if tie_log is not None and len(tied) > 1:
        tie_log.append((target_name, list(tied)))

    msg = f"Finished optimizing {target_name}. Provisional best: {best_val} (Score: {m:.4f})"
    if len(tied) > 1:
        msg += f" [TIED: {tied} → re-eval in resolve_tie_breaks]"
    logger.info(msg)
    return best_val, m

def train_and_save_best_head_weights(current_params, weights_path):
    """
    現在の params（head-only 想定）で head 学習を1回走らせ、
    ベスト重みを weights_path に保存する（キャッシュ無視・必ず実行）。
    Returns:
        float: 学習結果の val 精度（失敗時 0.0）
    """
    params = current_params.copy()
    # head-only 保証
    params['fine_tune'] = 'False'
    # head-only の本学習は Step1 と同じ 20 epochs に揃える
    params.setdefault('epochs', 20)
    # 念のため既存重み load は無効化（head-only の再学習時に自分自身を上書きしないため）
    params['init_weights_path'] = ''
    params['save_best_head_weights_path'] = weights_path
    params['save_best_head_model_path'] = BEST_HEAD_MODEL_PATH

    logger.info(f"\n{'='*50}")
    logger.info(f"Saving Best Head Weights (epochs={params['epochs']})")
    logger.info(f"  learning_rate = {params.get('learning_rate'):.8f}")
    logger.info(f"  weights_path  = {weights_path}")
    logger.info(f"  model_path    = {BEST_HEAD_MODEL_PATH}")
    logger.info(f"{'='*50}")

    cmd = [PYTHON_EXEC, "components/train_multitask_trial.py"]
    _skip_keys = {'learning_rate_nohead', 'learning_rate_head', 'learning_rate_ft'}
    for key, value in params.items():
        if key in _skip_keys:
            continue
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--single_task_mode", str(SINGLE_TASK_MODE)])

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if any(kw in line for kw in ['Epoch ', 'FINAL_VAL_ACCURACY', 'BEST_EPOCH', 'MinClassAcc', 'Avg=', 'BestHeadWeightsSaver']):
            logger.info(f"  [HeadSave] {line}")
    process.wait()

    if process.returncode != 0:
        logger.error(f"train_and_save_best_head_weights failed (returncode={process.returncode})")
        return 0.0

    full_output = "\n".join(output_lines)
    match_score = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
    score = float(match_score.group(1)) if match_score else 0.0

    if os.path.exists(weights_path):
        logger.info(f"Best head weights saved: {weights_path} (score={score:.4f})")
    else:
        logger.warning(f"Best head weights file was NOT created: {weights_path}")
    if os.path.exists(BEST_HEAD_MODEL_PATH):
        logger.info(f"Best head-only model saved: {BEST_HEAD_MODEL_PATH}")
    else:
        logger.warning(f"Best head-only model was NOT created: {BEST_HEAD_MODEL_PATH}")
    return score


def run_calibration_trial(current_params, lr, cal_epochs=5):
    """
    LRキャリブレーション用: 指定パラメータで学習を実行し、BEST_EPOCHを返す。
    ※キャッシュなし（毎回実行）
    
    Returns:
        tuple: (best_epoch, score)
    """
    params = current_params.copy()
    params['learning_rate'] = lr
    params['epochs'] = cal_epochs

    cmd = [PYTHON_EXEC, "components/train_multitask_trial.py"]
    # learning_rate_nohead/head/ft は train_sequential 側のメタ情報で train_multitask_trial には存在しない引数
    _skip_keys = {'auto_lr_target_epoch', 'learning_rate_nohead', 'learning_rate_head', 'learning_rate_ft'}
    for key, value in params.items():
        if key in _skip_keys:
            continue
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--single_task_mode", str(SINGLE_TASK_MODE)])
    cmd.extend(["--enable_early_stopping", "False"])
    
    logger.info(f"[Calibration] Running {cal_epochs} epochs with LR={lr:.8f}...")
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    
    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        _log_subprocess_line(line, tag="[Cal]")
    
    rc = process.wait()
    full_output = "\n".join(output_lines)
    if rc != 0:
        _log_subprocess_fail("[Calibration] train_multitask_trial", rc, output_lines)

    # BEST_EPOCH抽出 (FT時はFT_BEST_EPOCHを優先)
    is_ft = str(params.get('fine_tune', 'False')).lower() == 'true'
    if is_ft:
        match_epoch = re.search(r"FT_BEST_EPOCH:\s*(\d+)", full_output)
    else:
        match_epoch = re.search(r"BEST_EPOCH:\s*(\d+)", full_output)
    best_epoch = int(match_epoch.group(1)) if match_epoch else cal_epochs
    
    # スコア抽出
    match_score = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
    score = float(match_score.group(1)) if match_score else 0.0
    # 最終エポックの精度（LR調整終了条件と合わせるため）
    epoch_scores = re.findall(r"MinClassAcc=([\d.]+)", full_output)
    last_epoch_accu = float(epoch_scores[-1]) if epoch_scores else 0.0

    logger.info(f"[Calibration] Result: BestEpoch={best_epoch}/{cal_epochs}, Score={score:.4f}")
    if not is_ft:
        _maybe_record_head_only_phase_best(params, score)
    return best_epoch, score, last_epoch_accu


def calibrate_base_lr(current_params, initial_lr, cal_epochs=10, target_best_epoch=None, score_priority=False):
    """
    cal_epochs の学習を繰り返し、target_best_epoch でベストになるLRを二分探索で探す。
    - best_epoch < target → LR高すぎ → 下げる（上限記録）
    - best_epoch > target → LR低すぎ → 上げる（下限記録）
    - 両方の境界が揃ったら中間値を試す
    
    Returns:
        tuple: (calibrated_lr, final_score)
    """
    if target_best_epoch is None:
        target_min = target_max = float(max(1, cal_epochs // 2))
    elif isinstance(target_best_epoch, tuple):
        target_min = float(target_best_epoch[0])
        target_max = float(target_best_epoch[1])
    else:
        target_min = target_max = float(target_best_epoch)
        
    current_lr = initial_lr
    
    logger.info(f"\n{'='*50}")
    logger.info(f"LR Calibration Start")
    if target_min == target_max:
        logger.info(f"  cal_epochs={cal_epochs}, target_best_epoch={target_min:.0f}")
    else:
        logger.info(f"  cal_epochs={cal_epochs}, target_best_epoch=[{target_min:.0f}, {target_max:.0f}]")
    logger.info(f"  initial_lr = {initial_lr:.8f}")
    logger.info(f"{'='*50}")
    
    best_candidate = None  # (sort_key, lr, best_epoch, score)
    max_iterations = LR_CALIBRATION_MAX_ITERATIONS

    # 二分探索の境界 (LR値で管理)
    lr_low = None   # best_epoch > target になった最大LR（LR低すぎ側）
    lr_high = None  # best_epoch < target になった最小LR（LR高すぎ側）

    for iteration in range(max_iterations):
        best_epoch, score, last_epoch_accu = run_calibration_trial(current_params, current_lr, cal_epochs)

        if best_epoch < target_min:
            distance = target_min - best_epoch
        elif best_epoch > target_max:
            distance = best_epoch - target_max
        else:
            distance = 0.0
            
        if score_priority:
            candidate = (-score, distance, current_lr, best_epoch, score)
        else:
            candidate = (distance, -score, current_lr, best_epoch, score)
        if best_candidate is None or candidate < best_candidate:
            best_candidate = candidate
        
        logger.info(f"Calibration #{iteration+1}: LR={current_lr:.8f}, BestEpoch={best_epoch}/{cal_epochs}, Score={score:.4f}, Distance={distance:.0f}")

        should_stop, stop_msg = lr_calibration_should_stop(best_epoch, last_epoch_accu, score)
        if should_stop and stop_msg:
            logger.info(stop_msg)
            break
        if iteration >= LR_CALIBRATION_MAX_ITERATIONS - 1:
            logger.info(f"Reached max calibration iterations ({LR_CALIBRATION_MAX_ITERATIONS}). Stopping calibration.")
            break

        # 境界更新
        if best_epoch < target_min:
            # LR高すぎ → 上限を記録
            if lr_high is None or current_lr < lr_high:
                lr_high = current_lr
        else:
            # LR低すぎ → 下限を記録
            if lr_low is None or current_lr > lr_low:
                lr_low = current_lr
        
        # 次のLR決定
        if lr_low is not None and lr_high is not None:
            # 両境界あり → 幾何平均（log空間の中点）
            # LR は乗算スケールで効くため、算術平均 (low+high)/2 より
            # 幾何平均 sqrt(low*high) の方が対称で収束が速い。
            new_lr = math.sqrt(lr_low * lr_high)
            logger.info(f"  Bisection (geom): low={lr_low:.8f}, high={lr_high:.8f}, mid={new_lr:.8f}")
        else:
            # 片側のみ → raw ratio scaling（クランプなし）
            # 旧: scale = max(0.3, min(scale, 3.0))
            # 撤去理由: 二分探索に入ると振動せず単調収束するため、片側探索でも大胆に
            # 動いた方が早く反対側境界を踏める。optimize_sequential.py と挙動統一。
            target_mid = (target_min + target_max) / 2.0
            scale = compute_lr_adjustment_ratio(best_epoch, target_epoch=int(target_mid), total_epochs=cal_epochs)
            new_lr = current_lr * scale
            logger.info(
                f"  Ratio scale: {scale:.4f}"
            )
        
        logger.info(f"  LR: {current_lr:.8f} -> {new_lr:.8f}")
        
        # 収束判定: LR変化が十分小さい
        if abs(new_lr - current_lr) / max(current_lr, 1e-12) < 0.02:
            logger.info(f"  LR change too small. Stopping.")
            break
        
        current_lr = new_lr

    if best_candidate is not None:
        _, _, chosen_lr, chosen_epoch, chosen_score = best_candidate
        logger.info(
            f"Calibration selected best candidate: LR={chosen_lr:.8f}, "
            f"BestEpoch={chosen_epoch}/{cal_epochs}, Score={chosen_score:.4f}"
        )
        current_lr, score = chosen_lr, chosen_score

    logger.info(f"\nCalibrated Base LR: {current_lr:.8f}, Final Score: {score:.4f}")
    logger.info(f"{'='*50}")
    return current_lr, score


def search_ft_lr_by_targets(current_params, initial_lr, targets=(10, 11, 12, 13, 14, 15), cal_epochs=20):
    """
    Step 4.7: FT 用 LR を **一回の** `calibrate_base_lr` で決める。

    `targets` の最小・最大を帯 ``[lo, hi]`` とみなし、`target_best_epoch=(lo, hi)` として
    ベスト epoch がその帯に入るように二分＋片側比で LR を探索する（**各 target ごとに
    `initial_lr` に戻して別キャリブを繰り返さない**）。

    旧: 10..15 を 1 本ずつ `calibrate_base_lr(..., target_best_epoch=t)` で実行し、
    毎回 `initial_lr` からやり直していた。
    """
    if targets is None:
        lo, hi = 10, 15
    else:
        tlist = list(targets)
        if not tlist:
            lo, hi = 10, 15
        else:
            lo, hi = int(min(tlist)), int(max(tlist))
    logger.info(f"\n{'='*50}")
    logger.info(
        f"FT LR Search (single pass): target_best_epoch band=[{lo}, {hi}], "
        f"cal_epochs={cal_epochs}, start_lr={initial_lr:.8f}"
    )
    logger.info(f"{'='*50}")
    lr, score = calibrate_base_lr(
        current_params,
        initial_lr=initial_lr,
        cal_epochs=cal_epochs,
        target_best_epoch=(lo, hi),
    )
    logger.info(f"\n{'='*50}")
    logger.info(
        f"FT LR Search Complete: band=[{lo},{hi}], Selected LR={lr:.8f}, Val={score:.4f}"
    )
    logger.info(f"{'='*50}")
    return lr, score


def main():
    logger.info("Starting Sequential Training Optimization (Full Replacement for Bayesian)")
    # Step 1.1 でバックボーン確定 → Step 1.2 で確定 model の head LR をキャリブ（旧 Step1 の B0 先行キャリブは廃止）。

    persisted_calib_ctx = None
    if os.path.isfile(BEST_PARAMS_FILE):
        try:
            with open(BEST_PARAMS_FILE, encoding='utf-8') as f:
                _bp_prev = json.load(f)
            persisted_calib_ctx = parse_lr_calib_context(_bp_prev.get(LR_CALIB_CONTEXT_JSON_KEY))
        except Exception as e:
            logger.warning(f"Could not read prior {LR_CALIB_CONTEXT_JSON_KEY} from {BEST_PARAMS_FILE}: {e}")
    _LR_CALIB_STATE['persisted_ctx'] = persisted_calib_ctx
    _LR_CALIB_STATE['last_ctx'] = None
    _reset_head_only_phase_best()

    def initial_lr_for_calibrate(model_name: str, fine_tune_val) -> float:
        n = train_sequential_data_file_count()
        mode = lr_calib_mode_from_fine_tune(fine_tune_val)
        lr, msg = resolve_calib_initial_lr(
            model_name, n, mode,
            last_ctx=_LR_CALIB_STATE['last_ctx'],
            persisted_ctx=_LR_CALIB_STATE['persisted_ctx'],
        )
        logger.info(f"  LR calib: {msg}")
        logger.info(
            f"  → initial_lr={lr:.8g} (data_file_count={n}, mode={mode}, model={model_name})"
        )
        return lr

    def record_calibrate_context(model_name: str, fine_tune_val, calibrated_lr: float) -> None:
        n = train_sequential_data_file_count()
        mode = lr_calib_mode_from_fine_tune(fine_tune_val)
        _LR_CALIB_STATE['last_ctx'] = make_lr_calib_context(
            model_name, n, mode, calibrated_lr
        )

    # 初期パラメータ (デフォルト値)
    current_params = {
        'model_name': LRCALIB_BASE_BACKBONE,
        'num_dense_layers': 1,
        'dense_units': 128,
        'dropout': 0.3,
        'head_dropout': 0.3,
        'learning_rate': 1e-3,
        'epochs': 20,
        'rotation_range': 0.0,
        'width_shift_range': 0.0,
        'height_shift_range': 0.0,
        'zoom_range': 0.0,
        'horizontal_flip': 'False',
        'mixup_alpha': 0.0,
        'label_smoothing': 0.0,
        'weight_decay': 0.0,
        'fine_tune': 'False'
    }

    # head-only: optimize_param / shift で同点が出た (param, 同点候補) を蓄積（Step1.1・1.5・2/3 すべて渡す）→3.8
    head_only_tie_log: list = []

    # --- Step 1.1: Model Architecture ---
    # 各候補 model は run_trial 内で (model, data_count, head|ft) に応じた開始 LR を resolve（B0 の LR をそのまま流用しない）。
    # 毎回 optimize_param（best_train_params の model_name ではスキップしない）
    best_model, _ = optimize_param(
        'model_name', MODEL_NAME_CANDIDATES, current_params, head_only_tie_log
    )
    current_params['model_name'] = best_model

    # --- Step 1.2: Head LR calibration（1.1 で確定したバックボーン向け・常に実行） ---
    logger.info(
        f"\n>>> Step 1.2: Head LR Calibration "
        f"(model_name={best_model}, target_best_epoch=13) <<<"
    )
    _il12 = initial_lr_for_calibrate(current_params['model_name'], current_params['fine_tune'])
    head_lr, _ = calibrate_base_lr(
        current_params, initial_lr=_il12,
        cal_epochs=20, target_best_epoch=13
    )
    record_calibrate_context(current_params['model_name'], current_params['fine_tune'], head_lr)
    current_params['learning_rate'] = head_lr
    logger.info(f"Calibrated head LR for {best_model}: {head_lr:.8f}")

    # --- Step 1.5: Weight Decay (Optimizer Selection) ---
    # 0.0=Adam, >0=AdamW
    best_wd, _ = optimize_param('weight_decay', [0.0, 1e-4, 1e-5], current_params, head_only_tie_log)
    current_params['weight_decay'] = best_wd
    
    # --- Step 2: Model Structure ---
    # Layers
    best_layers, _ = optimize_param('num_dense_layers', [1, 2], current_params, head_only_tie_log)
    current_params['num_dense_layers'] = best_layers
    
    # Units
    best_units, _ = optimize_param('dense_units', [128, 256], current_params, head_only_tie_log)
    current_params['dense_units'] = best_units
    
    # Dropout
    best_dropout, _ = optimize_param('dropout', [0.3, 0.5], current_params, head_only_tie_log)
    current_params['dropout'] = best_dropout

    # --- Step 3: Data Augmentation & Regularization ---
    # Label Smoothing (0.0=Off, 0.1=On)
    best_smoothing, _ = optimize_param('label_smoothing', [0.0, 0.1], current_params, head_only_tie_log)
    current_params['label_smoothing'] = best_smoothing

    # Mixup (0.0=Off, 0.2=On)
    # Mixupは強力な正則化なので、最初に決めるのが良い場合もあるが、他のAugmentationとの兼ね合いもある
    best_mixup, _ = optimize_param('mixup_alpha', [0.0, 0.2], current_params, head_only_tie_log)
    current_params['mixup_alpha'] = best_mixup

    # Rotation (0.0 - 0.2)
    best_rot, _ = optimize_param('rotation_range', [0.0, 0.1, 0.2], current_params, head_only_tie_log)
    current_params['rotation_range'] = best_rot
    
    # Shift (Width/Height set together)
    logger.info(f"\n>>> Optimizing Shift Ranges (Width & Height) <<<")
    shift_candidates = [0.0, 0.1]
    shift_scores = {}
    for val in shift_candidates:
        params = current_params.copy()
        params['width_shift_range'] = val
        params['height_shift_range'] = val
        shift_scores[val] = run_trial(params)
    m_sh = max(shift_scores.values()) if shift_scores else -1.0
    shift_tied = [c for c in shift_candidates if _is_same_score(shift_scores[c], m_sh)]
    if len(shift_tied) > 1:
        head_only_tie_log.append(("_shift_coupled", list(shift_tied)))
    best_shift = shift_tied[0]
    best_shift_score = m_sh
    if len(shift_tied) > 1:
        logger.info(
            f"Shift: TIED {shift_tied} (provisional first) → re-eval in resolve_tie_breaks"
        )
    logger.info(f"Finished optimizing Shift. Provisional best: {best_shift} (Score: {best_shift_score})")
    current_params['width_shift_range'] = best_shift
    current_params['height_shift_range'] = best_shift

    # Zoom
    best_zoom, _ = optimize_param('zoom_range', [0.0, 0.1], current_params, head_only_tie_log)
    current_params['zoom_range'] = best_zoom

    # Flip
    best_flip, final_score = optimize_param('horizontal_flip', ['True', 'False'], current_params, head_only_tie_log)
    current_params['horizontal_flip'] = best_flip

    if head_only_tie_log:
        logger.info(
            "\n>>> Step 3.8: Re-score tied candidates (all other head-only params fixed) <<<"
        )
        resolve_tie_breaks(current_params, head_only_tie_log)

    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION COMPLETE. STARTING FINE-TUNING...")
    logger.info(f"Best Params before FT: {current_params}")
    logger.info("="*50)

    N_FINAL_RUNS = 3
    FINAL_EPOCHS = 20

    # --- Step 3.9: Best Head Weights 再学習＆保存（FT の head carryover 用） ---
    # これまで Step 3 の best params は保存されても、その重み自体は破棄されていたため、
    # FT の warmup で head を再初期化 → 精度低下という問題があった。
    # 解決策: ここで best params の構成で head-only 学習を再実行し、ベスト重みを保存する。
    logger.info("\n>>> Step 3.9: Re-train head with best params and save best weights <<<")
    head_save_params = current_params.copy()
    head_save_params['fine_tune'] = 'False'
    head_save_params['epochs'] = 20
    # Step1/1.2 で確定した head LR を使う（current_params['learning_rate'] と同値）
    head_save_params['learning_rate'] = head_lr
    score_3_9 = train_and_save_best_head_weights(head_save_params, BEST_HEAD_WEIGHTS_PATH)
    _maybe_record_head_only_phase_best(head_save_params, score_3_9)

    # --- Step 3.5: Fine-Tuning LR Calibration（2 系統 → 高精度側を採用） ---
    # A: Step 3.9 の best_head をロード（warmup スキップ）
    # B: init なし・head_lr で warmup（従来どおり）
    # 同一 initial_lr で両方キャリブし、FINAL_VAL が高い方の LR・init/warmup を以降に引き継ぐ。
    current_params['fine_tune'] = 'True'
    current_params['epochs'] = 20
    current_params['unfreeze_layers'] = 60  # キャリブレーション用の暫定値
    base_ft = current_params.copy()

    _il35 = initial_lr_for_calibrate(current_params['model_name'], current_params['fine_tune'])

    lr_carry = None
    score_3_5_carry = -1.0
    if os.path.exists(BEST_HEAD_WEIGHTS_PATH):
        p_carry = base_ft.copy()
        p_carry['init_weights_path'] = BEST_HEAD_WEIGHTS_PATH
        p_carry['warmup_lr'] = 0.0
        p_carry['warmup_epochs'] = 0
        logger.info(
            "\n>>> Step 3.5 (A carryover): FT LR Calibration "
            f"(init_weights_path={BEST_HEAD_WEIGHTS_PATH}, warmup skipped) <<<"
        )
        lr_carry, score_3_5_carry = calibrate_base_lr(
            p_carry, initial_lr=_il35,
            cal_epochs=20, target_best_epoch=13
        )
        logger.info(
            f"Step 3.5 (A): calibrated_lr={lr_carry:.8g}, score={score_3_5_carry:.4f}"
        )
    else:
        logger.warning(
            "Step 3.5 (A carryover): skipped — best_head.weights.h5 missing"
        )

    p_scratch = base_ft.copy()
    p_scratch['init_weights_path'] = ''
    p_scratch['warmup_lr'] = head_lr
    p_scratch['warmup_epochs'] = 5
    logger.info(
        "\n>>> Step 3.5 (B warmup): FT LR Calibration "
        f"(no init_weights, warmup_lr={head_lr:.8g}, warmup_epochs=5) <<<"
    )
    lr_scratch, score_3_5_scratch = calibrate_base_lr(
        p_scratch, initial_lr=_il35,
        cal_epochs=20, target_best_epoch=13
    )
    logger.info(
        f"Step 3.5 (B): calibrated_lr={lr_scratch:.8g}, score={score_3_5_scratch:.4f}"
    )

    if lr_carry is not None and (
        score_3_5_carry > score_3_5_scratch
        or _is_same_score(score_3_5_carry, score_3_5_scratch)
    ):
        ft_calib_carryover_selected = True
        ft_lr, score_3_5 = lr_carry, score_3_5_carry
        logger.info(
            f"\nStep 3.5 winner: carryover "
            f"(score {score_3_5_carry:.4f} vs warmup path {score_3_5_scratch:.4f}; "
            f"tie goes to carryover)"
        )
        current_params['init_weights_path'] = BEST_HEAD_WEIGHTS_PATH
        current_params['warmup_lr'] = 0.0
        current_params['warmup_epochs'] = 0
    else:
        ft_calib_carryover_selected = False
        ft_lr, score_3_5 = lr_scratch, score_3_5_scratch
        logger.info(
            f"\nStep 3.5 winner: warmup / no carryover "
            f"(score {score_3_5_scratch:.4f}"
            + (
                f" vs carryover {score_3_5_carry:.4f}"
                if lr_carry is not None
                else " (carryover skipped)"
            )
            + ")"
        )
        current_params['init_weights_path'] = ''
        current_params['warmup_lr'] = head_lr
        current_params['warmup_epochs'] = 5

    record_calibrate_context(current_params['model_name'], current_params['fine_tune'], ft_lr)
    current_params['learning_rate'] = ft_lr

    head_only_best = float(_HEAD_ONLY_PHASE_BEST['score'])
    # FT キャリブが、FT 以前の head-only フェーズで観測したベスト検証より良くないとき head-only 仕上げ。
    head_finish = score_3_5 < head_only_best
    logger.info(
        f"\n{'='*50}\n"
        f"Head-only phase best (pre-FT) FINAL_VAL={head_only_best:.4f} "
        f"[includes Step3.9={score_3_9:.4f} and prior head trials/calib]\n"
        f"Step 3.5 (FT LR calib, selected={'carryover' if ft_calib_carryover_selected else 'warmup'}) "
        f"FINAL_VAL={score_3_5:.4f}\n"
        f"-> {'HEAD-ONLY finish: skip FT Steps 4–4.7, brush-up with head Best-of-N' if head_finish else 'FT pipeline: Steps 4–4.7 as usual'}\n"
        f"{'='*50}"
    )

    if head_finish:
        current_params['fine_tune'] = 'False'
        current_params['learning_rate'] = head_lr
        current_params['epochs'] = FINAL_EPOCHS
        if os.path.exists(BEST_HEAD_WEIGHTS_PATH):
            current_params['init_weights_path'] = BEST_HEAD_WEIGHTS_PATH
            current_params['warmup_lr'] = 0.0
            current_params['warmup_epochs'] = 0
        else:
            current_params['init_weights_path'] = ''
            current_params['warmup_lr'] = head_lr
            current_params['warmup_epochs'] = 5
            logger.warning(
                "head_finish: best_head.weights.h5 missing — brush-up runs without carryover"
            )
        logger.info(
            "\n>>> Step 4 & 4.5: skipped (head-only finish — unfreeze / FT 再キャリブ不要) <<<"
        )
    else:
        # --- Step 4: Unfreeze Layers Optimization ---
        best_unfreeze, _ = optimize_param('unfreeze_layers', [20, 40, 60, 999], current_params)
        current_params['unfreeze_layers'] = best_unfreeze

        # --- Step 4.5: FT LR Re-calibration (unfreeze_layers確定後) ---
        if best_unfreeze != 60:
            logger.info(
                f"\n>>> Step 4.5: FT LR Re-calibration "
                f"(unfreeze_layers={best_unfreeze}, 暫定60と異なるため再調整) <<<"
            )
            _il45 = initial_lr_for_calibrate(current_params['model_name'], current_params['fine_tune'])
            ft_lr2, _ = calibrate_base_lr(
                current_params, initial_lr=_il45,
                cal_epochs=20, target_best_epoch=13
            )
            record_calibrate_context(current_params['model_name'], current_params['fine_tune'], ft_lr2)
            current_params['learning_rate'] = ft_lr2
        else:
            logger.info(
                f"\n>>> Step 4.5: Skipped (unfreeze_layers=60 = キャリブレーション時と同値) <<<"
            )

    # --- Final: Best-of-N（FT: seed 選定→4.7 / head-only: Step3.9 からのブラッシュアップ） ---
    logger.info(f"\n{'='*50}")
    if head_finish:
        logger.info(
            f"Final: Best-of-{N_FINAL_RUNS} head-only brush-up "
            f"(epochs={FINAL_EPOCHS}, LR=head_lr, init=Step3.9 weights)"
        )
    else:
        logger.info(
            f"Final: Best-of-{N_FINAL_RUNS} runs (epochs={FINAL_EPOCHS}, LR=Step4.5 まで) — seed 選定"
        )
    logger.info(f"{'='*50}")

    best_bon_score = -1.0
    best_seed = 42

    for run_idx in range(N_FINAL_RUNS):
        seed = 42 + run_idx
        bon_params = current_params.copy()
        bon_params['epochs'] = FINAL_EPOCHS
        bon_params['seed'] = seed
        if head_finish:
            os.makedirs(MODEL_DIR, exist_ok=True)
            bon_params['export_model_path'] = os.path.join(MODEL_DIR, f"model_seed{seed}.keras")
        score = run_trial(bon_params)
        logger.info(f"  Best-of-N #{run_idx+1} (seed={seed}): Score={score:.4f}")
        if score > best_bon_score:
            best_bon_score = score
            best_seed = seed

    if head_finish:
        logger.info(
            f"Best-of-N (head-only): seed={best_seed}, Score={best_bon_score:.4f} (LR=head_lr)"
        )
    else:
        logger.info(f"Best-of-N: seed={best_seed}, Score={best_bon_score:.4f} (LR=pre-4.7)")

    _bon_src = os.path.join(MODEL_DIR, f"model_seed{best_seed}.keras")
    _bon_dst = os.path.join(MODEL_DIR, "best_sequential_model.keras")
    if os.path.exists(_bon_src):
        shutil.copy2(_bon_src, _bon_dst)
        if head_finish:
            logger.info(f"Best-of-N head-only model -> {_bon_dst}")
        else:
            logger.info(
                f"Best-of-N model -> {_bon_dst} (4.7 前に保存; 4.7 は採用 LR/スコアの探索のみ)"
            )

    if head_finish:
        logger.info("\n>>> Step 4.7: skipped (head-only finish — FT LR 帯探索は不要) <<<")
        final_lr = float(head_lr)
        final_report_score = best_bon_score
    else:
        if os.path.exists(CACHE_FILE):
            try:
                os.remove(CACHE_FILE)
                logger.info(f"Step 4.7: removed train opt cache: {CACHE_FILE}")
            except OSError as e:
                logger.warning(f"Step 4.7: could not remove {CACHE_FILE}: {e}")

        current_params['seed'] = best_seed
        logger.info("\n>>> Step 4.7: Final FT LR Search (after best-of-N seed) <<<")
        _il47 = initial_lr_for_calibrate(current_params['model_name'], current_params['fine_tune'])
        final_lr, final_ft_score = search_ft_lr_by_targets(
            current_params, initial_lr=_il47,
            targets=[10, 11, 12, 13, 14, 15], cal_epochs=20
        )
        record_calibrate_context(current_params['model_name'], current_params['fine_tune'], final_lr)
        current_params['learning_rate'] = final_lr
        final_report_score = final_ft_score

        if final_ft_score > best_bon_score:
            logger.info(
                f"\n>>> Step 4.7 export: score {final_ft_score:.4f} > Best-of-N {best_bon_score:.4f} — "
                f"full FT ({FINAL_EPOCHS} ep) at LR={final_lr:.8g}, seed={best_seed} <<<"
            )
            export_params = current_params.copy()
            export_params['seed'] = best_seed
            export_params['epochs'] = FINAL_EPOCHS
            export_params.pop('export_model_path', None)
            export_score = run_trial(export_params)
            logger.info(f"  Full FT at Step 4.7 LR: reported score={export_score:.4f}")
            _src47 = os.path.join(MODEL_DIR, f"model_seed{best_seed}.keras")
            _dst47 = os.path.join(MODEL_DIR, "best_sequential_model.keras")
            if os.path.exists(_src47):
                shutil.copy2(_src47, _dst47)
                logger.info(f"Step 4.7 LR model saved -> {_dst47}")
                final_report_score = float(export_score)
            else:
                logger.warning(
                    "Step 4.7 export: model_seed keras missing after run_trial; "
                    "keeping Best-of-N copy at best_sequential_model.keras"
                )
        else:
            logger.info(
                f"\n>>> Step 4.7: score {final_ft_score:.4f} <= Best-of-N {best_bon_score:.4f} — "
                f"keep Best-of-N weights in best_sequential_model.keras <<<"
            )
    for run_idx in range(N_FINAL_RUNS):
        s = 42 + run_idx
        path = os.path.join(MODEL_DIR, f'model_seed{s}.keras')
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    logger.info(f"\n{'='*50}")
    logger.info("ALL PROCESSES COMPLETE")
    if head_finish:
        logger.info(
            f"finish=head_only | head-only phase best={head_only_best:.4f} vs Step3.5(FT cal)={score_3_5:.4f} "
            f"(Step3.9={score_3_9:.4f}) | "
            f"best-of-N: seed={best_seed}, Score={best_bon_score:.4f}, LR={final_lr:.8g}"
        )
    else:
        logger.info(
            f"finish=ft | best-of-N (pre-4.7 LR): seed={best_seed}, Score={best_bon_score:.4f} | "
            f"Step4.7 calib (FT LR band [10,15]): Score={final_ft_score:.4f}, LR={final_lr:.8g} | "
            f"saved model score={final_report_score:.4f}"
        )
    logger.info(f"{'='*50}")

    # Save Best Params (ベストseedを含める)
    best_params = current_params.copy()
    best_params['seed'] = best_seed
    best_params['epochs'] = FINAL_EPOCHS
    best_params['finish_mode'] = 'head_only' if head_finish else 'fine_tune'
    best_params['score_step_3_9_head'] = float(score_3_9)
    best_params['score_step_3_5_ft_calib'] = float(score_3_5)
    best_params['score_head_only_phase_best'] = float(head_only_best)
    best_params['ft_calib_carryover_selected'] = bool(ft_calib_carryover_selected)
    best_params['score_step_3_5_ft_calib_carry'] = (
        float(score_3_5_carry) if lr_carry is not None else None
    )
    best_params['lr_step_3_5_ft_calib_carry'] = (
        float(lr_carry) if lr_carry is not None else None
    )
    best_params['score_step_3_5_ft_calib_warmup'] = float(score_3_5_scratch)
    best_params['lr_step_3_5_ft_calib_warmup'] = float(lr_scratch)
    # 保存する LR は learning_rate のみ（キャリブ結果・終端フェーズの採用値）。旧キーは削除。
    for _k in ('learning_rate_head', 'learning_rate_ft', 'learning_rate_nohead'):
        best_params.pop(_k, None)
    best_params['learning_rate'] = float(final_lr)
    _ctx_mode = 'head' if head_finish else 'ft'
    best_params[LR_CALIB_CONTEXT_JSON_KEY] = make_lr_calib_context(
        best_params.get('model_name', current_params['model_name']),
        train_sequential_data_file_count(),
        _ctx_mode,
        final_lr,
    )
    with open(BEST_PARAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Best training params saved to {BEST_PARAMS_FILE}")

    # 結果を自動記録
    from components.result_logger import log_result
    log_result("train_sequential", {
        "best_score": final_report_score,
        "best_params": best_params,
    })

if __name__ == "__main__":
    main()
    # 処理完了通知音
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
