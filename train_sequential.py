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
    LR_MAX_ADJUSTMENTS, LR_LAST_ACCU_EPS,
    compute_lr_adjustment_ratio, lr_adjustment_decision, lr_calibration_should_stop,
)

def load_best_params():
    """前回の最終採用パラメータ（存在すれば）を読む。"""
    if os.path.exists(BEST_PARAMS_FILE):
        try:
            with open(BEST_PARAMS_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load {BEST_PARAMS_FILE}: {e}")
    return {}

def _get_head_lr_from_best(best_params: dict, default: float) -> float:
    # head 側LR（= fine_tune=False 時、または head-only フェーズ用）を引き継ぐ。
    # body (learning_rate_nohead) や FT (learning_rate_ft) は別条件で乖離するため参照しない。
    # 互換として warmup_lr（旧 FT 用 warmup）/ learning_rate（最古形式）も fallback に残す。
    for k in ("learning_rate_head", "warmup_lr", "learning_rate"):
        if k in best_params:
            try:
                return float(best_params[k])
            except Exception:
                pass
    return float(default)

def _get_ft_lr_from_best(best_params: dict, default: float) -> float:
    for k in ("learning_rate_ft", "learning_rate"):
        if k in best_params:
            try:
                return float(best_params[k])
            except Exception:
                pass
    return float(default)


def count_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        count += len(files)
    return count

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
        logger.info(f"Cache Hit! Skipping execution. Val Accuracy: {cache[cache_key]}")
        logger.info(f"Original Params: {params}")
        logger.info(f"{'='*50}")
        return cache[cache_key]

    logger.info(f"Cache Miss. Running process... (File Count: {file_count})")
    logger.info(f"{'='*50}")

    try:
        # LR自動調整: 許容範囲外なら target で再調整（定数はモジュール共通）
        current_lr = params.get('learning_rate', 1e-3)
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
            # Conditional Extension は train_multitask_trial 側で必要時のみ発動させる

            # Popenでリアルタイム出力
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace'
            )
            
            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                if any(kw in line for kw in ['Epoch ', 'FINAL_VAL_ACCURACY', 'TASK_', 'MinClassAcc', 'Avg=', 'BEST_EPOCH', 'FT_BEST_EPOCH']):
                    logger.info(f"  [Train] {line}")
            
            process.wait()
            
            if process.returncode != 0:
                logger.error(f"Training failed (returncode={process.returncode})")
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
            
            return score
        else:
            logger.error("Score not found in output.")
            return 0.0

    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return 0.0

def optimize_param(target_name, candidates, current_params):
    """
    1つのパラメータを候補の中から最適化する
    """
    logger.info(f"\n>>> Optimizing {target_name} (Candidates: {candidates}) <<<")
    
    best_val = current_params.get(target_name)
    best_score = -1.0
    
    scores = {}
    
    for val in candidates:
        # パラメータ設定
        params = current_params.copy()
        params[target_name] = val
        
        score = run_trial(params)
        scores[val] = score
    
    # 最良の値を見つける
    best_val = max(scores, key=scores.get)
    best_score = scores[best_val]
    
    logger.info(f"Finished optimizing {target_name}. Best: {best_val} (Score: {best_score})")
    return best_val, best_score

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

    logger.info(f"\n{'='*50}")
    logger.info(f"Saving Best Head Weights (epochs={params['epochs']})")
    logger.info(f"  learning_rate = {params.get('learning_rate'):.8f}")
    logger.info(f"  weights_path  = {weights_path}")
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
    # Conditional Extension は train_multitask_trial 側で必要時のみ発動させる
    
    logger.info(f"[Calibration] Running {cal_epochs} epochs with LR={lr:.8f}...")
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    
    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if any(kw in line for kw in ['Epoch ', 'BEST_EPOCH', 'FT_BEST_EPOCH', 'MinClassAcc', 'Avg=']):
            logger.info(f"  [Cal] {line}")
    
    process.wait()
    full_output = "\n".join(output_lines)
    
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
    max_iterations = LR_MAX_ADJUSTMENTS + 1  # trainのLR調整と同じ最大試行数

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
        if iteration >= LR_MAX_ADJUSTMENTS:
            logger.info(f"Reached max LR adjustments ({LR_MAX_ADJUSTMENTS}). Stopping calibration.")
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
            # 両境界あり → 二分探索（中間値）
            new_lr = (lr_low + lr_high) / 2.0
            logger.info(f"  Binary search: low={lr_low:.8f}, high={lr_high:.8f}, mid={new_lr:.8f}")
        else:
            # 片側のみ → 累積学習率比でスケーリング
            target_mid = (target_min + target_max) / 2.0
            scale = compute_lr_adjustment_ratio(best_epoch, target_epoch=int(target_mid), total_epochs=cal_epochs)
            scale = max(0.3, min(scale, 3.0))  # 極端な変更を防ぐ
            new_lr = current_lr * scale
            logger.info(f"  Cumsum ratio scaling: scale={scale:.4f}")
        
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


def search_ft_lr_by_targets(current_params, initial_lr, targets=[10, 11, 12, 13, 14, 15], cal_epochs=20):
    """
    複数のtarget_best_epochでキャリブレーションを行い、
    最もval_accuracyが良かったLRを採用する（FT本番用）
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"FT LR Search Start: Targets={targets}")
    logger.info(f"{'='*50}")
    
    best_candidate_val = -1.0
    best_candidate_lr = None
    best_target = None
    
    for t in targets:
        logger.info(f"\n--- Searching for Target Epoch: {t} ---")
        lr, score = calibrate_base_lr(
            current_params, initial_lr=initial_lr,
            cal_epochs=cal_epochs, target_best_epoch=t
        )
        if score > best_candidate_val:
            best_candidate_val = score
            best_candidate_lr = lr
            best_target = t
            
    logger.info(f"\n{'='*50}")
    logger.info(f"FT LR Search Complete")
    logger.info(f"Best Target: {best_target}, Selected LR: {best_candidate_lr:.8f}, Val Acc: {best_candidate_val:.4f}")
    logger.info(f"{'='*50}")
    
    return best_candidate_lr, best_candidate_val


def main():
    logger.info("Starting Sequential Training Optimization (Full Replacement for Bayesian)")
    
    prev_best = load_best_params()

    # 初期パラメータ (デフォルト値)
    current_params = {
        'model_name': 'EfficientNetV2B0',
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
    
    # --- Step 1: Learning Rate Calibration (デフォルトB0で) ---
    # LR_TARGET_EPOCH(13)でベストになるLRをキャリブレーション
    logger.info("\n>>> Step 1: Base LR Calibration (Target Epoch 13) <<<")
    head_initial_lr = _get_head_lr_from_best(prev_best, default=5e-4)
    calibrated_lr, _ = calibrate_base_lr(
        current_params, initial_lr=head_initial_lr,
        cal_epochs=20, target_best_epoch=13
    )
    logger.info(f"Calibrated Head LR={calibrated_lr:.8f}")
    head_lr = calibrated_lr
    current_params['learning_rate'] = head_lr
    current_params['learning_rate_head'] = head_lr
    # --- Step 1.1: Model Architecture (キャリブレーション済みLRで比較) ---
    best_model, _ = optimize_param('model_name', ['EfficientNetV2B0', 'EfficientNetV2S'], current_params)
    current_params['model_name'] = best_model

    # --- Step 1.2: Head LR Re-calibration (model_name 確定後) ---
    # Step1 は B0 で Cal したため、S が選ばれた場合は S に合わせて LR を取り直す
    # （そのままだと S に対して LR 過大になり BestEpoch が target=13 から大きく外れることが確認されたため）
    if best_model != 'EfficientNetV2B0':
        logger.info(
            f"\n>>> Step 1.2: Head LR Re-calibration "
            f"(model_name={best_model}, Step1 時と異なるため再調整) <<<"
        )
        head_lr2, _ = calibrate_base_lr(
            current_params, initial_lr=head_lr,
            cal_epochs=20, target_best_epoch=13
        )
        head_lr = head_lr2
        current_params['learning_rate'] = head_lr
        current_params['learning_rate_head'] = head_lr
        # NOTE: head LR と body LR は別条件で乖離する想定のため、learning_rate_nohead は
        # ここでは更新しない（head calibration の結果を body 側へミラーすると、後続の
        # body/FT LR の引き継ぎが狂う）。
    else:
        logger.info(
            f"\n>>> Step 1.2: Skipped (model_name={best_model} = Step1 キャリブレーション時と同値) <<<"
        )

    # --- Step 1.5: Weight Decay (Optimizer Selection) ---
    # 0.0=Adam, >0=AdamW
    best_wd, _ = optimize_param('weight_decay', [0.0, 1e-4, 1e-5], current_params)
    current_params['weight_decay'] = best_wd
    
    # --- Step 2: Model Structure ---
    # Layers
    best_layers, _ = optimize_param('num_dense_layers', [1, 2], current_params)
    current_params['num_dense_layers'] = best_layers
    
    # Units
    best_units, _ = optimize_param('dense_units', [128, 256], current_params)
    current_params['dense_units'] = best_units
    
    # Dropout
    best_dropout, _ = optimize_param('dropout', [0.3, 0.5], current_params)
    current_params['dropout'] = best_dropout

    # --- Step 3: Data Augmentation & Regularization ---
    # Label Smoothing (0.0=Off, 0.1=On)
    best_smoothing, _ = optimize_param('label_smoothing', [0.0, 0.1], current_params)
    current_params['label_smoothing'] = best_smoothing

    # Mixup (0.0=Off, 0.2=On)
    # Mixupは強力な正則化なので、最初に決めるのが良い場合もあるが、他のAugmentationとの兼ね合いもある
    best_mixup, _ = optimize_param('mixup_alpha', [0.0, 0.2], current_params)
    current_params['mixup_alpha'] = best_mixup

    # Rotation (0.0 - 0.2)
    best_rot, _ = optimize_param('rotation_range', [0.0, 0.1, 0.2], current_params)
    current_params['rotation_range'] = best_rot
    
    # Shift (Width/Height set together)
    logger.info(f"\n>>> Optimizing Shift Ranges (Width & Height) <<<")
    shift_candidates = [0.0, 0.1]
    best_shift_score = -1.0
    best_shift = 0.0
    
    for val in shift_candidates:
        params = current_params.copy()
        params['width_shift_range'] = val
        params['height_shift_range'] = val
        score = run_trial(params)
        if score > best_shift_score:
            best_shift_score = score
            best_shift = val
            
    logger.info(f"Finished optimizing Shift. Best: {best_shift} (Score: {best_shift_score})")
    current_params['width_shift_range'] = best_shift
    current_params['height_shift_range'] = best_shift

    # Zoom
    best_zoom, _ = optimize_param('zoom_range', [0.0, 0.1], current_params)
    current_params['zoom_range'] = best_zoom
    
    # Flip
    best_flip, final_score = optimize_param('horizontal_flip', ['True', 'False'], current_params)
    current_params['horizontal_flip'] = best_flip
    
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION COMPLETE. STARTING FINE-TUNING...")
    logger.info(f"Best Params before FT: {current_params}")
    logger.info("="*50)

    # --- Step 3.9: Best Head Weights 再学習＆保存（FT の head carryover 用） ---
    # これまで Step 3 の best params は保存されても、その重み自体は破棄されていたため、
    # FT の warmup で head を再初期化 → 精度低下という問題があった。
    # 解決策: ここで best params の構成で head-only 学習を再実行し、ベスト重みを保存する。
    logger.info("\n>>> Step 3.9: Re-train head with best params and save best weights <<<")
    head_save_params = current_params.copy()
    head_save_params['fine_tune'] = 'False'
    head_save_params['epochs'] = 20
    # Step1/1.2 で確定した head LR を使う（learning_rate_head がキャリブ済み値）
    head_save_params['learning_rate'] = head_lr
    train_and_save_best_head_weights(head_save_params, BEST_HEAD_WEIGHTS_PATH)

    # --- Step 3.5: Fine-Tuning LR Calibration ---
    # FT本番: Head の重みは Step 3.9 で保存した best_head_weights をロードして引き継ぐ（head carryover）。
    # これにより warmup フェーズは不要になるため warmup_lr=0 に設定。
    current_params['fine_tune'] = 'True'
    current_params['epochs'] = 20
    current_params['unfreeze_layers'] = 60  # キャリブレーション用の暫定値
    if os.path.exists(BEST_HEAD_WEIGHTS_PATH):
        current_params['init_weights_path'] = BEST_HEAD_WEIGHTS_PATH
        current_params['warmup_lr'] = 0.0  # head carryover 済みなので warmup スキップ
        current_params['warmup_epochs'] = 0
        logger.info(
            f"Head carryover enabled: init_weights_path={BEST_HEAD_WEIGHTS_PATH} "
            f"(warmup skipped)"
        )
    else:
        # フォールバック: best_head_weights 保存に失敗した場合は従来どおり warmup を走らせる
        current_params['init_weights_path'] = ''
        current_params['warmup_lr'] = head_lr
        current_params['warmup_epochs'] = 5
        logger.warning(
            f"Head carryover disabled (file missing). Falling back to warmup_lr={head_lr:.8f}"
        )
    
    logger.info("\n>>> Step 3.5: FT LR Calibration (Target Epoch 13) <<<")
    ft_initial_lr = _get_ft_lr_from_best(prev_best, default=current_params['learning_rate'])
    ft_lr, _ = calibrate_base_lr(
        current_params, initial_lr=ft_initial_lr,
        cal_epochs=20, target_best_epoch=13
    )
    current_params['learning_rate'] = ft_lr
    current_params['learning_rate_ft'] = ft_lr
    
    # --- Step 4: Unfreeze Layers Optimization ---
    best_unfreeze, _ = optimize_param('unfreeze_layers', [20, 40, 60, 999], current_params)
    current_params['unfreeze_layers'] = best_unfreeze
    
    # --- Step 4.5: FT LR Re-calibration (unfreeze_layers確定後) ---
    if best_unfreeze != 60:
        logger.info(f"\n>>> Step 4.5: FT LR Re-calibration (unfreeze_layers={best_unfreeze}, 暫定60と異なるため再調整) <<<")
        ft_lr2, _ = calibrate_base_lr(
            current_params, initial_lr=current_params['learning_rate'],
            cal_epochs=20, target_best_epoch=13
        )
        current_params['learning_rate'] = ft_lr2
        current_params['learning_rate_ft'] = ft_lr2
    else:
        logger.info(f"\n>>> Step 4.5: Skipped (unfreeze_layers=60 = キャリブレーション時と同値) <<<")
    
    # --- Step 4.6: FT条件での正則化パラメータ再最適化 ---
    logger.info("\n>>> Step 4.6: Re-optimizing regularization under FT conditions <<<")
    
    # Dropout
    best_dropout_ft, _ = optimize_param('dropout', [0.3, 0.5], current_params)
    current_params['dropout'] = best_dropout_ft
    
    # Head Dropout (frozen phaseでは未最適化だったパラメータ)
    best_head_dropout, _ = optimize_param('head_dropout', [0.2, 0.3, 0.5], current_params)
    current_params['head_dropout'] = best_head_dropout
    
    # Weight Decay
    best_wd_ft, _ = optimize_param('weight_decay', [0.0, 1e-4, 1e-5], current_params)
    current_params['weight_decay'] = best_wd_ft
    
    # --- Step 4.7: Final FT LR Calibration (正則化変更後) ---
    logger.info("\n>>> Step 4.7: Final FT LR Calibration (after regularization re-opt) <<<")
    final_lr, _ = search_ft_lr_by_targets(
        current_params, initial_lr=current_params['learning_rate'],
        targets=[10, 11, 12, 13, 14, 15], cal_epochs=20
    )
    current_params['learning_rate'] = final_lr
    current_params['learning_rate_ft'] = final_lr
    
    # --- Final: Best-of-N Runs (上振れ狙い) ---
    N_FINAL_RUNS = 3
    FINAL_EPOCHS = 20
    logger.info(f"\n{'='*50}")
    logger.info(f"Final: Best-of-{N_FINAL_RUNS} runs (epochs={FINAL_EPOCHS}, different seeds)")
    logger.info(f"{'='*50}")
    
    best_final_score = -1.0
    best_seed = 42
    
    final_params = current_params.copy()
    final_params['epochs'] = FINAL_EPOCHS
    
    for run_idx in range(N_FINAL_RUNS):
        seed = 42 + run_idx
        final_params['seed'] = seed
        score = run_trial(final_params)
        logger.info(f"  Final Run #{run_idx+1} (seed={seed}): Score={score:.4f}")
        if score > best_final_score:
            best_final_score = score
            best_seed = seed
    
    final_ft_score = best_final_score
    
    # ベストseedのモデルを最終ファイルにコピー
    import shutil
    best_model_src = os.path.join(MODEL_DIR, f'model_seed{best_seed}.keras')
    best_model_dst = os.path.join(MODEL_DIR, 'best_sequential_model.keras')
    if os.path.exists(best_model_src):
        shutil.copy2(best_model_src, best_model_dst)
        logger.info(f"Best model (seed={best_seed}) -> {best_model_dst}")
    # 他seedのモデルファイルを削除
    for run_idx in range(N_FINAL_RUNS):
        seed = 42 + run_idx
        path = os.path.join(MODEL_DIR, f'model_seed{seed}.keras')
        if os.path.exists(path):
            os.remove(path)
    
    logger.info(f"\n{'='*50}")
    logger.info("ALL PROCESSES COMPLETE")
    logger.info(f"Best of {N_FINAL_RUNS} runs: seed={best_seed}, Score={final_ft_score:.4f}")
    logger.info(f"{'='*50}")

    # Save Best Params (ベストseedを含める)
    best_params = current_params.copy()
    best_params['seed'] = best_seed
    best_params['epochs'] = FINAL_EPOCHS
    # 互換: learning_rate はFT側として残す。head/ft は明示キーで保存する。
    # headなし側は learning_rate_nohead を正とし、互換で learning_rate_head も併記する
    best_params['learning_rate_nohead'] = float(best_params.get('learning_rate_nohead', best_params.get('learning_rate_head', head_lr)))
    best_params['learning_rate_head'] = float(best_params.get('learning_rate_head', best_params['learning_rate_nohead']))
    best_params['learning_rate_ft'] = float(best_params.get('learning_rate_ft', best_params.get('learning_rate', final_lr)))
    with open(BEST_PARAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Best training params saved to {BEST_PARAMS_FILE}")

    # 結果を自動記録
    from components.result_logger import log_result
    log_result("train_sequential", {
        "best_score": final_ft_score,
        "best_params": best_params,
    })

if __name__ == "__main__":
    main()
    # 処理完了通知音
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
