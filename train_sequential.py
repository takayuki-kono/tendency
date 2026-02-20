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

    cmd = [PYTHON_EXEC, "components/train_multitask_trial.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
        
    # シングルタスクモード設定
    cmd.extend(["--single_task_mode", str(SINGLE_TASK_MODE)])

    try:
        # Popenでリアルタイム出力
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace'
        )
        
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            # エポック情報やスコアを含む行をリアルタイム表示
            if any(kw in line for kw in ['Epoch ', 'FINAL_VAL_ACCURACY', 'TASK_', 'MinClassAcc', 'Avg=']):
                logger.info(f"  [Train] {line}")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Training failed (returncode={process.returncode})")
            return 0.0

        full_output = "\n".join(output_lines)

        # スコア抽出 (Task A)
        match_a = re.search(r"FINAL_VAL_ACCURACY:\s*(\d+\.\d+)", full_output)
        
        # 他のタスクも抽出してログに出す
        for char_code in range(ord('A'), ord('Z') + 1):
            task_label = chr(char_code)
            match_task = re.search(f"TASK_{task_label}_ACCURACY:\s*(\d+\.\d+)", full_output)
            if match_task:
                logger.info(f"Task {task_label} Accuracy: {float(match_task.group(1))}")

        # 詳細なクラス別精度をログに転記
        details_match = re.search(r"--- Task [A-Z] Details.*", full_output, re.DOTALL)
        if details_match:
            logger.info("\n[Detailed Class Accuracy]")
            logger.info(details_match.group(0).strip())

        if match_a:
            score = float(match_a.group(1))
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

def run_calibration_trial(current_params, lr, cal_epochs=5):
    """
    LRキャリブレーション用: 指定パラメータで学習を実行し、BEST_EPOCHを返す。
    
    Returns:
        tuple: (best_epoch, score)
    """
    params = current_params.copy()
    params['learning_rate'] = lr
    params['epochs'] = cal_epochs
    
    # キャッシュ確認
    params_str = json.dumps({k: params[k] for k in sorted(params.keys())})
    file_count = count_files(DATA_SOURCE_DIR)
    key_src = f"cal_{params_str}_count={file_count}"
    cache_key = hashlib.md5(key_src.encode('utf-8')).hexdigest()
    
    cache = load_cache()
    if cache_key in cache:
        best_epoch, score = cache[cache_key]
        logger.info(f"[Calibration] Cache Hit! LR={lr:.8f} -> BestEpoch={best_epoch}/{cal_epochs}, Score={score:.4f}")
        return best_epoch, score

    cmd = [PYTHON_EXEC, "components/train_multitask_trial.py"]
    for key, value in params.items():
        if key == 'auto_lr_target_epoch':
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
    
    logger.info(f"[Calibration] Result: BestEpoch={best_epoch}/{cal_epochs}, Score={score:.4f}")
    
    # キャッシュ保存
    cache = load_cache()
    cache[cache_key] = (best_epoch, score)
    save_cache(cache)
    
    return best_epoch, score


def calibrate_base_lr(current_params, initial_lr, cal_epochs=10, target_best_epoch=None, score_priority=False):
    """
    cal_epochs の学習を繰り返し、target_best_epoch でベストになるLRを探す。
    
    原理:
    - best_epoch < target → LR高すぎ → 下げる
    - best_epoch > target → LR低すぎ → 上げる
    - best_epoch / target の比率でLRをスケーリング
    
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
        
    target_in_cal = (target_min + target_max) / 2.0
    
    current_lr = initial_lr
    
    logger.info(f"\n{'='*50}")
    logger.info(f"LR Calibration Start")
    if target_min == target_max:
        logger.info(f"  cal_epochs={cal_epochs}, target_best_epoch={target_min:.0f}")
    else:
        logger.info(f"  cal_epochs={cal_epochs}, target_best_epoch=[{target_min:.0f}, {target_max:.0f}]")
    logger.info(f"  initial_lr = {initial_lr:.8f}")
    logger.info(f"{'='*50}")
    
    best_candidate = None  # (distance, -score, lr, best_epoch, score)
    epoch_history = []  # 中央値ベースの収束判定用
    max_iterations = 5
    for iteration in range(max_iterations):
        best_epoch, score = run_calibration_trial(current_params, current_lr, cal_epochs)
        epoch_history.append(best_epoch)
        
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
        
        # 中央値を計算
        sorted_epochs = sorted(epoch_history)
        median_epoch = sorted_epochs[len(sorted_epochs) // 2]
        
        if median_epoch < target_min:
            median_distance = target_min - median_epoch
        elif median_epoch > target_max:
            median_distance = median_epoch - target_max
        else:
            median_distance = 0.0
        
        logger.info(f"Calibration #{iteration+1}: LR={current_lr:.8f}, BestEpoch={best_epoch}/{cal_epochs}, Score={score:.4f}")
        logger.info(f"  Epoch history: {epoch_history}, median={median_epoch}, median_dist={median_distance:.0f}")
        
        # 中央値がターゲット範囲に一致（または内包）したら収束
        if target_min <= median_epoch <= target_max:
            logger.info(f"Calibration converged! median={median_epoch} is within target [{target_min:.0f}, {target_max:.0f}]")
            break
        
        # LRスケーリング: (best_epoch / target) ^ 0.75 で比率を計算
        if best_epoch < target_min:
            raw_scale = best_epoch / target_min
        elif best_epoch > target_max:
            raw_scale = best_epoch / target_max
        else:
            raw_scale = 1.0
            
        scale = raw_scale ** 0.75 if raw_scale >= 0 else 0.5
        scale = max(0.5, min(scale, 2.0))
        new_lr = current_lr * scale
        
        logger.info(f"  Adjusting: best_epoch={best_epoch} vs target_range=[{target_min:.0f}, {target_max:.0f}], raw={raw_scale:.2f}, scale(^0.75)={scale:.2f}")
        logger.info(f"  LR: {current_lr:.8f} -> {new_lr:.8f}")
        # 変化が小さすぎる場合は停滞とみなし終了
        if abs(new_lr - current_lr) / max(current_lr, 1e-12) < 0.01:
            logger.info("Calibration update became too small; stopping early.")
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
            cal_epochs=cal_epochs, target_best_epoch=t, score_priority=True
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
    # Epoch 5でベストになるLRをキャリブレーションし、取得したLRの0.8倍を採用
    logger.info("\n>>> Step 1: Base LR Calibration (Target Epoch 5, then x0.8) <<<")
    raw_base_lr, _ = calibrate_base_lr(
        current_params, initial_lr=1e-3,
        cal_epochs=10, target_best_epoch=5, score_priority=True
    )
    calibrated_lr = raw_base_lr * 0.8
    logger.info(f"Target 5 LR={raw_base_lr:.8f} -> Applied 80% LR={calibrated_lr:.8f}")
    current_params['learning_rate'] = calibrated_lr
    head_lr = calibrated_lr  # Phase 1 warmup用に保存
    # --- Step 1.1: Model Architecture (キャリブレーション済みLRで比較) ---
    best_model, _ = optimize_param('model_name', ['EfficientNetV2B0', 'EfficientNetV2S'], current_params)
    current_params['model_name'] = best_model
    
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
    
    # --- Step 3.5: Fine-Tuning LR Calibration ---
    # FT本番: target 10〜15の範囲で探索し、最も良かったスコアのLRを採用
    current_params['fine_tune'] = 'True'
    current_params['epochs'] = 20
    current_params['unfreeze_layers'] = 60  # キャリブレーション用の暫定値
    current_params['warmup_lr'] = head_lr  # Phase 1はヘッド用の高いLRを使用
    
    logger.info("\n>>> Step 3.5: FT LR Search (Targets 10-15) <<<")
    ft_lr, _ = search_ft_lr_by_targets(
        current_params, initial_lr=current_params['learning_rate'],
        targets=[10, 11, 12, 13, 14, 15], cal_epochs=20
    )
    current_params['learning_rate'] = ft_lr
    
    # --- Step 4: Unfreeze Layers Optimization ---
    best_unfreeze, _ = optimize_param('unfreeze_layers', [20, 40, 60, 999], current_params)
    current_params['unfreeze_layers'] = best_unfreeze
    
    # --- Step 4.5: FT LR Re-calibration (unfreeze_layers確定後) ---
    if best_unfreeze != 60:
        logger.info(f"\n>>> Step 4.5: FT LR Re-calibration (unfreeze_layers={best_unfreeze}, 暫定60と異なるため再調整) <<<")
        ft_lr2, _ = search_ft_lr_by_targets(
            current_params, initial_lr=current_params['learning_rate'],
            targets=[10, 11, 12, 13, 14, 15], cal_epochs=20
        )
        current_params['learning_rate'] = ft_lr2
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
