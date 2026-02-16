import subprocess
import sys
import re
import os
import logging
import time
import json
import hashlib
import winsound

# --- 設定 ---
# Python実行環境のパス
PYTHON_PREPROCESS = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe"
PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
DATA_SOURCE_DIR = "train" 

# 出力ディレクトリ
LOG_DIR = "outputs/logs"
CACHE_DIR = "outputs/cache"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "filter_opt_cache.json")
BEST_TRAIN_PARAMS_FILE = "outputs/best_train_params.json"

# 効率ベースの最適化設定
# 効率 = 精度向上 / フィルタリング枚数
# 効率が低いパラメータは自動的にスケールダウンまたは除外される
MIN_EFFICIENCY_THRESHOLD = 0.00001  # 最小効率閾値（これ以下は除外候補）

# LRキャリブレーション結果（main()で設定される）
CALIBRATED_BASE_LR = None
LR_SCALING_CONFIG_FILE = "outputs/lr_scaling_config.json"
LR_LOW_RATIO_THRESHOLD = 0.5
LR_LOW_RATIO_THRESHOLD = 0.5
LR_SCALING_EXP1 = 0.65 # Default
LR_SCALING_EXP2 = 1.25 # Default
BASE_RATIO = 1.0 # Default

# パスが存在しない場合はデフォルトを使用
if not os.path.exists(PYTHON_PREPROCESS): PYTHON_PREPROCESS = "python"
if not os.path.exists(PYTHON_TRAIN): PYTHON_TRAIN = "python"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'sequential_opt_log.txt'), mode='w', encoding='utf-8'),
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
    current_file_count = count_files("train") + count_files("validation")
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            # ファイル数が変わっていたらキャッシュクリア
            if cache.get('__file_count__') != current_file_count:
                logger.info(f"Data changed ({cache.get('__file_count__')} -> {current_file_count}). Clearing cache.")
                return {'__file_count__': current_file_count}
            return cache
        except:
            return {'__file_count__': current_file_count}
    return {'__file_count__': current_file_count}

def load_best_train_params():
    """最適化された学習パラメータを読み込む"""
    if os.path.exists(BEST_TRAIN_PARAMS_FILE):
        try:
            with open(BEST_TRAIN_PARAMS_FILE, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"Loaded best training params: {params}")
            return params
        except Exception as e:
            logger.warning(f"Failed to load best train params: {e}")
    else:
        logger.info("Best train params not found. Using defaults.")
    return {}

def save_cache(cache):
    # ファイル数も保存
    cache['__file_count__'] = count_files("train") + count_files("validation")
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

def run_calibration_trial(model_name, lr, cal_epochs=5):
    """
    LRキャリブレーション用: 前処理済みデータで学習のみ実行し、BEST_EPOCHを返す。
    preprocessed_multitask が既に準備されている前提で呼ぶこと。
    
    Returns:
        tuple: (best_epoch, score)
    """
    best_params = load_best_train_params()
    
    cmd_train = [PYTHON_TRAIN, "components/train_multitask_trial.py", "--model_name", model_name]
    for k, v in best_params.items():
        if k not in ['model_name', 'fine_tune', 'epochs', 'learning_rate', 'auto_lr_target_epoch']:
            cmd_train.extend([f"--{k}", str(v)])
    cmd_train.extend(["--learning_rate", str(lr)])
    cmd_train.extend(["--epochs", str(cal_epochs)])
    cmd_train.extend(["--fine_tune", "False"])
    cmd_train.extend(["--enable_early_stopping", "False"])
    
    logger.info(f"[Calibration] Running {cal_epochs} epochs with LR={lr:.8f}...")
    
    process = subprocess.Popen(
        cmd_train, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    
    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if any(kw in line for kw in ['Epoch ', 'BEST_EPOCH', 'MinClassAcc', 'Avg=']):
            logger.info(f"  [Cal] {line}")
    
    process.wait()
    full_output = "\n".join(output_lines)
    
    # BEST_EPOCH抽出
    match_epoch = re.search(r"BEST_EPOCH:\s*(\d+)", full_output)
    best_epoch = int(match_epoch.group(1)) if match_epoch else cal_epochs
    
    # スコア抽出
    match_score = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
    score = float(match_score.group(1)) if match_score else 0.0
    
    logger.info(f"[Calibration] Result: BestEpoch={best_epoch}/{cal_epochs}, Score={score:.4f}")
    return best_epoch, score


def calibrate_base_lr(model_name, initial_lr, cal_epochs=10, target_best_epoch=None):
    """
    cal_epochs の学習を繰り返し、中間 epoch でベストになるLRを探す。
    
    原理:
    - 10 epoch中の中間（5）でベスト → 適切な収束速度
    - best_epoch < target → LR高すぎ → 下げる
    - best_epoch > target → LR低すぎ → 上げる
    - best_epoch / target_in_cal の比率でLRをスケーリング
    
    preprocessed_multitask が既に準備されている前提で呼ぶこと。
    
    Returns:
        tuple: (calibrated_lr, final_score)
    """
    # cal_epochs中での目標ベストepoch
    # 既定は "前半の中間" を採用（10 epoch -> 5）
    if target_best_epoch is None:
        target_in_cal = float(max(1, cal_epochs // 2))
    else:
        target_in_cal = float(target_best_epoch)
    
    current_lr = initial_lr
    
    logger.info(f"\n{'='*50}")
    logger.info(f"LR Calibration Start")
    logger.info(f"  cal_epochs={cal_epochs}, target_best_epoch={target_in_cal:.0f}")
    logger.info(f"  initial_lr = {initial_lr:.8f}")
    logger.info(f"{'='*50}")
    
    best_candidate = None  # (distance, -score, lr, best_epoch, score)
    epoch_history = []  # 中央値ベースの収束判定用
    max_iterations = 5
    for iteration in range(max_iterations):
        best_epoch, score = run_calibration_trial(model_name, current_lr, cal_epochs)
        epoch_history.append(best_epoch)
        distance = abs(best_epoch - target_in_cal)
        candidate = (distance, -score, current_lr, best_epoch, score)
        if best_candidate is None or candidate < best_candidate:
            best_candidate = candidate
        
        # 中央値を計算
        sorted_epochs = sorted(epoch_history)
        median_epoch = sorted_epochs[len(sorted_epochs) // 2]
        
        logger.info(f"Calibration #{iteration+1}: LR={current_lr:.8f}, BestEpoch={best_epoch}/{cal_epochs}, Score={score:.4f}")
        logger.info(f"  Epoch history: {epoch_history}, median={median_epoch}")
        
        # 中央値がターゲットに一致したら収束
        if median_epoch == target_in_cal:
            logger.info(f"Calibration converged! median={median_epoch} == target={target_in_cal:.0f}")
            break
        
        # LRスケーリング: (best_epoch / target) ^ 0.75 で比率を計算
        # best_epoch=2, target=10 → 0.2^0.75=0.30 → LRを下げる
        # best_epoch=15, target=10 → 1.5^0.75=1.36 → LRを上げる
        raw_scale = best_epoch / target_in_cal
        scale = raw_scale ** 0.75 if raw_scale >= 0 else 0.5
        # 極端な変更を防ぐ（収束しやすいように少し保守的）
        scale = max(0.5, min(scale, 2.0))
        new_lr = current_lr * scale
        
        logger.info(f"  Adjusting: best_epoch={best_epoch} vs target={target_in_cal:.1f}, raw={raw_scale:.2f}, scale(^0.75)={scale:.2f}")
        logger.info(f"  LR: {current_lr:.8f} -> {new_lr:.8f}")
        # 変化が小さすぎる場合は停滞とみなし終了
        # if abs(new_lr - current_lr) / max(current_lr, 1e-12) < 0.01:
        #     logger.info("Calibration update became too small; stopping early.")
        #     break
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


def run_trial(pitch, sym, y_diff, mouth_open, eb_eye_high, eb_eye_low, sharpness_low, sharpness_high, face_size_low=0, face_size_high=0, retouching=0, mask=0, glasses=0, grayscale=False, model_name='EfficientNetV2B0'):
    """
    指定されたパラメータとモデルで前処理と学習を実行し、スコアを返す
    
    Returns:
        tuple: (raw_score, total_images, filtered_count)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating: Model={model_name}, Pitch={pitch}%, Sym={sym}%, Y-Diff={y_diff}%, Mouth-Open={mouth_open}%, Eb-High={eb_eye_high}%, Eb-Low={eb_eye_low}%, Sharp-L={sharpness_low}%, Sharp-H={sharpness_high}%, FaceSize-L={face_size_low}%, FaceSize-H={face_size_high}%, Retouch={retouching}%, Mask={mask}%, Glasses={glasses}%, Grayscale={grayscale}")
    
    file_count = count_files("train") + count_files("validation")
    # キャッシュキーにLR情報を含める（キャリブレーション後はLRが変わるため）
    lr_tag = f"_clr={CALIBRATED_BASE_LR:.8f}" if CALIBRATED_BASE_LR else ""
    cache_key = f"model={model_name}_pitch={pitch}_sym={sym}_ydiff={y_diff}_mouth={mouth_open}_ebh={eb_eye_high}_ebl={eb_eye_low}_sharplow={sharpness_low}_sharphigh={sharpness_high}_fsl={face_size_low}_fsh={face_size_high}_retouch={retouching}_mask={mask}_glasses={glasses}_gray={grayscale}{lr_tag}_cnt={file_count}"
    
    cache = load_cache()
    if cache_key in cache:
        cached = cache[cache_key]
        # キャッシュには(raw_score, total_images, filtered_count)を保存
        if isinstance(cached, (list, tuple)) and len(cached) >= 2:
            if len(cached) == 3:
                raw_score, total_images, filtered_count = cached
            else:
                # 旧形式(raw_score, filtering_rate)の互換性
                raw_score, filtering_rate = cached
                total_images = file_count
                filtered_count = int(total_images * filtering_rate)
            logger.info(f"Cache Hit! RawScore={raw_score:.4f}, Total={total_images}, Filtered={filtered_count}")
            return (raw_score, total_images, filtered_count)
        else:
            # 古い形式のキャッシュ（互換性のため）
            logger.info(f"Cache Hit (legacy format)! Score: {cached}")
            return (cached, file_count, 0)

    try:
        # Preprocess
        cmd_pre = [
            PYTHON_PREPROCESS,
            "preprocess_multitask.py",
            "--out_dir", "preprocessed_multitask",
            "--pitch_percentile", str(pitch),
            "--symmetry_percentile", str(sym),
            "--y_diff_percentile", str(y_diff),
            "--mouth_open_percentile", str(mouth_open),
            "--eyebrow_eye_percentile_high", str(eb_eye_high),
            "--eyebrow_eye_percentile_low", str(eb_eye_low),
            "--sharpness_percentile_low", str(sharpness_low),
            "--sharpness_percentile_high", str(sharpness_high),
            "--face_size_percentile_low", str(face_size_low),
            "--face_size_percentile_high", str(face_size_high),
            "--retouching_percentile", str(retouching),
            "--mask_percentile", str(mask),
            "--glasses_percentile", str(glasses)
        ]
        if grayscale:
            cmd_pre.append("--grayscale")
        logger.info("Running preprocessing...")
        ret_pre = subprocess.run(cmd_pre, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret_pre.stdout:
            logger.info(f"Preprocessing STDOUT:\n{ret_pre.stdout}")
        if ret_pre.stderr:
            logger.warning(f"Preprocessing STDERR:\n{ret_pre.stderr}")

        if ret_pre.returncode != 0:
            logger.error(f"Preprocessing failed with return code {ret_pre.returncode}")
            return (0.0, 0, 0)

        # フィルタリング枚数を計算（前処理の出力からTotal/Savedを取得）
        # logger.infoはstderrに出力されるため、stdout + stderr の両方を検索
        preprocess_output = (ret_pre.stdout or "") + "\n" + (ret_pre.stderr or "")
        total_images = 0
        saved_images = 0
        for line in preprocess_output.split('\n'):
            # "Processed {src_root}: Total={total}, Saved={saved_count}, Skipped={skipped_count}"
            match_stats = re.search(r'Total=(\d+), Saved=(\d+)', line)
            if match_stats:
                total_images += int(match_stats.group(1))
                saved_images += int(match_stats.group(2))
        
        filtered_count = total_images - saved_images
        if total_images > 0:
            logger.info(f"Filtering Stats: Total={total_images}, Saved={saved_images}, Filtered={filtered_count} ({filtered_count/total_images*100:.1f}%)")

        # Train with specific Model and Optimized Params
        train_script = "components/train_multitask_trial.py"
        cmd_train = [PYTHON_TRAIN, train_script, "--model_name", model_name]
        
        # Load best params and append to command
        # epochs, fine_tune, model_name, learning_rate は明示的に設定するので除外
        best_params = load_best_train_params()
        for k, v in best_params.items():
            if k not in ['model_name', 'fine_tune', 'epochs', 'learning_rate', 'auto_lr_target_epoch']:
                cmd_train.extend([f"--{k}", str(v)])
        
        # 学習率の設定 (2026-02-14 改良):
        # ratio >= threshold -> exp1
        # ratio < threshold  -> exp2
        if CALIBRATED_BASE_LR is not None and total_images > 0 and saved_images > 0:
            ratio = saved_images / total_images
            safe_ratio = max(ratio, 0.001)
            
            # Use relative ratio (ratio / BASE_RATIO)
            relative_ratio = safe_ratio / BASE_RATIO if BASE_RATIO > 0 else safe_ratio

            # Dynamic Exponent Logic (2nd order polynomial, 2026-02-16)
            # y = 0.8491836 - 1.638353*x + 1.833087*x^2
            # x = relative_ratio
            x = relative_ratio
            exponent = 0.8491836 - 1.638353 * x + 1.833087 * (x**2)
            
            # Apply min(1.0, exponent) as per user request
            exponent = min(1.0, exponent)

            # Keep safety range [0.1, 2.0] just in case (though min(1.0) makes upper mostly redundant)
            exponent = max(0.1, min(exponent, 2.0))
            
            # adjusted_lr = base_lr * (ratio ** exp)
            scale_factor = relative_ratio ** exponent
            adjusted_lr = CALIBRATED_BASE_LR * scale_factor
            
            logger.info(
                f"LR Scaling (Dynamic): Base={CALIBRATED_BASE_LR:.8f}, Ratio={safe_ratio:.4f} (BaseRatio={BASE_RATIO:.4f}), "
                f"RelRatio={relative_ratio:.4f} -> Exp={exponent:.4f} -> LR={adjusted_lr:.8f}"
            )
        elif CALIBRATED_BASE_LR is not None:
            adjusted_lr = CALIBRATED_BASE_LR
            logger.info(f"LR (Calibrated): {adjusted_lr:.8f}")
        else:
            adjusted_lr = best_params.get('learning_rate', 0.0001)
            logger.info(f"LR (Fallback): {adjusted_lr:.8f}")
        
        cmd_train.extend(["--learning_rate", str(adjusted_lr)])
        
        # 評価用なのでFine-tuningはOff、Epochs=20
        cmd_train.extend(["--epochs", "20"])
        cmd_train.extend(["--fine_tune", "False"])
        cmd_train.extend(["--auto_lr_target_epoch", "0"]) # 内部自動スケーリングを無効化

        logger.info(f"Running training with {model_name} (epochs=20, fine_tune=False)...")
        
        # Popenでリアルタイム出力 + スコア抽出
        process = subprocess.Popen(
            cmd_train, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace'
        )
        
        train_output = []
        for line in process.stdout:
            line = line.rstrip()
            train_output.append(line)
            # エポック情報やスコアを含む行のみ表示
            if any(kw in line for kw in ['Epoch ', 'FINAL_VAL_ACCURACY', 'TASK_', 'MinClassAcc', 'Avg=']):
                logger.info(f"  [Train] {line}")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Training failed (returncode={process.returncode})")
            return (0.0, 0, 0)
        
        full_output = "\n".join(train_output)

        # Extract Score
        match = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
        if match:
            raw_score = float(match.group(1))
            
            # 各タスクのスコアを抽出してログに出力
            for char_code in range(ord('A'), ord('Z') + 1):
                task_label = chr(char_code)
                match_task = re.search(f"TASK_{task_label}_ACCURACY:\s*([\d\.]+)", full_output)
                if match_task:
                    logger.info(f"  Task {task_label}: {float(match_task.group(1)):.4f}")

            # 詳細なクラス別精度をログに転記
            details_match = re.search(r"--- Task [A-Z] Details.*", full_output, re.DOTALL)
            if details_match:
                logger.info("\n[Detailed Class Accuracy]")
                logger.info(details_match.group(0).strip())

            logger.info(f"Result: RawScore={raw_score:.4f} (Average), Total={total_images}, Filtered={filtered_count}")
            
            # キャッシュに保存
            cache = load_cache()
            cache[cache_key] = (raw_score, total_images, filtered_count)
            save_cache(cache)
            return (raw_score, total_images, filtered_count)
        else:
            logger.error("Score not found in training output.")
            logger.error(f"STDOUT:\n{full_output}")
            return (0.0, 0, 0)

    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return (0.0, 0, 0)

def optimize_single_param(target_name, current_params, model_name, baseline_score, baseline_filtered, points=[0, 2, 5, 25, 50]):
    """
    1つのパラメータを最適化する。
    
    Returns:
        tuple: (best_val, best_score, improvement, filtered_diff, efficiency)
        - best_val: 最適なパラメータ値
        - best_score: 最高スコア
        - improvement: ベースラインからの精度向上
        - filtered_diff: ベースラインからのフィルタリング枚数増加
        - efficiency: 効率 = improvement / (filtered_diff + 1)
    """
    logger.info(f"\n>>> Optimizing {target_name} [Model: {model_name}] <<<")
    logger.info(f"Baseline: Score={baseline_score:.4f}, Filtered={baseline_filtered}")
    
    def evaluate_wrapper(val):
        # Independent Optimization: Always start from all-zero params
        test_params = {
            'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
            'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0, 'sharpness_high': 0,
            'face_size_low': 0, 'face_size_high': 0, 'retouching': 0, 'mask': 0, 'glasses': 0
        }
        test_params[target_name] = val
        
        # run_trialは(raw_score, total_images, filtered_count)を返す
        result = run_trial(
            test_params['pitch'], test_params['sym'], test_params['y_diff'], test_params['mouth_open'],
            test_params['eb_eye_high'], test_params['eb_eye_low'],
            test_params['sharpness_low'], test_params['sharpness_high'],
            test_params['face_size_low'], test_params['face_size_high'],
            test_params['retouching'], test_params['mask'], test_params['glasses'],
            model_name=model_name
        )
        return result  # (raw_score, total_images, filtered_count)

    best_val = 0
    best_score = baseline_score
    best_filtered = baseline_filtered
    
    # 各探索点のスコアを記録（二分探索用）
    scores = {}
    
    for p in points:
        logger.info(f"Testing {target_name}={p}...")
        raw_score, total_images, filtered_count = evaluate_wrapper(p)
        logger.info(f"  {target_name}={p} -> Score: {raw_score:.4f}, Filtered: {filtered_count}")
        scores[p] = (raw_score, total_images, filtered_count)
        
        if raw_score > best_score:
            best_score = raw_score
            best_val = p
            best_filtered = filtered_count
            logger.info(f"  [NEW BEST] {target_name}={p} (Score: {raw_score:.4f})")
    
    # --- 二分探索 Refinement ---
    # --- 二分探索 Refinement (1%単位まで探索) ---
    logger.info("  [Refinement] Starting binary search refinement to 1% resolution...")
    
    # 既存のスコア辞書 (scores) を使用して探索を継続
    # 上位2点間を埋めていく方針
    
    refinement_iter = 0
    MAX_REFINEMENT_ITER = 20 # 0-50の範囲なら十分収束する
    
    while refinement_iter < MAX_REFINEMENT_ITER:
        # スコア順にソート (降順)
        sorted_items = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        if len(sorted_items) < 2:
            break
            
        top1_val, top1_data = sorted_items[0]
        top1_score = top1_data[0]
        
        top2_val, top2_data = sorted_items[1]
        
        # 1位と2位の差が1以下なら終了 (これ以上探索できない)
        diff = abs(top1_val - top2_val)
        if diff <= 1:
            logger.info(f"  [Refinement] Converged (diff={diff}). Best={top1_val}, 2nd={top2_val}")
            break
            
        # 中間点を計算
        mid_val = (top1_val + top2_val) // 2
        
        # 既に評価済みなら、周辺を探索し尽くしている可能性が高い
        if mid_val in scores:
            # 中間が既に評価済みの場合、上位2点の間にはピークがないか、平坦
            # 念のため、1位のもう一方の隣(もしあれば)との間もチェックすべきだが、
            # 単純化のため、ここでは探索終了とする（あるいはランダムに振る）
            
            # 特例: diff > 1 なのに mid が key にある -> つまり (top1+top2)//2 が既に評価済み
            # 例: top1=0, top2=2 -> mid=1. evaluated.
            # 例: top1=0, top2=5 -> mid=2. evaluated.
            # この場合、これ以上分割しても新しい整数が出ないなら終了
            
            # まだ埋まっていない場所を探す (Lower bound logic)
            # top1とtop2の間で、まだ評価していない点を探す
            low, high = min(top1_val, top2_val), max(top1_val, top2_val)
            found_new = False
            # 単純に mid-1, mid+1 などを試すか、このループを抜ける
            # 今回はシンプルに、(low + high) // 2 が済みなら終了
            logger.info(f"  [Refinement] Midpoint {mid_val} already evaluated. Stopping refinement.")
            break
            
        logger.info(f"  [Refinement #{refinement_iter+1}] Testing mid={mid_val} (between {top1_val} and {top2_val})...")
        raw_score, total_images, filtered_count = evaluate_wrapper(mid_val)
        logger.info(f"  {target_name}={mid_val} -> Score: {raw_score:.4f}, Filtered: {filtered_count}")
        scores[mid_val] = (raw_score, total_images, filtered_count)
        
        if raw_score > best_score:
            best_score = raw_score
            best_val = mid_val
            best_filtered = filtered_count
            logger.info(f"  [REFINED BEST] {target_name}={mid_val} (Score: {raw_score:.4f})")
            
        refinement_iter += 1
    
    # --- Collect Candidates (Best Score & Best Efficiency) ---
    # すべての探索点（Refinement含む）から候補を選定
    best_score_candidate = None
    best_eff_candidate = None
    max_score_val = -1.0
    max_eff_val = -float('inf')
    
    for p, res in scores.items():
        r_score, r_total, r_filtered = res
        
        # Improvement / Efficiency vs Baseline
        imp = r_score - baseline_score
        f_diff = r_filtered - baseline_filtered
        eff = imp / (f_diff + 1) if f_diff >= 0 else imp # Safety
        
        # Valid only if improvement is positive (or at least non-negative?)
        # User constraint: "score上昇" -> improvement > 0
        if imp > 0:
            # Update Best Score Candidate
            if r_score > max_score_val:
                max_score_val = r_score
                best_score_candidate = {
                    'val': p, 'score': r_score, 'improvement': imp, 
                    'filtered': r_filtered, 'efficiency': eff
                }
            
            # Update Best Efficiency Candidate
            if eff > max_eff_val:
                max_eff_val = eff
                best_eff_candidate = {
                    'val': p, 'score': r_score, 'improvement': imp, 
                    'filtered': r_filtered, 'efficiency': eff
                }

    logger.info(f"Finished optimizing {target_name}.")
    
    candidates = []
    if best_score_candidate:
        candidates.append(best_score_candidate)
        logger.info(f"  Best Score Param: {best_score_candidate['val']} (Score={best_score_candidate['score']:.4f}, Eff={best_score_candidate['efficiency']:.6f})")
        
    if best_eff_candidate:
        # Avoid duplicate object if same point is both best score and best eff
        if not best_score_candidate or best_eff_candidate['val'] != best_score_candidate['val']:
            candidates.append(best_eff_candidate)
            logger.info(f"  Best Eff Param:   {best_eff_candidate['val']} (Score={best_eff_candidate['score']:.4f}, Eff={best_eff_candidate['efficiency']:.6f})")

    if not candidates:
        logger.info("  No valid candidates found (no improvement).")

    return candidates

def main():
    global CALIBRATED_BASE_LR, LR_LOW_RATIO_THRESHOLD, LR_SCALING_EXP1, LR_SCALING_EXP2, BASE_RATIO
    logger.info("Starting Sequential Optimization (Efficiency-Based)")

    # LRスケーリング設定（calibrate_lr_scaling.py の結果）を読み込む
    cal_score = 0.0
    if os.path.exists(LR_SCALING_CONFIG_FILE):
        try:
            with open(LR_SCALING_CONFIG_FILE, 'r', encoding='utf-8') as f:
                lr_config = json.load(f)
            LR_LOW_RATIO_THRESHOLD = float(lr_config.get('threshold', LR_LOW_RATIO_THRESHOLD))
            BASE_RATIO = float(lr_config.get('base_ratio', 1.0))
            
            if 'base_lr' in lr_config:
                CALIBRATED_BASE_LR = float(lr_config['base_lr'])
                logger.info(f"Loaded CALIBRATED_BASE_LR from config: {CALIBRATED_BASE_LR:.8f}")
            
            if 'score' in lr_config:
                cal_score = float(lr_config['score'])
                logger.info(f"Loaded Base Score from config: {cal_score:.4f}")

            # New Keys: exp1, exp2 (Deprecated/Unused with dynamic formula but loaded for log)
            if 'exp1' in lr_config:
                LR_SCALING_EXP1 = float(lr_config['exp1'])
            if 'exp2' in lr_config:
                LR_SCALING_EXP2 = float(lr_config['exp2'])
            
            logger.info(
                f"Loaded LR scaling config: threshold={LR_LOW_RATIO_THRESHOLD:.2f}, "
                f"base_ratio={BASE_RATIO:.4f}, base_lr={CALIBRATED_BASE_LR}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load {LR_SCALING_CONFIG_FILE}: {e}. Using defaults."
            )
    else:
        logger.info(
            f"LR scaling config {LR_SCALING_CONFIG_FILE} not found. Using defaults."
        )
    
    # Initial Params
    current_params = {
        'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
        'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0, 'sharpness_high': 0,
        'face_size_low': 0, 'face_size_high': 0, 'retouching': 0, 'mask': 0, 'glasses': 0
    }
    
    # 効率情報を記録する辞書
    param_efficiency = {}
    
    # --- LR Calibration: epoch中間でベストになるbase LRを決定 ---
    # 設定ファイルから読み込めなかった場合のみ実行
    if CALIBRATED_BASE_LR is None:
        logger.info("\n>>> LR Calibration: Finding optimal base learning rate (Fallback) <<<")
        
        # まずフィルタなしで前処理
        cmd_pre_cal = [
            PYTHON_PREPROCESS, "preprocess_multitask.py",
            "--out_dir", "preprocessed_multitask",
            "--pitch_percentile", "0", "--symmetry_percentile", "0",
            "--y_diff_percentile", "0", "--mouth_open_percentile", "0",
            "--eyebrow_eye_percentile_high", "0", "--eyebrow_eye_percentile_low", "0",
            "--sharpness_percentile_low", "0", "--sharpness_percentile_high", "0",
            "--face_size_percentile_low", "0", "--face_size_percentile_high", "0",
            "--retouching_percentile", "0", "--mask_percentile", "0",
            "--glasses_percentile", "0"
        ]
        logger.info("Preprocessing for LR calibration (no filters)...")
        subprocess.run(cmd_pre_cal, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        # LRキャリブレーション実行（20 epoch、epoch 10でベストをターゲット）
        best_params = load_best_train_params()
        initial_lr = best_params.get('learning_rate', 0.0001)
        # Note: calibrate_base_lr function must be defined or imported. Assuming it exists based on lines 580-585.
        # If it's imported from somewhere, fine. If it's not defined, this will error. 
        # But since user showed log running it, it must be there.
        CALIBRATED_BASE_LR, cal_score = calibrate_base_lr(
            'EfficientNetV2B0',
            initial_lr,
            cal_epochs=20,
            target_best_epoch=10,
        )
        logger.info(f"Using Calibrated Base LR: {CALIBRATED_BASE_LR:.8f}")
    else:
        logger.info(f"Skipping Re-Calibration. Using Base LR: {CALIBRATED_BASE_LR:.8f}")
    
    # --- Step 0: Model Architecture Selection & Baseline ---
    # キャリブレーション最終結果をB0のベースラインとして流用
    logger.info("\n>>> Step 0: Model Architecture Selection & Baseline <<<")
    best_model = 'EfficientNetV2B0'
    best_model_score = cal_score
    baseline_filtered = 0  # キャリブレーションはフィルタなし
    logger.info(f"B0 Baseline from calibration: Score={cal_score:.4f}")
    
    # 他のモデル候補をテスト
    other_models = ['EfficientNetV2S']
    for m in other_models:
        logger.info(f"Testing Model: {m}")
        raw_score, total_images, filtered_count = run_trial(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, model_name=m)
        if raw_score > best_model_score:
            best_model_score = raw_score
            best_model = m
            baseline_filtered = filtered_count
            
    logger.info(f"Best Model Selected: {best_model}")
    logger.info(f"Baseline: Score={best_model_score:.4f}, Filtered={baseline_filtered}")
    
    baseline_score = best_model_score
    
    # --- Parameter Optimization (recording efficiency) ---
    param_names = [
        'pitch', 'sym', 'y_diff', 'mouth_open',
        'eb_eye_high', 'eb_eye_low', 'sharpness_low', 'sharpness_high',
        'face_size_low', 'face_size_high', 'retouching', 'mask', 'glasses'
    ]
    
    # Phase 1 loop
    # Phase 1 loop
    global_best_score = baseline_score
    global_best_params = {k: 0 for k in current_params}
    global_best_desc = "Baseline (No Filter)"
    
    # 全候補リスト (Best Score, Best Eff mixed)
    all_candidates = []

    for param_name in param_names:
        # Returns list of dicts: [{'val':..., 'score':..., 'efficiency':...}, ...]
        candidates = optimize_single_param(
            param_name, current_params, best_model, baseline_score, baseline_filtered
        )
        
        for cand in candidates:
            cand['param_name'] = param_name
            all_candidates.append(cand)
            
            # Check for Global Single Best
            if cand['score'] > global_best_score:
                global_best_score = cand['score']
                global_best_params = {k: 0 for k in current_params}
                global_best_params[param_name] = cand['val']
                global_best_desc = f"Single Best ({param_name}={cand['val']})"
                logger.info(f"  [Global Best Update] New best found in single trial: {global_best_desc}, Score: {cand['score']:.4f}")

    # Phase 2: Efficiency-Based Greedy Integration (Grayscale無しで実行)
    logger.info("\n>>> Phase 2: Efficiency-Based Greedy Integration <<<")
    
    # 効率順にソート (降順)
    # User Requirement: "score上昇かつ、効率の良い方をgreedyで採用"
    # -> 効率順に試行し、スコアが上がるなら採用 (Greedy)
    sorted_candidates = sorted(all_candidates, key=lambda x: x['efficiency'], reverse=True)
    
    current_greedy_params = {k: 0 for k in current_params}
    
    # ベースライン（Grayscale=False）のスコア計測
    base_res = run_trial(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        grayscale=False, model_name=best_model
    )
    current_best_score = base_res[0]
    logger.info(f"Base Score (No filters): {current_best_score:.4f}")
    
    greedy_history = []
    
    for cand in sorted_candidates:
        p_name = cand['param_name']
        p_val = cand['val']
        p_eff = cand['efficiency']
        p_score_single = cand['score']
        
        logger.info(f"Trying candidate: {p_name}={p_val} (Eff: {p_eff:.6f}, SingleScore: {p_score_single:.4f})...")
        
        # 既にセットされている値がある場合、それと比較して更新するか判断
        prev_val = current_greedy_params[p_name]
        
        if prev_val == p_val:
            logger.info(f"  -> Already set to {p_val}. Skipping.")
            continue
            
        # 一時的にパラメータ適用
        temp_params = current_greedy_params.copy()
        temp_params[p_name] = p_val
        
        res = run_trial(
            temp_params['pitch'], temp_params['sym'], temp_params['y_diff'], temp_params['mouth_open'],
            temp_params['eb_eye_high'], temp_params['eb_eye_low'],
            temp_params['sharpness_low'], temp_params['sharpness_high'],
            temp_params['face_size_low'], temp_params['face_size_high'],
            temp_params['retouching'], temp_params['mask'], temp_params['glasses'],
            grayscale=False, model_name=best_model
        )
        score, total, filtered = res
        
        # 採用基準: スコアが現状以上 (>=) なら採用
        # (効率が良い順に試しているので、同じスコアなら効率が良い方が先に採用されているはずだが、
        #  後から来た「効率は低いがスコアが高い候補」がさらにスコアを上げるなら更新する)
        if score >= current_best_score:
            diff = score - current_best_score
            logger.info(f"  -> Accepted (Score: {score:.4f} >= {current_best_score:.4f}, Diff: +{diff:.4f})")
            current_best_score = score
            current_greedy_params[p_name] = p_val
            greedy_history.append((p_name, p_val, score))
        else:
            logger.info(f"  -> Skipped (Score: {score:.4f} < {current_best_score:.4f})")
            
    greedy_score = current_best_score
    greedy_params = current_greedy_params.copy()

    # 元のパラメータでもスコアを確認（比較用、Grayscale無し）
    logger.info("\n--- Testing Original Parameters (no scaling, no grayscale) ---")
    original_result = run_trial(
        current_params['pitch'], current_params['sym'], current_params['y_diff'], current_params['mouth_open'],
        current_params['eb_eye_high'], current_params['eb_eye_low'],
        current_params['sharpness_low'], current_params['sharpness_high'],
        current_params['face_size_low'], current_params['face_size_high'],
        current_params['retouching'], current_params['mask'], current_params['glasses'],
        grayscale=False, model_name=best_model
    )
    original_score, original_total, original_filtered = original_result

    # --- Final Selection Phase (Grayscale適用前) ---
    logger.info("\n>>> Final Selection Phase (without Grayscale) <<<")
    
    candidates = []
    
    # Greedy Integration
    candidates.append({'name': "Greedy Integration", 'params': greedy_params, 'score': greedy_score})
    logger.info(f"Strategy: {'Greedy':<15} Score={greedy_score:.4f}")
    
    # Original (all Phase 1 params applied)
    candidates.append({
        'name': "Original (No Scale)",
        'params': {k: v for k, v in current_params.items()},
        'score': original_score,
    })
    logger.info(f"Strategy: {'Original':<15} Score={original_score:.4f}")
    
    # Single Best
    logger.info(f"\n--- Testing Single Best ({global_best_desc}) ---")
    sb_res = run_trial(
        global_best_params['pitch'], global_best_params['sym'], global_best_params['y_diff'], global_best_params['mouth_open'],
        global_best_params['eb_eye_high'], global_best_params['eb_eye_low'],
        global_best_params['sharpness_low'], global_best_params['sharpness_high'],
        global_best_params['face_size_low'], global_best_params['face_size_high'],
        global_best_params['retouching'], global_best_params['mask'], global_best_params['glasses'],
        grayscale=False, model_name=best_model
    )
    sb_score = sb_res[0]
    candidates.append({
        'name': f"Single Best ({global_best_desc})",
        'params': global_best_params.copy(),
        'score': sb_score,
    })
    logger.info(f"Strategy: {'Single Best':<15} Score={sb_score:.4f}")

    # ベストを選択 (Score最大)
    best_candidate = max(candidates, key=lambda x: x['score'])
    logger.info(f"-> Selected: {best_candidate['name']} (Score: {best_candidate['score']:.4f})")
    
    final_params = best_candidate['params'].copy()
    final_score = best_candidate['score']
    final_desc = best_candidate['name']
    
    # --- Grayscale Test (最終ベストに対してのみ) ---
    logger.info("\n>>> Grayscale Test (on final best) <<<")
    
    res_color = run_trial(
        final_params['pitch'], final_params['sym'], final_params['y_diff'], final_params['mouth_open'],
        final_params['eb_eye_high'], final_params['eb_eye_low'],
        final_params['sharpness_low'], final_params['sharpness_high'],
        final_params['face_size_low'], final_params['face_size_high'],
        final_params['retouching'], final_params['mask'], final_params['glasses'],
        grayscale=False, model_name=best_model
    )
    res_gray = run_trial(
        final_params['pitch'], final_params['sym'], final_params['y_diff'], final_params['mouth_open'],
        final_params['eb_eye_high'], final_params['eb_eye_low'],
        final_params['sharpness_low'], final_params['sharpness_high'],
        final_params['face_size_low'], final_params['face_size_high'],
        final_params['retouching'], final_params['mask'], final_params['glasses'],
        grayscale=True, model_name=best_model
    )
    
    score_color, score_gray = res_color[0], res_gray[0]
    
    if score_gray > score_color:
        final_params['grayscale'] = True
        final_score = score_gray
        logger.info(f"Grayscale selected ({score_gray:.4f} > {score_color:.4f})")
    else:
        final_params['grayscale'] = False
        final_score = score_color
        logger.info(f"Color selected ({score_color:.4f} >= {score_gray:.4f})")

    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Selected Strategy: {final_desc}")
    logger.info(f"Final Best Params: {final_params}")
    logger.info(f"Final Best Model: {best_model}")
    logger.info(f"Final Score: {final_score:.4f}")
    logger.info(f"Baseline Score: {baseline_score:.4f}")
    logger.info(f"Total Improvement: +{final_score - baseline_score:.4f}")
    logger.info("="*50)

    # 結果を自動記録
    from components.result_logger import log_result
    log_result("optimize_sequential", {
        "baseline_score": round(baseline_score, 4),
        "best_score": round(final_score, 4),
        "improvement": round(final_score - baseline_score, 4),
        "strategy": final_desc,
        "model": best_model,
        "filter_params": {k: v for k, v in final_params.items() if k != 'grayscale'},
        "grayscale": final_params.get('grayscale', False),
        "param_efficiency": {k: {ek: round(ev, 6) if isinstance(ev, float) else ev for ek, ev in v.items()} for k, v in param_efficiency.items()},
    })

    # Generate and print the final preprocessing command
    final_cmd = (
        f"{PYTHON_PREPROCESS} preprocess_multitask.py --out_dir preprocessed_multitask "
        f"--pitch_percentile {final_params['pitch']} "
        f"--symmetry_percentile {final_params['sym']} "
        f"--y_diff_percentile {final_params['y_diff']} "
        f"--mouth_open_percentile {final_params['mouth_open']} "
        f"--eyebrow_eye_percentile_high {final_params['eb_eye_high']} "
        f"--eyebrow_eye_percentile_low {final_params['eb_eye_low']} "
        f"--sharpness_percentile_low {final_params['sharpness_low']} "
        f"--sharpness_percentile_high {final_params['sharpness_high']} "
        f"--face_size_percentile_low {final_params['face_size_low']} "
        f"--face_size_percentile_high {final_params['face_size_high']} "
        f"--retouching_percentile {final_params['retouching']} "
        f"--mask_percentile {final_params['mask']} "
        f"--glasses_percentile {final_params['glasses']} "
    )
    if final_params.get('grayscale', False):
        final_cmd += "--grayscale "
    logger.info("\nRun this command to apply the best filters:")
    logger.info(final_cmd)
    logger.info(f"\nRecommended Model for Training: {best_model}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
    # 処理完了通知音
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
