"""
LRスケーリング指数キャリブレーション

フィルタによりデータ量が変わった際の学習率調整:
  adjusted_lr = base_lr / (t * (ratio / t) ^ exponent)
    - t: low_ratio_threshold (default 0.25)
    - exponent: 0.3〜0.8 で探索

各フィルタレベル (0/5/40/80%) で LR を epoch 10 でベストになるようキャリブレーションし、
生スコアを epoch差で減衰した実効スコアが最も高い exponent を採用する:
  weighted_score = raw_score / ((abs(best_epoch - target_epoch) + 1) ** exponent)

探索範囲は exponent = 0.3〜0.8 を二分探索する。

結果は outputs/lr_scaling_config.json に保存され、
optimize_sequential.py が読み込んで使用する。
"""
import subprocess
import sys
import re
import os
import logging
import json
import winsound

# --- 設定 ---
PYTHON_PREPROCESS = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe"
PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
DATA_SOURCE_DIR = "train"
BEST_TRAIN_PARAMS_FILE = "outputs/best_train_params.json"
OUTPUT_FILE = "outputs/lr_scaling_config.json"
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'lr_scaling_calibration.txt'), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
LOW_RATIO_THRESHOLD = 0.5


def count_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        count += len(files)
    return count


def load_best_train_params():
    if os.path.exists(BEST_TRAIN_PARAMS_FILE):
        try:
            with open(BEST_TRAIN_PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load best train params: {e}")
    return {}


def run_preprocess(filter_percentile, param_name="y_diff_percentile"):
    """
    指定パラメータでフィルタリングを適用して前処理を実行する。
    """
    cmd = [
        PYTHON_PREPROCESS,
        "preprocess_multitask.py",
        "--out_dir", "preprocessed_multitask",
        f"--{param_name}", str(filter_percentile),
    ]

    logger.info(f"Running preprocessing (y_diff_filter={filter_percentile}%)...")
    ret = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    if ret.returncode != 0:
        logger.error(f"Preprocessing failed: {ret.stderr}")
        return 0, 0

    output = (ret.stdout or "") + "\n" + (ret.stderr or "")
    total_images = 0
    saved_images = 0
    for line in output.split('\n'):
        match = re.search(r'Total=(\d+), Saved=(\d+)', line)
        if match:
            total_images += int(match.group(1))
            saved_images += int(match.group(2))

    # Ratioは「残存率」（実際に学習に使用されるデータの割合）
    # filter指定値とは必ずしも一致しない（分布や上下カットの影響）
    ratio = saved_images / total_images if total_images > 0 else 1.0
    logger.info(f"  Total={total_images}, Saved={saved_images}, Ratio={ratio:.4f}")
    return total_images, saved_images


def run_training_trial(lr, model_name='EfficientNetV2B0', epochs=20, target_epoch=0):
    """指定LRで学習を実行してスコアとbest_epochを返す"""
    best_params = load_best_train_params()

    cmd = [PYTHON_TRAIN, "components/train_multitask_trial.py", "--model_name", model_name]
    for k, v in best_params.items():
        if k not in ['model_name', 'fine_tune', 'epochs', 'learning_rate', 'auto_lr_target_epoch', 'seed']:
            cmd.extend([f"--{k}", str(v)])
    cmd.extend(["--learning_rate", str(lr)])
    cmd.extend(["--epochs", str(epochs)])
    cmd.extend(["--fine_tune", "False"])

    # 内部自動スケーリングを無効化（このスクリプトで制御するため）
    cmd.extend(["--auto_lr_target_epoch", "0"]) 
    # if target_epoch > 0:
    #     cmd.extend(["--auto_lr_target_epoch", str(target_epoch)])

    logger.info(f"  Training with LR={lr:.8f} (target_epoch={target_epoch})...")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )

    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if any(kw in line for kw in ['Epoch ', 'FINAL_VAL_ACCURACY', 'BEST_EPOCH', 'MinClassAcc']):
            logger.info(f"    [Train] {line}")

    process.wait()
    full_output = "\n".join(output_lines)

    match_score = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
    score = float(match_score.group(1)) if match_score else 0.0

    match_epoch = re.search(r"BEST_EPOCH:\s*(\d+)", full_output, re.IGNORECASE)
    if match_epoch:
        best_epoch = int(match_epoch.group(1))
    else:
        logger.warning("Could not find BEST_EPOCH in training output! Defaulting to max epochs.")
        best_epoch = epochs

    logger.info(f"  Score: {score:.4f}, BestEpoch: {best_epoch}/{epochs}")
    return score, best_epoch


def distance_weighted_score(raw_score, best_epoch, target_epoch, exponent):
    """best_epoch が target_epoch に近いほど高くなる重み付きスコア"""
    distance = abs(best_epoch - target_epoch)
    return raw_score / ((distance + 1.0) ** exponent)


def scale_lr_by_ratio(base_lr, ratio, exp1, exp2, threshold=0.5, base_ratio=1.0):
    """
    残データ率 ratio に基づくLR補正:
      ratio >= threshold: adjusted_lr = base_lr * (relative_ratio ^ exp1)
      ratio < threshold : adjusted_lr = base_lr * (relative_ratio ^ exp2)
      (relative_ratio = ratio / base_ratio)
    """
    safe_ratio = max(ratio, 0.001)
    
    if safe_ratio >= threshold:
        exponent = exp1
        mode = f"High(exp1={exp1:.4f})"
    else:
        exponent = exp2
        mode = f"Low(exp2={exp2:.4f})"
        
    # relative_ratio = safe_ratio / base_ratio (基準に対して何割か)
    relative_ratio = safe_ratio / base_ratio if base_ratio > 0 else safe_ratio
    # relative_ratio < 1.0 -> LR increases (division)
    adjusted_lr = base_lr * (relative_ratio ** exponent)
    
    return adjusted_lr, safe_ratio, exponent, mode


def calibrate_lr_for_target(initial_lr, target_epoch=10, model_name='EfficientNetV2B0',
                            cal_epochs=20, max_iter=3, distance_exponent=0.5,
                            strict_target=False):
    """
    target_epochでベストになるLRをキャリブレーションし、(lr, score, best_epoch)を返す。
    strict_target=False: 候補選択は epoch差を加味した重み付きスコアを優先する。
    strict_target=True : epoch差を最優先し、target_epoch一致まで調整を継続する。
    """
    current_lr = initial_lr
    best_candidate = None
    reached_target = False

    for i in range(max_iter):
        # run_training_trial 内で --enable_early_stopping False が指定されていることを想定
        score, best_epoch = run_training_trial(
            current_lr, model_name, cal_epochs, target_epoch=target_epoch
        )
        distance = abs(best_epoch - target_epoch)
        weighted = distance_weighted_score(score, best_epoch, target_epoch, distance_exponent)
        
        if strict_target:
            candidate = (distance, -score, -weighted, current_lr, best_epoch, score, weighted)
        else:
            candidate = (-weighted, distance, -score, current_lr, best_epoch, score, weighted)
            
        if best_candidate is None or candidate < best_candidate:
            best_candidate = candidate

        logger.info(
            f"  [Cal #{i+1}] LR={current_lr:.8f}, BestEpoch={best_epoch}/{cal_epochs}, "
            f"Score={score:.4f}, WeightedScore={weighted:.4f}"
        )

        if best_epoch == target_epoch:
            reached_target = True
            break

        # LRスケーリング: epoch比のべき指数
        raw_scale = best_epoch / target_epoch
        scale = raw_scale ** 0.75 if raw_scale > 0 else 0.5
        scale = max(0.5, min(scale, 2.0))
        new_lr = current_lr * scale
        current_lr = new_lr

    _, _, _, chosen_lr, chosen_epoch, chosen_score, chosen_weighted_score = best_candidate
    logger.info(
        f"  Calibrated: LR={chosen_lr:.8f}, BestEpoch={chosen_epoch}, "
        f"Score={chosen_score:.4f}, WeightedScore={chosen_weighted_score:.4f}"
    )
    return chosen_lr, chosen_score, chosen_epoch





def main():
    logger.info("=" * 60)
    logger.info("LR Scaling Exponent Calibration (Range Search: Epoch 10-15)")
    logger.info("=" * 60)

    model_name = 'EfficientNetV2B0'
    target_epoch = 10 
    cal_epochs = 20
    
    # ユーザー指定の探索範囲
    range_exp1 = (0.15, 1.0)  # High Ratio (>= threshold)
    range_exp2 = (0.15, 1.0)  # Low Ratio (< threshold)
    threshold = 0.5

    # --- Step 1: ベースライン（フィルタなし）---
    logger.info("\n>>> Step 1: Baseline (no filter, optimize LR for Epoch 10) <<<")
    best_params = load_best_train_params()
    initial_lr = best_params.get('learning_rate', 0.0001)

    total_base, saved_base = run_preprocess(0)
    base_ratio = saved_base / total_base if total_base > 0 else 1.0
    
    # 毎回キャリブレーション実行（キャッシュなし）
    logger.info(">>> Finding Base LR for Target Epoch 10...")
    base_lr, base_score, base_epoch = calibrate_lr_for_target(
        initial_lr, target_epoch=10, model_name=model_name, 
        cal_epochs=cal_epochs, max_iter=5, strict_target=True
    )


    def optimize_exponent_for_levels(levels_subset, exp_range, param_name, initial_points=None):
        if not levels_subset:
            logger.info(f"No levels for {param_name}. Skipping optimization, returning range midpoint.")
            return (exp_range[0] + exp_range[1]) / 2, 0.0

        if len(levels_subset) > 1:
            # Prioritize 50% level as requested
            target_50 = next((l for l in levels_subset if l['pct'] == 50), None)
            
            if target_50:
                selected_level = target_50
                logger.info(f"Selecting requested 50% percentile level: {selected_level['pct']}% (Ratio={selected_level['ratio']:.4f})")
            else:
                # Sort by ratio ascending (lowest ratio first) -> Most critical
                levels_subset.sort(key=lambda x: x['ratio'])
                selected_level = levels_subset[0]
                logger.info(f"50% level not found. Selecting lowest ratio level: {selected_level['pct']}% (Ratio={selected_level['ratio']:.4f})")
            
            levels_subset = [selected_level]
            
        logger.info(f"\n--- Optimizing {param_name} for level: {[l['pct'] for l in levels_subset]} ---")
        # Fixed Points Search (user request: 0.25, 0.5, 0.75, 1.0)
        search_points = [0.25, 0.5, 0.75, 1.0]
        logger.info(f"Fixed Search Points: {search_points}")

        # 評価結果のキャッシュ
        eval_cache = {}

        def eval_exp(exp_val):
            # メモ化
            param_key = round(exp_val, 6)
            if param_key in eval_cache:
                logger.info(f"  Skipping re-evaluation for exp={exp_val:.4f} (Cached)")
                return eval_cache[param_key]

            total_weighted = 0.0
            total_raw = 0.0
            for l in levels_subset:
                pct, ratio, p_name = l['pct'], l['ratio'], l.get('param', 'unknown')
                run_preprocess(pct, param_name=p_name)
                # scale_lr_by_ratioを使わず直接計算 (exp1/exp2が混在しないため単純化)
                # ratioは相対値 (ratio / base_ratio) を使う
                relative_ratio = ratio / base_ratio if base_ratio > 0 else ratio
                relative_ratio = max(relative_ratio, 0.001) # Safety
                # adjusted_lr = base_lr * (relative_ratio ** exp)  <-- Multiplicative
                adjusted_lr = base_lr * (relative_ratio ** exp_val)
                logger.info(f"  Level {pct}% (Ratio={ratio:.4f}): LR={adjusted_lr:.8f} (exp={exp_val:.4f})")
                
                # ここでは「算出されたLRで学習した結果」を評価したいので
                # LRの再調整(max_iter>1)は行わず、1回だけ学習して結果を見る
                _, score, epoch = calibrate_lr_for_target(
                    adjusted_lr, target_epoch, model_name, cal_epochs, max_iter=1,
                    distance_exponent=0.5 # Dummy, unused since we take ret val
                )
                # 評価スコア: 単純にScoreを使用 (2026-02-18 User Request)
                # 高いほど良い
                metric = score
                
                # 最大化問題として扱うため、そのまま返す
                response_val = metric
                
                total_weighted += response_val
                total_raw += score
                distance = abs(epoch - target_epoch)
                logger.info(f"    -> Score={score:.4f}, Epoch={epoch}, Dist={distance}, Metric(Score)={metric:.4f}")
            
            avg_response = total_weighted / len(levels_subset)
            logger.info(f"  {param_name}={exp_val:.4f} -> AvgScore={avg_response:.4f}")
            
            eval_cache[param_key] = avg_response
            return avg_response

        # Fixed Point Evaluation (No Binary Search)
        scores = {}
        history = []
        
        for x in search_points:
            scores[x] = eval_exp(x)
            history.append((x, scores[x]))
            
        # Select Best
        best_x, best_s = max(scores.items(), key=lambda item: item[1])
        logger.info(f"Best {param_name}: {best_x:.4f} (Score={best_s:.4f})")
        return best_x, best_s

    # --- Step 2: 各パラメータごとのレベル準備 & 最適Exp探索 ---
    logger.info("\n>>> Step 2: Per-Parameter Exponent Calibration <<<")
    
    target_params = [
        'pitch_percentile',
        'symmetry_percentile',
        'y_diff_percentile',
        'mouth_open_percentile',
        'eyebrow_eye_percentile_high',
        'eyebrow_eye_percentile_low',
        'sharpness_percentile_low',
        'sharpness_percentile_high',
        'face_size_percentile_low',
        'face_size_percentile_high',
        'retouching_percentile',
        'mask_percentile',
        'glasses_percentile'
    ]
    
    # フィルタ強度 (Percentiles) - ユーザー指定: 50%のみ
    filter_percentiles = [50] 

    param_results = {} # param -> { 'exp1': val, 'exp2': val, 'levels': [...] }
    all_high_exps = []
    all_low_exps = []

    for param_name in target_params:
        logger.info(f"\n--- Calibrating Parameter: {param_name} ---")
        
        levels_high = []
        levels_low = []
        
        for pct in filter_percentiles:
            total, saved = run_preprocess(pct, param_name=param_name)
            if total > 0 and saved > 0:
                ratio = saved / total
                item = {'pct': pct, 'ratio': ratio, 'param': param_name}
                if ratio >= threshold:
                    levels_high.append(item)
                else:
                    levels_low.append(item)
            else:
                logger.warning(f"Filter {param_name}={pct}% produced no data.")
        
        if not levels_high and not levels_low:
            logger.warning(f"No valid levels for {param_name}. Skipping.")
            continue
            # Optimize for this parameter (Only one of p_exp1 or p_exp2 will be calculated based on 50%'s ratio)
        p_exp1 = 0.5 # Default
        p_exp2 = 0.5 # Default
        
        if levels_high:
            p_exp1, _ = optimize_exponent_for_levels(levels_high, range_exp1, f"{param_name} (High/50%)", initial_points=[0.15, 0.5, 1.0])
            # Copy to exp2 if not computed (Assume same tendency)
            p_exp2 = p_exp1
        elif levels_low:
            p_exp2, _ = optimize_exponent_for_levels(levels_low, range_exp2, f"{param_name} (Low/50%)", initial_points=[0.15, 0.5, 1.0])
            # Copy to exp1 if not computed
            p_exp1 = p_exp2
        
        logger.info(f"Set Exp for {param_name}: Exp1={p_exp1:.4f}, Exp2={p_exp2:.4f}")
        
        param_results[param_name] = {'exp1': p_exp1, 'exp2': p_exp2}
        all_high_exps.append(p_exp1)
        all_low_exps.append(p_exp2)

    # --- Step 3: グローバル設定 (平均) ---
    if all_high_exps:
        best_exp1 = sum(all_high_exps) / len(all_high_exps)
    else:
        best_exp1 = (range_exp1[0] + range_exp1[1]) / 2
        
    if all_low_exps:
        best_exp2 = sum(all_low_exps) / len(all_low_exps)
    else:
        best_exp2 = (range_exp2[0] + range_exp2[1]) / 2






    # --- Step 4: グループ最適化実行 (本番設定用) ---
    logger.info("\n>>> Step 4: Finalizing Config <<<")
    # best_exp1/2 are already calculated as averages above

    # --- Step 5: 保存 ---
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION COMPLETE")
    logger.info(f"Best Exp1 (High Ratio >= {threshold}): {best_exp1:.4f}")
    logger.info(f"Best Exp2 (Low Ratio < {threshold}) : {best_exp2:.4f}")
    
    logger.info("Per-Parameter Results:")
    for p, res in param_results.items():
        logger.info(f"  {p}: Exp1={res['exp1']:.4f}, Exp2={res['exp2']:.4f}")
        
    logger.info("=" * 60)

    result = {
        'exp1': best_exp1,
        'exp2': best_exp2,
        'threshold': threshold,
        'individual_exponents': param_results, # Changed to param-specific dict
        'base_lr': base_lr,
        'search_range_exp1': range_exp1,
        'search_range_exp2': range_exp2,
        'base_ratio': base_ratio,
        'formula': 'base_lr * ((ratio/base_ratio) ** exp)',
        'note': 'exp=exp1 if ratio >= threshold else exp2'
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    logger.info(f"Result saved to {OUTPUT_FILE}")

    run_preprocess(0)

if __name__ == "__main__":
    main()
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
