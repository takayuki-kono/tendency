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


def run_preprocess(filter_percentile):
    """
    単一パラメータ(y_diff)でフィルタリングを適用して前処理を実行する。
    """
    cmd = [
        PYTHON_PREPROCESS,
        "preprocess_multitask.py",
        "--out_dir", "preprocessed_multitask",
        "--y_diff_percentile", str(filter_percentile),
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
    logger.info("LR Scaling Exponent Calibration (Two-Stage)")
    logger.info("=" * 60)

    model_name = 'EfficientNetV2B0'
    target_epoch = 10
    cal_epochs = 20
    
    # ユーザー指定の探索範囲
    range_exp1 = (0.3, 1.0)  # High Ratio (>= threshold)
    range_exp2 = (0.3, 1.0)  # Low Ratio (< threshold)
    threshold = 0.5

    # --- Step 1: ベースライン（フィルタなし）---
    logger.info("\n>>> Step 1: Baseline (no filter, calibrate to epoch 10) <<<")
    best_params = load_best_train_params()
    initial_lr = best_params.get('learning_rate', 0.0001)
    
    # キャリブレーション済みLRキャッシュを確認
    cal_cache_file = "outputs/cache/calibrated_lr.json"
    base_lr = None
    base_score = None
    base_epoch = None

    base_ratio = 1.0
    
    if os.path.exists(cal_cache_file):
        try:
            with open(cal_cache_file, 'r', encoding='utf-8') as f:
                cal_cache = json.load(f)
            base_lr = cal_cache['lr']
            base_score = cal_cache['score']
            base_epoch = cal_cache.get('best_epoch')
            base_ratio = cal_cache.get('base_ratio', 1.0) # Load base_ratio
            cached_initial_lr_raw = cal_cache.get('initial_lr')
            
            # Check consistency
            cached_initial_lr = None
            if cached_initial_lr_raw is not None:
                try: cached_initial_lr = float(cached_initial_lr_raw)
                except: pass
            
            initial_lr_match = (
                cached_initial_lr is not None and
                abs(cached_initial_lr - float(initial_lr)) <= max(1e-12, abs(float(initial_lr)) * 1e-6)
            )
            
            if base_epoch == target_epoch and initial_lr_match:
                # base_ratioが1.0の場合（古いキャッシュ等）、念のため再計測して正しい値にする
                if abs(base_ratio - 1.0) < 1e-6:
                    logger.info("Base ratio is 1.0 in cache. Re-calculating actual base ratio...")
                    total_base, saved_base = run_preprocess(0)
                    real_base_ratio = saved_base / total_base if total_base > 0 else 1.0
                    if abs(real_base_ratio - 1.0) > 1e-4:
                        base_ratio = real_base_ratio
                        logger.info(f"Updated Base Ratio: {base_ratio:.4f}")

                logger.info(f"Cache Hit! LR={base_lr:.8f}, Ratio={base_ratio:.4f}")
            else:
                logger.warning("Cache ignored (mismatch).")
                base_lr = None
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            base_lr = None

    if base_lr is None:
        total_base, saved_base = run_preprocess(0)
        base_ratio = saved_base / total_base if total_base > 0 else 1.0
        
        base_lr, base_score, base_epoch = calibrate_lr_for_target(
            initial_lr, target_epoch, model_name, cal_epochs, max_iter=12, strict_target=True
        )
        logger.info(f"Baseline: LR={base_lr:.8f}, Score={base_score:.4f}, Ratio={base_ratio:.4f}")
        os.makedirs(os.path.dirname(cal_cache_file), exist_ok=True)
        with open(cal_cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'lr': base_lr, 'score': base_score, 'best_epoch': base_epoch, 
                'initial_lr': initial_lr, 'base_ratio': base_ratio
            }, f, indent=4)

    if base_epoch != target_epoch:
        logger.error(f"Baseline did not converge to epoch {target_epoch}. Aborting.")
        return

    # --- Step 2: フィルタレベル準備 & 分類 ---
    logger.info("\n>>> Step 2: Prepare filter levels & Split by threshold <<<")
    filter_percentiles = [5, 25, 50, 75] # Updated: 5%, 25%, 50%, 75% for calibration
    levels_high = [] # ratio >= threshold
    levels_low = []  # ratio < threshold
    
    for pct in filter_percentiles:
        total, saved = run_preprocess(pct)
        if total > 0 and saved > 0:
            ratio = saved / total
            item = {'pct': pct, 'ratio': ratio}
            if ratio >= threshold:
                levels_high.append(item)
            else:
                levels_low.append(item)
        else:
            logger.warning(f"Filter {pct}% produced no data.")

    logger.info(f"Threshold: {threshold}")
    logger.info(f"High Ratio Levels (Use exp1): {[l['pct'] for l in levels_high]}")
    logger.info(f"Low Ratio Levels (Use exp2): {[l['pct'] for l in levels_low]}")
    
    if not levels_high and not levels_low:
        logger.error("No valid levels.")
        return

    def optimize_exponent_for_levels(levels_subset, exp_range, param_name):
        if not levels_subset:
            logger.info(f"No levels for {param_name}. Skipping optimization, returning range midpoint.")
            return (exp_range[0] + exp_range[1]) / 2, 0.0

        logger.info(f"\n--- Optimizing {param_name} for levels: {[l['pct'] for l in levels_subset]} ---")
        logger.info(f"Search Range: {exp_range}")

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
                pct, ratio = l['pct'], l['ratio']
                run_preprocess(pct)
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
                # 評価スコア: ユーザー指定の数式
                # abs(21 - best_score * 30) + epoch_distance
                # 低いほど良い評価
                distance = abs(epoch - target_epoch)
                metric = abs(21.0 - score * 30.0) + float(distance)
                
                # 最大化問題として扱うため、マイナスを返す
                response_val = -metric
                
                total_weighted += response_val
                total_raw += score
                logger.info(f"    -> Score={score:.4f}, Epoch={epoch}, Dist={distance}, Metric={metric:.4f}")
            
            avg_response = total_weighted / len(levels_subset)
            logger.info(f"  {param_name}={exp_val:.4f} -> AvgMetric={-avg_response:.4f} (negated)")
            
            eval_cache[param_key] = avg_response
            return avg_response

        # 二分探索 (3 iterations)
        lo, hi = exp_range
        
        # 特例: 探索範囲が固定点の場合
        if abs(hi - lo) < 1e-6:
             logger.info(f"Fixed range detected for {param_name}. Evaluating single point {lo:.4f}.")
             score = eval_exp(lo)
             return lo, score

        mid = (lo + hi) / 2
        history = []
        scores = {}
        
        # Initial 3 points
        for x in [lo, mid, hi]:
            scores[x] = eval_exp(x)
            history.append((x, scores[x]))
            
        # Refinement Loop
        for i in range(3):
            # Sort by score descending
            sorted_res = sorted(scores.items(), key=lambda item: -item[1])
            
            if len(sorted_res) < 2:
                break
                
            top1_x, top1_s = sorted_res[0]
            top2_x, top2_s = sorted_res[1]
            
            # 1位と2位の間、あるいは勾配方向を探索
            new_x = (top1_x + top2_x) / 2
            
            # Check if new_x is already evaluated (or very close)
            if any(abs(h[0] - new_x) < 0.001 for h in history):
                # 詰まったら範囲内の別ポイントを適当に試す (Perturbation)
                diff = abs(top1_x - top2_x)
                if diff < 0.001: break # Converged
                
                if top1_x > top2_x: new_x = top1_x + diff * 0.5
                else: new_x = top1_x - diff * 0.5
                
            new_x = max(lo, min(hi, new_x))

            logger.info(f"  [Iter {i+1}] Trying {new_x:.4f}...")
            scores[new_x] = eval_exp(new_x)
            history.append((new_x, scores[new_x]))

        best_x, best_s = max(scores.items(), key=lambda item: item[1])
        logger.info(f"Best {param_name}: {best_x:.4f} (Score={best_s:.4f})")
        return best_x, best_s
    # --- Step 3: 個別レベルでの最適Exp探索 (分析用) ---
    logger.info("\n>>> Step 3: Individual Exponent Optimization for each level <<<")
    individual_results = {}
    for pct in filter_percentiles:
        # Find the level object
        target_level = None
        for l in levels_high + levels_low:
            if l['pct'] == pct:
                target_level = l
                break
        
        if target_level:
            # Determine range based on threshold logic (just for initial guess/range)
            search_range = range_exp1 if target_level['ratio'] >= threshold else range_exp2
            # Optimize for single level
            best_e, best_s = optimize_exponent_for_levels([target_level], search_range, f"Exp(Level {pct}%)")
            individual_results[pct] = best_e
            logger.info(f"===> Best Exp for Level {pct}% (Ratio={target_level['ratio']:.4f}): {best_e:.4f} (Score={best_s:.4f})")

    # --- Step 4: グループ最適化実行 (本番設定用) ---
    logger.info("\n>>> Step 4: Group Optimization (High/Low) for Config <<<")
    best_exp1, score1 = optimize_exponent_for_levels(levels_high, range_exp1, "exp1 (High Ratio)")
    best_exp2, score2 = optimize_exponent_for_levels(levels_low, range_exp2, "exp2 (Low Ratio)")

    # --- Step 5: 保存 ---
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION COMPLETE")
    logger.info(f"Best Exp1 (High Ratio >= {threshold}): {best_exp1:.4f}")
    logger.info(f"Best Exp2 (Low Ratio < {threshold}) : {best_exp2:.4f}")
    
    logger.info("Individual Results:")
    for p, e in individual_results.items():
        logger.info(f"  Level {p}%: {e:.4f}")
        
    logger.info("=" * 60)

    result = {
        'exp1': best_exp1,
        'exp2': best_exp2,
        'threshold': threshold,
        'individual_exponents': individual_results, # Added
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
