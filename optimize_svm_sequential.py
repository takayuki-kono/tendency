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
PYTHON_TRAIN = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe" # SVMはpreprocess環境(CPU/GPUどちらでも可)で動く
DATA_SOURCE_DIR = "train" 

# 出力ディレクトリ
LOG_DIR = "outputs/logs"
CACHE_DIR = "outputs/cache"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "svm_filter_opt_cache.json")

# 効率ベースの最適化設定
# 効率 = 精度向上 / フィルタリング枚数
MIN_EFFICIENCY_THRESHOLD = 0.00001

# パスが存在しない場合はデフォルトを使用
if not os.path.exists(PYTHON_PREPROCESS): PYTHON_PREPROCESS = "python"
if not os.path.exists(PYTHON_TRAIN): PYTHON_TRAIN = "python"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'svm_sequential_opt_log.txt'), mode='w', encoding='utf-8'),
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
            if cache.get('__file_count__') != current_file_count:
                logger.info(f"Data changed ({cache.get('__file_count__')} -> {current_file_count}). Clearing cache.")
                return {'__file_count__': current_file_count}
            return cache
        except:
            return {'__file_count__': current_file_count}
    return {'__file_count__': current_file_count}

def save_cache(cache):
    cache['__file_count__'] = count_files("train") + count_files("validation")
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

def run_trial(pitch, sym, y_diff, mouth_open, eb_eye_high, eb_eye_low, sharpness_low, sharpness_high, face_size_low=0, face_size_high=0, retouching=0, mask=0, glasses=0, grayscale=False, model_name='SVM', svm_params={}):
    """
    指定されたパラメータとモデルで前処理と学習を実行し、スコアを返す
    """
    # Merge SVM params into log string if present
    param_log = f"Model={model_name}, Pitch={pitch}%, Sym={sym}%, Y-Diff={y_diff}%, Mouth-Open={mouth_open}%, Eb-High={eb_eye_high}%, Eb-Low={eb_eye_low}%, Sharp-L={sharpness_low}%, Sharp-H={sharpness_high}%, FaceSize-L={face_size_low}%, FaceSize-H={face_size_high}%, Retouch={retouching}%, Mask={mask}%, Glasses={glasses}%, Grayscale={grayscale}"
    if svm_params:
        param_log += f", SVM_Params={svm_params}"

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating: {param_log}")
    
    file_count = count_files("train") + count_files("validation")
    # Cache key includes SVM params if present
    svm_key = f"_C={svm_params.get('C')}_k={svm_params.get('kernel')}_g={svm_params.get('gamma')}" if svm_params else ""
    cache_key = f"model={model_name}{svm_key}_pitch={pitch}_sym={sym}_ydiff={y_diff}_mouth={mouth_open}_ebh={eb_eye_high}_ebl={eb_eye_low}_sharplow={sharpness_low}_sharphigh={sharpness_high}_fsl={face_size_low}_fsh={face_size_high}_retouch={retouching}_mask={mask}_glasses={glasses}_gray={grayscale}_cnt={file_count}"
    
    cache = load_cache()
    if cache_key in cache:
        cached = cache[cache_key]
        if isinstance(cached, (list, tuple)) and len(cached) >= 2:
            if len(cached) == 3:
                raw_score, total_images, filtered_count = cached
            else:
                raw_score, filtering_rate = cached
                total_images = file_count
                filtered_count = int(total_images * filtering_rate)
            logger.info(f"Cache Hit! RawScore={raw_score:.4f}, Total={total_images}, Filtered={filtered_count}")
            return (raw_score, total_images, filtered_count)
        else:
            logger.info(f"Cache Hit (legacy)! Score: {cached}")
            return (float(cached), file_count, 0)

    try:
        # Preprocess
        cmd_pre = [
            PYTHON_PREPROCESS,
            "preprocess_multitask.py",
            "--out_dir", "preprocessed_multitask_svm", 
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
        
        # フィルタリング枚数を取得
        preprocess_output = (ret_pre.stdout or "") + "\n" + (ret_pre.stderr or "")
        total_images = 0
        saved_images = 0
        for line in preprocess_output.split('\n'):
            match_stats = re.search(r'Total=(\d+), Saved=(\d+)', line)
            if match_stats:
                total_images += int(match_stats.group(1))
                saved_images += int(match_stats.group(2))
        
        filtered_count = total_images - saved_images
        if total_images > 0:
            logger.info(f"Filtering Stats: Total={total_images}, Saved={saved_images}, Filtered={filtered_count} ({filtered_count/total_images*100:.1f}%)")

        if ret_pre.returncode != 0:
            logger.error(f"Preprocessing failed: {ret_pre.stderr}")
            return (0.0, 0, 0)

        # Train with SVM
        cmd_train = [PYTHON_TRAIN, "components/train_svm.py", "--data_dir", "preprocessed_multitask_svm"]
        
        # Pass SVM parameters if provided
        if svm_params:
            if 'C' in svm_params: cmd_train.extend(["--C", str(svm_params['C'])])
            if 'kernel' in svm_params: cmd_train.extend(["--kernel", str(svm_params['kernel'])])
            if 'gamma' in svm_params: cmd_train.extend(["--gamma", str(svm_params['gamma'])])

        logger.info(f"Running training with {model_name}...")
        ret_train = subprocess.run(cmd_train, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret_train.returncode != 0:
            logger.error(f"Training failed: {ret_train.stderr}")
            return (0.0, 0, 0)
            
        # Debug: Print full training output to see class distributions
        # (Logging this to understand why score is 0.5)
        logger.info(f"--- SVM Training Output ---\n{ret_train.stdout}\n-----------------------------")

        # Extract Score (Same as before)
        match = re.search(r"FINAL_SCORE:\s*([\d.]+)", ret_train.stdout)
        if match:
            raw_score = float(match.group(1))
            # 各タスクのスコアを抽出
            found_tasks = False
            for char_code in range(ord('A'), ord('Z') + 1):
                task_label = chr(char_code)
                # SVMスクリプトは "Task A Score: 0.50000" のような形式
                match_task = re.search(f"Task {task_label} Score:\s*([\d\.]+)", ret_train.stdout)
                if match_task:
                    logger.info(f"  Task {task_label}: {float(match_task.group(1)):.4f}")
                    found_tasks = True
            
            logger.info(f"Total Score (Average) = {raw_score:.4f}")
            
            # Retry logic for 0.5 score with custom params
            if raw_score == 0.5 and svm_params:
                logger.warning("Score is exactly 0.5 with custom SVM params. These params might be unsuitable for the current data.")
                logger.warning("Retrying with DEFAULT SVM params (gamma='scale', C=1.0)...")
                return run_trial(
                    pitch, sym, y_diff, mouth_open, eb_eye_high, eb_eye_low, 
                    sharpness_low, sharpness_high, face_size_low, face_size_high, 
                    retouching, mask, glasses, grayscale, model_name, svm_params={}
                )

            cache = load_cache()
            cache[cache_key] = (raw_score, total_images, filtered_count)
            save_cache(cache)
            return (raw_score, total_images, filtered_count)
        else:
            logger.error("Score not found in training output.")
            return (0.0, 0, 0)

    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return (0.0, 0, 0)

def load_best_svm_params():
    params_file = "outputs/best_svm_train_params.json"
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def optimize_single_param(target_name, current_params, model_name, baseline_score, baseline_filtered, svm_params={}, points=[0, 2, 5, 25, 50]):
    logger.info(f"\n>>> Optimizing {target_name} [Model: {model_name}] <<<")
    logger.info(f"Baseline: Score={baseline_score:.4f}, Filtered={baseline_filtered}")
    
    def evaluate_wrapper(val):
        # Independent Optimization: Always start from zero
        test_params = {
            'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
            'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0, 'sharpness_high': 0,
            'face_size_low': 0, 'face_size_high': 0, 'retouching': 0, 'mask': 0, 'glasses': 0
        }
        test_params[target_name] = val
        
        return run_trial(
            test_params['pitch'], test_params['sym'], test_params['y_diff'], test_params['mouth_open'],
            test_params['eb_eye_high'], test_params['eb_eye_low'],
            test_params['sharpness_low'], test_params['sharpness_high'],
            test_params['face_size_low'], test_params['face_size_high'],
            test_params['retouching'], test_params['mask'], test_params['glasses'],
            model_name=model_name,
            svm_params=svm_params
        )

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
    # best_valが0以外の場合、隣接する探索点との中間値を試す
    if best_val > 0:
        sorted_points = sorted(scores.keys())
        best_idx = sorted_points.index(best_val)
        
        # 隣接点との中間値を計算
        refinement_points = []
        if best_idx > 0:
            mid_low = (sorted_points[best_idx - 1] + best_val) // 2
            if mid_low not in scores and mid_low != sorted_points[best_idx - 1] and mid_low != best_val:
                refinement_points.append(mid_low)
        if best_idx < len(sorted_points) - 1:
            mid_high = (best_val + sorted_points[best_idx + 1]) // 2
            if mid_high not in scores and mid_high != best_val and mid_high != sorted_points[best_idx + 1]:
                refinement_points.append(mid_high)
        
        if refinement_points:
            logger.info(f"  [Refinement] Testing midpoints: {refinement_points}")
            for mid_p in refinement_points:
                raw_score, total_images, filtered_count = evaluate_wrapper(mid_p)
                logger.info(f"  {target_name}={mid_p} -> Score: {raw_score:.4f}, Filtered: {filtered_count}")
                scores[mid_p] = (raw_score, total_images, filtered_count) # 記録追加
                
                if raw_score > best_score:
                    best_score = raw_score
                    best_val = mid_p
                    best_filtered = filtered_count
                    logger.info(f"  [REFINED BEST] {target_name}={mid_p} (Score: {raw_score:.4f})")
    
    # --- Selection: Best Score vs Best Efficiency ---
    logger.info(f"  [Selection] Selecting between Best Score and Best Efficiency candidate...")
    
    candidates_info = []
    
    for p, (s_score, s_total, s_filtered) in scores.items():
        s_improvement = s_score - baseline_score
        s_filtered_diff = s_filtered - baseline_filtered
        
        # 効率計算 (フィルタ数増加あたりの精度向上)
        # filtered_diffが負（フィルタが減った）の場合は意味がない（通常ありえないが）
        # improvement <= 0 なら効率ゼロ以下
        if s_improvement <= 0:
             s_efficiency = -1.0 # 無視
        else:
             s_efficiency = s_improvement / (max(0, s_filtered_diff) + 1)
        
        candidates_info.append({
            'val': p,
            'score': s_score,
            'improvement': s_improvement,
            'filtered': s_filtered,
            'filtered_diff': s_filtered_diff,
            'efficiency': s_efficiency
        })
    
    if not candidates_info:
        # Default fallback
        logger.info(f"  -> No valid candidates found. Using baseline.")
        return 0, baseline_score, 0.0, 0, 0.0

    # 1. Best Score Candidate (current best_val logic)
    # improvement > 0 の中で最高スコアを探す（なければ0）
    valid_score_candidates = [c for c in candidates_info if c['improvement'] > 0]
    if valid_score_candidates:
        cand_score = max(valid_score_candidates, key=lambda x: x['score'])
    else:
        # 改善なしならBaseline (0)
        cand_score = next((c for c in candidates_info if c['val'] == 0), candidates_info[0])

    # 2. Best Efficiency Candidate
    # improvement > 0 の中で最高効率を探す
    if valid_score_candidates:
        cand_eff = max(valid_score_candidates, key=lambda x: x['efficiency'])
    else:
        cand_eff = cand_score

    logger.info(f"    Candidate Score (Best Score): {target_name}={cand_score['val']} (Score: {cand_score['score']:.4f}, Eff: {cand_score['efficiency']:.8f})")
    logger.info(f"    Candidate Eff   (Best Eff)  : {target_name}={cand_eff['val']} (Score: {cand_eff['score']:.4f}, Eff: {cand_eff['efficiency']:.8f})")

    # 3. Compare Efficiency and Select
    if cand_eff['efficiency'] > cand_score['efficiency']:
        selected = cand_eff
        reason = "Higher Efficiency"
    else:
        selected = cand_score
        reason = "Best Score (Efficiency tie or higher)"

    final_val = selected['val']
    final_score = selected['score']
    final_improvement = selected['improvement']
    final_filtered_diff = selected['filtered_diff']
    final_efficiency = selected['efficiency']

    logger.info(f"  -> Selected: {target_name}={final_val} via {reason}")
    logger.info(f"Finished optimizing {target_name}. Best: {final_val}, Score: {final_score:.4f}, Efficiency: {final_efficiency:.6f}")
    
    return final_val, final_score, final_improvement, final_filtered_diff, final_efficiency

def main():
    logger.info("Starting SVM Sequential Optimization (Efficiency-Based)")
    
    # Load Best SVM Params
    best_svm_params = load_best_svm_params()
    if best_svm_params:
        logger.info(f"Loaded Best SVM Params: {best_svm_params}")
    else:
        logger.info("Using default SVM params (no best params found)")

    # --- Step -1: SVM Hyperparameter Search ---
    logger.info("\n>>> Step -1: SVM Hyperparameter Search <<<")
    svm_candidates = [
        {'C': 1.0, 'gamma': 'scale'},
        {'C': 10.0, 'gamma': 'scale'},
        {'C': 0.1, 'gamma': 'scale'},
        {'C': 1.0, 'gamma': 'auto'},
        {'C': 10.0, 'gamma': 'auto'},
    ]
    
    best_svm_score = -1.0
    for candidate in svm_candidates:
        logger.info(f"Testing SVM params: {candidate}")
        score, _, _ = run_trial(0,0,0,0,0,0,0,0,0,0,0,0,0, model_name="SVM", svm_params=candidate)
        logger.info(f"  -> Score: {score:.4f}")
        if score > best_svm_score:
            best_svm_score = score
            best_svm_params = candidate
            logger.info(f"  [NEW BEST SVM] {candidate} (Score: {score:.4f})")
    
    logger.info(f"Best SVM Params: {best_svm_params} (Score: {best_svm_score:.4f})")

    # Initial Params
    current_params = {
        'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
        'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0, 'sharpness_high': 0,
        'face_size_low': 0, 'face_size_high': 0, 'retouching': 0, 'mask': 0, 'glasses': 0
    }
    
    param_efficiency = {}
    best_model = "SVM"

    # --- Step 0: Baseline ---
    logger.info("\n>>> Step 0: Baseline <<<")
    baseline_score, total_images, baseline_filtered = run_trial(0,0,0,0,0,0,0,0,0,0,0,0,0, model_name=best_model, svm_params=best_svm_params)
    logger.info(f"Baseline: Score={baseline_score:.4f}, Filtered={baseline_filtered}")
    
    # --- Parameter Optimization ---
    param_names = [
        'pitch', 'sym', 'y_diff', 'mouth_open',
        'eb_eye_high', 'eb_eye_low', 'sharpness_low', 'sharpness_high',
        'face_size_low', 'face_size_high', 'retouching', 'mask', 'glasses'
    ]
    
    global_best_score = baseline_score
    global_best_params = {k: 0 for k in current_params}
    global_best_desc = "Baseline (No Filter)"

    for param_name in param_names:
        val, score, improvement, filtered_diff, efficiency = optimize_single_param(
            param_name, current_params, best_model, baseline_score, baseline_filtered, svm_params=best_svm_params
        )
        current_params[param_name] = val
        param_efficiency[param_name] = {
            'val': val, 'improvement': improvement,
            'filtered_diff': filtered_diff, 'efficiency': efficiency
        }
        
        # Single Best Tracking
        if score > global_best_score:
            global_best_score = score
            temp_params = {k: 0 for k in current_params}
            temp_params[param_name] = val
            global_best_params = temp_params
            global_best_desc = f"Single Best ({param_name}={val})"
            logger.info(f"  [Global Best Update] New best found: {global_best_desc}, Score: {score:.4f}")

    # Phase 2: Efficiency-Based Greedy Integration (Grayscale無しで実行)
    logger.info("\n>>> Phase 2: Efficiency-Based Greedy Integration <<<")
    
    # 効率順にソート
    sorted_params = sorted(
        [p for p in param_efficiency.items() if p[1]['efficiency'] > 0 and p[1]['improvement'] > 0],
        key=lambda x: x[1]['efficiency'], 
        reverse=True
    )
    
    current_greedy_params = {k: 0 for k in current_params}
    
    # Base check (Grayscale=False)
    base_res = run_trial(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        grayscale=False, model_name=best_model, svm_params=best_svm_params
    )
    current_best_score = base_res[0]
    logger.info(f"Base Score: {current_best_score:.4f}")
    
    for param_name, info in sorted_params:
        val = info['val']
        logger.info(f"Trying to add {param_name}={val} (Efficiency: {info['efficiency']:.8f})...")
        
        temp_params = current_greedy_params.copy()
        temp_params[param_name] = val
        
        res = run_trial(
            temp_params['pitch'], temp_params['sym'], temp_params['y_diff'], temp_params['mouth_open'],
            temp_params['eb_eye_high'], temp_params['eb_eye_low'],
            temp_params['sharpness_low'], temp_params['sharpness_high'],
            temp_params['face_size_low'], temp_params['face_size_high'],
            temp_params['retouching'], temp_params['mask'], temp_params['glasses'],
            grayscale=False, model_name=best_model, svm_params=best_svm_params
        )
        score = res[0]
        
        if score >= current_best_score:
            logger.info(f"  -> Accepted (Score: {score:.4f} >= {current_best_score:.4f})")
            current_best_score = score
            current_greedy_params[param_name] = val
        else:
            logger.info(f"  -> Skipped (Score: {score:.4f} < {current_best_score:.4f})")
    
    greedy_score = current_best_score
    greedy_params = current_greedy_params.copy()
    
    # --- Final Selection Phase (Grayscale適用前) ---
    logger.info("\n>>> Final Selection Phase (without Grayscale) <<<")
    
    candidates = []
    
    # Greedy Integration
    candidates.append({'name': "Greedy Integration", 'params': greedy_params, 'score': greedy_score})
    logger.info(f"Strategy: {'Greedy':<15} Score={greedy_score:.4f}")
    
    # Single Best
    logger.info(f"\n--- Testing Single Best ({global_best_desc}) ---")
    sb_res = run_trial(
        global_best_params['pitch'], global_best_params['sym'], global_best_params['y_diff'], global_best_params['mouth_open'],
        global_best_params['eb_eye_high'], global_best_params['eb_eye_low'],
        global_best_params['sharpness_low'], global_best_params['sharpness_high'],
        global_best_params['face_size_low'], global_best_params['face_size_high'],
        global_best_params['retouching'], global_best_params['mask'], global_best_params['glasses'],
        grayscale=False, model_name=best_model, svm_params=best_svm_params
    )
    sb_score = sb_res[0]
    candidates.append({'name': f"Single Best ({global_best_desc})", 'params': global_best_params.copy(), 'score': sb_score})
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
        grayscale=False, model_name=best_model, svm_params=best_svm_params
    )
    res_gray = run_trial(
        final_params['pitch'], final_params['sym'], final_params['y_diff'], final_params['mouth_open'],
        final_params['eb_eye_high'], final_params['eb_eye_low'],
        final_params['sharpness_low'], final_params['sharpness_high'],
        final_params['face_size_low'], final_params['face_size_high'],
        final_params['retouching'], final_params['mask'], final_params['glasses'],
        grayscale=True, model_name=best_model, svm_params=best_svm_params
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
    
    logger.info("="*50)
    logger.info("OPTIMIZATION COMPLETE.")
    logger.info(f"Selected Strategy: {final_desc}")
    logger.info(f"Best Score: {final_score:.4f}")
    logger.info(f"Baseline Score: {baseline_score:.4f}")
    logger.info(f"Total Improvement: +{final_score - baseline_score:.4f}")
    logger.info(f"Params: {final_params}")
    
    # 結果を自動記録
    from components.result_logger import log_result
    log_result("optimize_svm_sequential", {
        "baseline_score": round(baseline_score, 4),
        "best_score": round(final_score, 4),
        "improvement": round(final_score - baseline_score, 4),
        "strategy": final_desc,
        "svm_params": best_svm_params,
        "filter_params": {k: v for k, v in final_params.items() if k != 'grayscale'},
        "grayscale": final_params.get('grayscale', False),
        "param_efficiency": {k: {ek: round(ev, 6) if isinstance(ev, float) else ev for ek, ev in v.items()} for k, v in param_efficiency.items()},
    })
    
    # Command Generation
    final_cmd = (
        f"{PYTHON_PREPROCESS} preprocess_multitask.py --out_dir preprocessed_multitask_svm "
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
    logger.info("="*50)

if __name__ == "__main__":
    main()
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
