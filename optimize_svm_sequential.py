import subprocess
import sys
import re
import os
import logging
import time
import json
import hashlib

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

def run_trial(pitch, sym, y_diff, mouth_open, eb_eye_high, eb_eye_low, sharpness_low, grayscale=False, model_name='SVM'):
    """
    指定されたパラメータとモデルで前処理と学習を実行し、スコアを返す
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating: Model={model_name}, Pitch={pitch}%, Sym={sym}%, Y-Diff={y_diff}%, Mouth-Open={mouth_open}%, Eb-High={eb_eye_high}%, Eb-Low={eb_eye_low}%, Sharp={sharpness_low}%, Grayscale={grayscale}")
    
    file_count = count_files("train") + count_files("validation")
    cache_key = f"model={model_name}_pitch={pitch}_sym={sym}_ydiff={y_diff}_mouth={mouth_open}_ebh={eb_eye_high}_ebl={eb_eye_low}_sharplow={sharpness_low}_gray={grayscale}_cnt={file_count}"
    
    cache = load_cache()
    if cache_key in cache:
        logger.info(f"Cache Hit! Score: {cache[cache_key]}")
        return cache[cache_key]

    try:
        # Preprocess (same as before)
        cmd_pre = [
            PYTHON_PREPROCESS,
            "preprocess_multitask.py",
            "--out_dir", "preprocessed_multitask_svm", # SVM専用に出力先を変更
            "--pitch_percentile", str(pitch),
            "--symmetry_percentile", str(sym),
            "--y_diff_percentile", str(y_diff),
            "--mouth_open_percentile", str(mouth_open),
            "--eyebrow_eye_percentile_high", str(eb_eye_high),
            "--eyebrow_eye_percentile_low", str(eb_eye_low),
            "--sharpness_percentile_low", str(sharpness_low)
        ]
        if grayscale:
            cmd_pre.append("--grayscale")
            
        logger.info("Running preprocessing...")
        ret_pre = subprocess.run(cmd_pre, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret_pre.stdout:
            # logger.info(f"Preprocessing STDOUT:\n{ret_pre.stdout}") # ログ抑制
            pass
        if ret_pre.stderr:
            # logger.warning(f"Preprocessing STDERR:\n{ret_pre.stderr}")
            pass

        if ret_pre.returncode != 0:
            logger.error(f"Preprocessing failed with return code {ret_pre.returncode}")
            logger.error(ret_pre.stderr)
            return 0.0

        # Train with SVM
        cmd_train = [PYTHON_TRAIN, "components/train_svm.py", "--data_dir", "preprocessed_multitask_svm"]
        logger.info(f"Running training with {model_name}...")
        ret_train = subprocess.run(cmd_train, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret_train.returncode != 0:
            logger.error(f"Training failed: {ret_train.stderr}")
            return 0.0

        # Extract Score
        match = re.search(r"FINAL_SCORE:\s*([\d.]+)", ret_train.stdout)
        if match:
            score = float(match.group(1))
            logger.info(f"Result: Score = {score}")
            
            cache = load_cache()
            cache[cache_key] = score
            save_cache(cache)
            return score
        else:
            logger.error("Score not found in training output.")
            logger.error(f"STDOUT:\n{ret_train.stdout}")
            return 0.0

    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return 0.0

def optimize_single_param(target_name, current_params, model_name, points=[0, 25, 50]):
    """
    1つのパラメータを最適化する。
    Step 1: pointsで指定された点を評価
    Step 2: 上位2点の中間を探索（2分探索的アプローチ）
    """
    logger.info(f"\n>>> Optimizing {target_name} (Points: {points}) [Model: {model_name}] <<<")
    
    best_val = current_params[target_name]
    best_score = -1.0
    history = {}

    def evaluate_wrapper(val):
        if val in history: return history[val]
        
        # Independent Optimization: Always start from all-zero params
        test_params = {
            'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
            'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0
        }
        test_params[target_name] = val
        
        score = run_trial(
            test_params['pitch'], test_params['sym'], test_params['y_diff'], test_params['mouth_open'],
            test_params['eb_eye_high'], test_params['eb_eye_low'],
            test_params['sharpness_low'], model_name=model_name
        )
        history[val] = score
        return score

    # Step 1: Evaluate initial points
    logger.info(f"Step 1: Evaluate points {points}")
    scores = {}
    for p in points:
        scores[p] = evaluate_wrapper(p)
    
    # Step 2: Refinement
    logger.info("Step 2: Binary Search Refinement")
    while True:
        sorted_history = sorted(history.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_history) < 2: break
            
        best1_val, best1_score = sorted_history[0]
        best2_val, best2_score = sorted_history[1]
        
        mid_val = int((best1_val + best2_val) / 2)
        
        if mid_val in history:
            logger.info(f"Refinement converged at {mid_val} (Already evaluated).")
            # Safety fallback: Choose the lower value between the top 2
            safer_val = min(best1_val, best2_val)
            logger.info(f"Selecting lower value for reproducibility/safety: {safer_val}")
            best_val = safer_val
            best_score = history[safer_val]
            logger.info(f"Finished optimizing {target_name}. Best: {best_val} (Score: {best_score})")
            return best_val, best_score
            
        logger.info(f"Refining: Best1={best1_val}({best1_score:.4f}), Best2={best2_val}({best2_score:.4f}) -> Next: {mid_val}")
        scores[mid_val] = evaluate_wrapper(mid_val)
            
    best_val = max(scores, key=scores.get)
    best_score = scores[best_val]
    logger.info(f"Finished optimizing {target_name}. Best: {best_val} (Score: {best_score})")
    return best_val, best_score

def main():
    logger.info("Starting SVM Sequential Optimization")
    
    # Initial Params
    current_params = {
        'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
        'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0
    }
    
    best_model = 'SVM'
    
    # Optimization Sequence
    
    # 1. Pitch
    val, _ = optimize_single_param('pitch', current_params, best_model, points=[0, 25, 50])
    current_params['pitch'] = val
    
    # 2. Symmetry
    val, _ = optimize_single_param('sym', current_params, best_model, points=[0, 25, 50])
    current_params['sym'] = val
    
    # 3. Y-Diff
    val, _ = optimize_single_param('y_diff', current_params, best_model, points=[0, 25, 50])
    current_params['y_diff'] = val
    
    # 4. Mouth Open
    val, _ = optimize_single_param('mouth_open', current_params, best_model, points=[0, 25, 50])
    current_params['mouth_open'] = val
    
    # 5. Eyebrow-Eye High (Top X% cut)
    val, _ = optimize_single_param('eb_eye_high', current_params, best_model, points=[0, 25, 50])
    current_params['eb_eye_high'] = val

    # 6. Eyebrow-Eye Low (Bottom X% cut)
    val, _ = optimize_single_param('eb_eye_low', current_params, best_model, points=[0, 25, 50])
    current_params['eb_eye_low'] = val
    
    # 7. Sharpness Low (Bottom X% cut - blurry images)
    val, _ = optimize_single_param('sharpness_low', current_params, best_model, points=[0, 25, 50])
    current_params['sharpness_low'] = val
    
    # 8. Grayscale (Boolean comparison)
    # SVMの場合も一応試すが、Embedding抽出器がRGB前提の場合どうなるか注意
    # InsightFaceは基本RGBだが、グレースケールでも動く（内部で変換されることも）
    logger.info("\n>>> Optimizing Grayscale (True vs False) <<<")
    score_color = run_trial(
        current_params['pitch'], current_params['sym'], current_params['y_diff'], current_params['mouth_open'],
        current_params['eb_eye_high'], current_params['eb_eye_low'], current_params['sharpness_low'],
        grayscale=False, model_name=best_model
    )
    score_gray = run_trial(
        current_params['pitch'], current_params['sym'], current_params['y_diff'], current_params['mouth_open'],
        current_params['eb_eye_high'], current_params['eb_eye_low'], current_params['sharpness_low'],
        grayscale=True, model_name=best_model
    )
    if score_gray > score_color:
        current_params['grayscale'] = True
        logger.info(f"Grayscale selected (Score: {score_gray} > {score_color})")
    else:
        current_params['grayscale'] = False
        logger.info(f"Color selected (Score: {score_color} >= {score_gray})")
    
    # Phase 2: Global Scaling Optimization
    logger.info("\n>>> Phase 2: Global Scaling Optimization (x1.0, x0.5, x0.25) <<<")
    
    best_independent_params = current_params.copy()
    scaling_factors = [1.0, 0.5, 0.25]
    
    best_scaling_score = -1.0
    best_scaled_params = None
    best_factor = 1.0
    
    for factor in scaling_factors:
        scaled_params = {
            k: int(v * factor) for k, v in best_independent_params.items() if k != 'grayscale'
        }
        scaled_params['grayscale'] = best_independent_params.get('grayscale', False)
        logger.info(f"Testing Scaling Factor x{factor}: {scaled_params}")
        
        score = run_trial(
            scaled_params['pitch'], scaled_params['sym'], scaled_params['y_diff'], scaled_params['mouth_open'],
            scaled_params['eb_eye_high'], scaled_params['eb_eye_low'],
            scaled_params['sharpness_low'], grayscale=scaled_params['grayscale'], model_name=best_model
        )
        
        if score > best_scaling_score:
            best_scaling_score = score
            best_scaled_params = scaled_params
            best_factor = factor
            
    logger.info(f"Phase 2 Complete. Best Factor: x{best_factor} (Score: {best_scaling_score})")
    current_params = best_scaled_params
    final_score = best_scaling_score
    
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Final Best Params: {current_params}")
    logger.info(f"Final Score: {final_score}")
    logger.info("="*50)

    # Generate and print the final preprocessing command
    final_cmd = (
        f"{PYTHON_PREPROCESS} preprocess_multitask.py --out_dir preprocessed_multitask_svm "
        f"--pitch_percentile {current_params['pitch']} "
        f"--symmetry_percentile {current_params['sym']} "
        f"--y_diff_percentile {current_params['y_diff']} "
        f"--mouth_open_percentile {current_params['mouth_open']} "
        f"--eyebrow_eye_percentile_high {current_params['eb_eye_high']} "
        f"--eyebrow_eye_percentile_low {current_params['eb_eye_low']} "
        f"--sharpness_percentile_low {current_params['sharpness_low']} "
    )
    if current_params.get('grayscale', False):
        final_cmd += "--grayscale "
    logger.info("\nRun this command to apply the best filters:")
    logger.info(final_cmd)
    logger.info("="*50)

if __name__ == "__main__":
    main()
