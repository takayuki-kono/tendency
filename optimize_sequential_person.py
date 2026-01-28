import subprocess
import sys
import re
import os
import logging
import json
import hashlib

# --- Settings ---
PYTHON_PREPROCESS = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe"
PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
DATA_SOURCE_DIR = "train" 

# Directories
LOG_DIR = "outputs/logs"
CACHE_DIR = "outputs/cache"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "filter_opt_person_cache.json")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'sequential_person_filter_opt_log.txt'), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def count_files(directory):
    if not os.path.exists(directory): return 0
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

def run_trial(pitch, sym, y_diff, mouth_open, eb_eye_high, eb_eye_low, sharpness_low, grayscale=False, model_name='EfficientNetV2B0'):
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating: Model={model_name}, Pitch={pitch}%, Sym={sym}%, Y-Diff={y_diff}%, Mouth-Open={mouth_open}%, Eb-High={eb_eye_high}%, Eb-Low={eb_eye_low}%, Sharp={sharpness_low}%, Grayscale={grayscale}")
    
    file_count = count_files("train") + count_files("validation")
    cache_key = f"model={model_name}_pitch={pitch}_sym={sym}_ydiff={y_diff}_mouth={mouth_open}_ebh={eb_eye_high}_ebl={eb_eye_low}_sharplow={sharpness_low}_gray={grayscale}_cnt={file_count}"
    
    cache = load_cache()
    if cache_key in cache:
        logger.info(f"Cache Hit! Score: {cache[cache_key]}")
        return cache[cache_key]

    try:
        # Preprocess
        cmd_pre = [
            PYTHON_PREPROCESS,
            "preprocess_person.py",
            "--out_dir", "preprocessed_person",
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
            logger.info(f"PREPROCESS STDOUT:\n{ret_pre.stdout}")
        if ret_pre.stderr:
            logger.info(f"PREPROCESS STDERR:\n{ret_pre.stderr}")
            
        if ret_pre.returncode != 0:
            logger.error(f"Preprocessing failed: {ret_pre.stderr}")
            return 0.0

        # Train (Filter Search)
        cmd_train = [PYTHON_TRAIN, "components/train_filter_person_trial.py", "--model_name", model_name]
        logger.info(f"Running filter trial training with {model_name}...")
        ret_train = subprocess.run(cmd_train, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret_train.returncode != 0:
            logger.error(f"Training failed: {ret_train.stderr}")
            return 0.0

        # Extract Score
        match = re.search(r"FINAL_SCORE:\s*([\d.]+)", ret_train.stdout)
        if match:
            score = float(match.group(1))
            logger.info(f"Result: Score = {score}")
            if score == 0.0:
                 logger.info(f"TRAIN STDOUT:\n{ret_train.stdout}")
                 logger.info(f"TRAIN STDERR:\n{ret_train.stderr}")
            
            cache = load_cache()
            cache[cache_key] = score
            save_cache(cache)
            return score
        else:
            logger.error("Score not found in training output.")
            return 0.0

    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return 0.0

def optimize_single_param(target_name, current_params, model_name, points=[0, 25, 50]):
    logger.info(f"\n>>> Optimizing {target_name} (Points: {points}) [Model: {model_name}] <<<")
    
    # Independent Optimization: Baseline is all 0
    test_params = {
        'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
        'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0
    }
    
    scores = {}
    for val in points:
        test_params[target_name] = val
        score = run_trial(
            test_params['pitch'], test_params['sym'], test_params['y_diff'], test_params['mouth_open'],
            test_params['eb_eye_high'], test_params['eb_eye_low'],
            test_params['sharpness_low'], model_name=model_name
        )
        scores[val] = score
        
    best_val = max(scores, key=scores.get)
    best_score = scores[best_val]
    logger.info(f"Finished optimizing {target_name}. Best: {best_val} (Score: {best_score})")
    return best_val, best_score

def main():
    logger.info("Starting Sequential Optimization (Person/Single Task)")
    
    current_params = {
        'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
        'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0
    }
    
    # Step 0: Model
    logger.info("\n>>> Step 0: Model Architecture Selection <<<")
    candidate_models = ['EfficientNetV2B0', 'EfficientNetV2S']
    best_model = 'EfficientNetV2B0'
    best_model_score = -1.0
    
    for m in candidate_models:
        logger.info(f"Testing Model: {m}")
        score = run_trial(0, 0, 0, 0, 0, 0, 0, model_name=m)
        if score > best_model_score:
            best_model_score = score
            best_model = m
    
    logger.info(f"Best Model Selected: {best_model}")
    
    # Optimization Sequence
    # 1. Pitch
    current_params['pitch'], _ = optimize_single_param('pitch', current_params, best_model, points=[0, 25, 50])
    
    # 2. Symmetry
    current_params['sym'], _ = optimize_single_param('sym', current_params, best_model, points=[0, 25, 50])
    
    # ... (Adding other params as needed, keeping it simple for now as requested "componentize")
    # Doing full sequence as originally planned
    current_params['y_diff'], _ = optimize_single_param('y_diff', current_params, best_model, points=[0, 25, 50])
    current_params['mouth_open'], _ = optimize_single_param('mouth_open', current_params, best_model, points=[0, 25, 50])
    current_params['eb_eye_high'], _ = optimize_single_param('eb_eye_high', current_params, best_model, points=[0, 25, 50])
    current_params['eb_eye_low'], _ = optimize_single_param('eb_eye_low', current_params, best_model, points=[0, 25, 50])
    current_params['sharpness_low'], _ = optimize_single_param('sharpness_low', current_params, best_model, points=[0, 25, 50])
    
    # Grayscale
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
    
    grayscale_best = False
    if score_gray > score_color:
        grayscale_best = True
        logger.info("Grayscale selected.")
    else:
        logger.info("Color selected.")
        
    current_params['grayscale'] = grayscale_best
    
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Final Best Params: {current_params}")
    
    final_cmd = (
        f"{PYTHON_PREPROCESS} preprocess_person.py --out_dir preprocessed_person "
        f"--train_dir train --val_dir validation "
        f"--pitch_percentile {current_params['pitch']} "
        f"--symmetry_percentile {current_params['sym']} "
        f"--y_diff_percentile {current_params['y_diff']} "
        f"--mouth_open_percentile {current_params['mouth_open']} "
        f"--eyebrow_eye_percentile_high {current_params['eb_eye_high']} "
        f"--eyebrow_eye_percentile_low {current_params['eb_eye_low']} "
        f"--sharpness_percentile_low {current_params['sharpness_low']} "
    )
    if grayscale_best:
        final_cmd += "--grayscale "
        
    logger.info("\nRun this command to apply the best filters:")
    logger.info(final_cmd)
    logger.info("="*50)

if __name__ == "__main__":
    main()
