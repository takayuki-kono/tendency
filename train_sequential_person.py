import subprocess
import sys
import re
import os
import logging
import json
import hashlib

# --- Settings ---
# Adjust to your environment
PYTHON_EXEC = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
DATA_SOURCE_DIR = "preprocessed_person/train" 

# Directories
LOG_DIR = "outputs/logs"
CACHE_DIR = "outputs/cache"
MODEL_DIR = "outputs/models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "train_opt_person_cache.json")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'sequential_person_opt_log.txt'), mode='w', encoding='utf-8'),
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

def run_trial(params):
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating Params: {params}")
    
    params_str = json.dumps(params, sort_keys=True)
    file_count = count_files(DATA_SOURCE_DIR)
    
    key_src = f"{params_str}_count={file_count}"
    cache_key = hashlib.md5(key_src.encode('utf-8')).hexdigest()
    
    cache = load_cache()
    if cache_key in cache:
        logger.info(f"Cache Hit! Val Accuracy: {cache[cache_key]}")
        return cache[cache_key]

    logger.info(f"Cache Miss. Running process... (File Count: {file_count})")
    
    cmd = [PYTHON_EXEC, "components/train_person_trial.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    try:
        ret = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        # Check STDERR for critical errors but don't fail immediately as TF outputs lots of logs to stderr
        if ret.returncode != 0:
            logger.error(f"Training failed: {ret.stderr}")
            return 0.0

        # Extract Score (FINAL_VAL_ACCURACY)
        match = re.search(r"FINAL_VAL_ACCURACY:\s*(\d+\.\d+)", ret.stdout)
        
        if match:
            score = float(match.group(1))
            logger.info(f"Result: Val Accuracy = {score}")
            
            cache = load_cache()
            cache[cache_key] = score
            save_cache(cache)
            return score
        else:
            logger.error("Score not found in output.")
            # logger.error(ret.stdout) # Uncomment to debug
            return 0.0

    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return 0.0

def optimize_param(target_name, candidates, current_params):
    logger.info(f"\n>>> Optimizing {target_name} (Candidates: {candidates}) <<<")
    scores = {}
    for val in candidates:
        params = current_params.copy()
        params[target_name] = val
        score = run_trial(params)
        scores[val] = score
    
    best_val = max(scores, key=scores.get)
    best_score = scores[best_val]
    logger.info(f"Finished optimizing {target_name}. Best: {best_val} (Score: {best_score})")
    return best_val, best_score

def main():
    logger.info("Starting Sequential Training Optimization (Person/Single Task)")
    
    current_params = {
        'model_name': 'EfficientNetV2B0',
        'num_dense_layers': 1,
        'dense_units': 128,
        'dropout': 0.3,
        'head_dropout': 0.3,
        'learning_rate': 1e-3,
        'rotation_range': 0.0,
        'width_shift_range': 0.0,
        'height_shift_range': 0.0,
        'zoom_range': 0.0,
        'horizontal_flip': 'False',
        'mixup_alpha': 0.0,
        'fine_tune': 'False'
    }
    
    # Step 0: Model
    best_model, _ = optimize_param('model_name', ['EfficientNetV2B0', 'EfficientNetV2S'], current_params)
    current_params['model_name'] = best_model

    # Step 1: Learning Rate
    best_lr, _ = optimize_param('learning_rate', [1e-3, 5e-4, 1e-4], current_params)
    current_params['learning_rate'] = best_lr
    
    # Step 2: Layers
    best_layers, _ = optimize_param('num_dense_layers', [1, 2], current_params)
    current_params['num_dense_layers'] = best_layers
    
    best_units, _ = optimize_param('dense_units', [128, 256], current_params)
    current_params['dense_units'] = best_units
    
    best_dropout, _ = optimize_param('dropout', [0.3, 0.5], current_params)
    current_params['dropout'] = best_dropout
    
    # Head Dropout
    best_head_dropout, _ = optimize_param('head_dropout', [0.3, 0.5], current_params)
    current_params['head_dropout'] = best_head_dropout

    # Step 3: Augmentation
    # Basic Flip
    best_flip, _ = optimize_param('horizontal_flip', ['True', 'False'], current_params)
    current_params['horizontal_flip'] = best_flip

    # Rotation
    best_rot, _ = optimize_param('rotation_range', [0.0, 0.1], current_params)
    current_params['rotation_range'] = best_rot
    
    # Width Shift
    best_w_shift, _ = optimize_param('width_shift_range', [0.0, 0.1], current_params)
    current_params['width_shift_range'] = best_w_shift

    # Height Shift
    best_h_shift, _ = optimize_param('height_shift_range', [0.0, 0.1], current_params)
    current_params['height_shift_range'] = best_h_shift

    # Zoom
    best_zoom, _ = optimize_param('zoom_range', [0.0, 0.1], current_params)
    current_params['zoom_range'] = best_zoom

    # Mixup (0.0=Off, 0.2=On)
    best_mixup, _ = optimize_param('mixup_alpha', [0.0, 0.2], current_params)
    current_params['mixup_alpha'] = best_mixup

    # Step 4: Final Fine-Tuning
    logger.info("\n>>> STARTING FINAL FINE-TUNING <<<")
    current_params['fine_tune'] = 'True'
    current_params['epochs'] = 30 
    
    final_score = run_trial(current_params)
    
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Best Params: {current_params}")
    logger.info(f"Final Score: {final_score}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
