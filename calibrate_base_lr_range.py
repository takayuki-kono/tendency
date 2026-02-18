"""
Base LR Range Calibration for Epochs 15-20

This script calibrates the Base Learning Rate such that the model converges
(reaches Best Epoch) at specific target epochs: 15, 16, 17, 18, 19, 20.

Usage:
    python calibrate_base_lr_range.py
"""
import subprocess
import sys
import re
import os
import logging
import json
import math

# --- Config ---
PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
BEST_TRAIN_PARAMS_FILE = "outputs/best_train_params.json"
OUTPUT_FILE = "outputs/base_lr_range_calibration.json"
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'base_lr_range_calibration.txt'), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_best_train_params():
    if os.path.exists(BEST_TRAIN_PARAMS_FILE):
        try:
            with open(BEST_TRAIN_PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load best train params: {e}")
    return {}

def run_training_trial(lr, model_name='EfficientNetV2B0', epochs=25):
    """Run training trial with specified LR"""
    best_params = load_best_train_params()
    
    cmd = [PYTHON_TRAIN, "components/train_multitask_trial.py", "--model_name", model_name]
    
    # Load default params but override relevant ones
    params = best_params.copy()
    params['learning_rate'] = lr
    params['epochs'] = epochs
    params['fine_tune'] = 'False'
    params['auto_lr_target_epoch'] = 0 # Manual LR control
    params['enable_early_stopping'] = 'True' # Valid for calibration
    
    for k, v in params.items():
        cmd.extend([f"--{k}", str(v)])
        
    logger.info(f"  Training with LR={lr:.8f}...")
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    
    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if any(kw in line for kw in ['BEST_EPOCH', 'MinClassAcc', 'FINAL_VAL_ACCURACY']):
             logger.info(f"    [Train] {line}")
             
    process.wait()
    full_output = "\n".join(output_lines)
    
    match_score = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
    score = float(match_score.group(1)) if match_score else 0.0
    
    match_epoch = re.search(r"BEST_EPOCH:\s*(\d+)", full_output, re.IGNORECASE)
    best_epoch = int(match_epoch.group(1)) if match_epoch else epochs
    
    logger.info(f"  Result: Score={score:.4f}, BestEpoch={best_epoch}")
    return score, best_epoch

def calibrate_lr_for_target(target_epoch, initial_lr):
    """Find LR that makes BestEpoch close to target_epoch"""
    logger.info(f"\n--- Calibrating LR for Target Epoch {target_epoch} ---")
    
    current_lr = initial_lr
    best_res = None
    min_dist = float('inf')
    
    history = {}
    
    max_iter = 5
    cal_epochs = target_epoch + 10 # 余裕を持たせる
    
    for i in range(max_iter):
        if current_lr in history:
            score, epoch = history[current_lr]
            logger.info(f"  (Cached) LR={current_lr:.8f} -> Epoch={epoch}")
        else:
            score, epoch = run_training_trial(current_lr, epochs=cal_epochs)
            history[current_lr] = (score, epoch)
            
        dist = abs(epoch - target_epoch)
        
        # Keep best match (closest epoch, then highest score)
        if best_res is None:
            best_res = (current_lr, score, epoch)
            min_dist = dist
        else:
            if dist < min_dist:
                min_dist = dist
                best_res = (current_lr, score, epoch)
            elif dist == min_dist:
                if score > best_res[1]:
                    best_res = (current_lr, score, epoch)
        
        logger.info(f"  Iter {i+1}: LR={current_lr:.8f} -> Epoch={epoch} (Target={target_epoch}, Dist={dist})")
        
        if dist == 0:
            logger.info("  Target Matched!")
            break
            
        # Adjust LR
        # High LR -> Fast convergence (Small Epoch)
        # Low LR -> Slow convergence (Large Epoch)
        
        if epoch < target_epoch:
            # Too fast (LR too high) -> Decrease LR
            # If dist is large, decrease aggressively
            factor = 0.5 if dist > 5 else 0.8
        else:
            # Too slow (LR too low) -> Increase LR
            factor = 2.0 if dist > 5 else 1.25
            
        new_lr = current_lr * factor
        current_lr = new_lr
        
    logger.info(f"  => Best for Target {target_epoch}: LR={best_res[0]:.8f}, Score={best_res[1]:.4f}, Epoch={best_res[2]}")
    return best_res

def main():
    logger.info("Starting Base LR Range Calibration (Epochs 15-20)")
    
    targets = [15, 16, 17, 18, 19, 20]
    results = {}
    
    # Initial guess
    best_params = load_best_train_params()
    current_lr_guess = best_params.get('learning_rate', 0.0001)
    
    # 昇順で探索することで、LRの変動推移を利用できる
    # (Epochが増える -> LRは下がる傾向にあるはず)
    
    for t in targets:
        lr, score, epoch = calibrate_lr_for_target(t, current_lr_guess)
        results[t] = {
            'learning_rate': lr,
            'score': score,
            'epoch': epoch
        }
        # Next guess: start from current best LR
        current_lr_guess = lr
        
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    
    for t in targets:
        res = results[t]
        logger.info(f"Target Epoch {t}: LR={res['learning_rate']:.8f}, Score={res['score']:.4f} (Actual Epoch {res['epoch']})")
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
