import subprocess
import sys
import re
import os
import logging
import json
import hashlib

# --- 設定 ---
PYTHON_EXEC = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
DATA_SOURCE_DIR = "preprocessed_multitask/train" # ファイル数カウント用

# 出力ディレクトリ
LOG_DIR = "outputs/logs"
CACHE_DIR = "outputs/cache"
MODEL_DIR = "outputs/models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "train_opt_cache.json")

if not os.path.exists(PYTHON_EXEC): PYTHON_EXEC = "python"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'sequential_train_opt_log.txt'), mode='w', encoding='utf-8'),
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

    cmd = [PYTHON_EXEC, "components/train_single_trial.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    try:
        ret = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret.returncode != 0:
            logger.error(f"Training failed: {ret.stderr}")
            return 0.0

        # スコア抽出
        # スコア抽出 (Task A)
        match_a = re.search(r"FINAL_VAL_ACCURACY:\s*(\d+\.\d+)", ret.stdout)
        
        # 他のタスクも抽出してログに出す
        for task_label in ['A', 'B', 'C', 'D']:
            match_task = re.search(f"TASK_{task_label}_ACCURACY:\s*(\d+\.\d+)", ret.stdout)
            if match_task:
                logger.info(f"Task {task_label} Accuracy: {float(match_task.group(1))}")

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
        'rotation_range': 0.0,
        'width_shift_range': 0.0,
        'height_shift_range': 0.0,
        'zoom_range': 0.0,
        'horizontal_flip': 'False',
        'fine_tune': 'False'
    }
    
    # --- Step 1: Learning Rate ---
    best_lr, _ = optimize_param('learning_rate', [1e-3, 5e-4, 1e-4], current_params)
    current_params['learning_rate'] = best_lr
    
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

    # --- Step 3: Data Augmentation ---
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
    
    # --- Step 4: Final Fine-Tuning ---
    current_params['fine_tune'] = 'True'
    current_params['epochs'] = 50 # エポック数を増やす
    
    # ファインチューニングは1回だけ実行（最適化されたパラメータで）
    final_ft_score = run_trial(current_params)
    
    logger.info("\n" + "="*50)
    logger.info("ALL PROCESSES COMPLETE")
    logger.info(f"Final Fine-Tuned Score: {final_ft_score}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
