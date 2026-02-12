import subprocess
import sys
import re
import os
import logging
import json
import hashlib
import winsound

# --- 設定 ---
# SVMの学習はcomponents/train_svm.pyで行う
# 実行環境はGPU/CPUどちらでもよいが、preprocess環境(CPU/GPU)を使用する
PYTHON_EXEC = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe"
DATA_SOURCE_DIR = "preprocessed_multitask_svm/train" # SVM用のデータソース

# 出力ディレクトリ
LOG_DIR = "outputs/logs"
CACHE_DIR = "outputs/cache"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "svm_train_opt_cache.json")
BEST_PARAMS_FILE = "outputs/best_svm_train_params.json"

if not os.path.exists(PYTHON_EXEC): PYTHON_EXEC = "python"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'svm_train_seq_log.txt'), mode='w', encoding='utf-8'),
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
    SVMの場合、パラメータはKernel, C, Gammaなどが考えられるが、
    現在のtrain_svm.pyは引数を受け取っていないため、まずはデータソース指定のみとなる。
    今後train_svm.pyを拡張する場合に備えてパラメータ受け渡し構造にしておく。
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating Params: {params}")
    
    # キャッシュ確認
    params_str = json.dumps(params, sort_keys=True)
    file_count = count_files(DATA_SOURCE_DIR)
    
    key_src = f"{params_str}_count={file_count}"
    cache_key = hashlib.md5(key_src.encode('utf-8')).hexdigest()
    
    cache = load_cache()
    if cache_key in cache:
        logger.info(f"Cache Hit! Skipping execution. Val Accuracy: {cache[cache_key]}")
        return cache[cache_key]

    logger.info(f"Cache Miss. Running process... (File Count: {file_count})")
    
    # SVM学習スクリプトの実行
    # 現在のtrain_svm.pyは --data_dir 引数のみ受け取る
    # パラメータチューニングをするなら train_svm.py も修正が必要だが、
    # まずは現状のSVM学習をSequential Optimizationのフローに乗せる
    cmd = [PYTHON_EXEC, "components/train_svm.py", "--data_dir", "preprocessed_multitask_svm"]
    
    # train_svm.py supports arguments now.
    if 'C' in params:
        cmd.extend(["--C", str(params['C'])])
    if 'kernel' in params:
        cmd.extend(["--kernel", str(params['kernel'])])
    if 'gamma' in params:
        cmd.extend(["--gamma", str(params['gamma'])])

    try:
        ret = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret.returncode != 0:
            logger.error(f"Training failed: {ret.stderr}")
            return 0.0

        # スコア抽出
        match = re.search(r"FINAL_SCORE:\s*([\d.]+)", ret.stdout)
        
        if match:
            score = float(match.group(1))
            logger.info(f"Result: Val Score = {score}")
            
            cache = load_cache()
            cache[cache_key] = score
            save_cache(cache)
            
            return score
        else:
            logger.error("Score not found in output.")
            logger.error(f"STDOUT:\n{ret.stdout}")
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
    
    logger.info(f"Finished optimization. Best: {best_val} (Score: {best_score})")
    return best_val, best_score

def main():
    logger.info("Starting SVM Sequential Training Optimization")
    
    # 初期パラメータ
    current_params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    }
    
    # --- Step 1: Kernel Type ---
    # linear: 単純な分離、高速
    # rbf: 非線形、汎用的
    # poly: 多項式 (時間がかかる場合がある)
    best_kernel, _ = optimize_param('kernel', ['linear', 'rbf'], current_params)
    current_params['kernel'] = best_kernel
    
    # --- Step 2: C Parameter (Regularization) ---
    # Cが大きい: 誤分類を許さない（過学習のリスク）
    # Cが小さい: 誤分類を許容（滑らかな境界）
    best_c, _ = optimize_param('C', [0.1, 1.0, 10.0, 100.0], current_params)
    current_params['C'] = best_c
    
    # --- Step 3: Gamma (RBF kernel coefficient) ---
    # scale: 1 / (n_features * X.var())
    # auto: 1 / n_features
    if current_params['kernel'] == 'rbf':
        best_gamma, _ = optimize_param('gamma', ['scale', 'auto', 0.1, 0.01], current_params)
        current_params['gamma'] = best_gamma
    
    final_score = run_trial(current_params)
    
    logger.info("\n" + "="*50)
    logger.info("ALL PROCESSES COMPLETE")
    logger.info(f"Final Best SVM Score: {final_score}")
    logger.info(f"Best Params: {current_params}")
    logger.info("="*50)

    # Save Best Params
    with open(BEST_PARAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(current_params, f, indent=4)

if __name__ == "__main__":
    main()
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
