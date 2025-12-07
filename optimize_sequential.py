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
PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
CACHE_FILE = "filter_opt_cache.json"
DATA_SOURCE_DIR = "train" # ファイル数カウント用

# パスが存在しない場合はデフォルトを使用
if not os.path.exists(PYTHON_PREPROCESS): PYTHON_PREPROCESS = "python"
if not os.path.exists(PYTHON_TRAIN): PYTHON_TRAIN = "python"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('sequential_opt_log.txt', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def count_files(directory):
    """ディレクトリ以下のファイル総数をカウント（高速化のため拡張子フィルタは簡易的）"""
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

def run_trial(pitch, sym, y_diff):
    """
    指定されたパラメータで前処理と学習を実行し、スコアを返す
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating: Pitch={pitch}%, Sym={sym}%, Y-Diff={y_diff}%")
    
    # キャッシュ確認
    file_count = count_files(DATA_SOURCE_DIR)
    cache_key = f"pitch={pitch}_sym={sym}_ydiff={y_diff}_count={file_count}"
    
    cache = load_cache()
    if cache_key in cache:
        logger.info(f"Cache Hit! Skipping execution. Score: {cache[cache_key]}")
        logger.info(f"{'='*50}")
        return cache[cache_key]

    logger.info(f"Cache Miss. Running process... (File Count: {file_count})")
    logger.info(f"{'='*50}")

    try:
        # 1. 前処理
        cmd_pre = [
            PYTHON_PREPROCESS,
            "preprocess_multitask.py",
            "--pitch_percentile", str(pitch),
            "--symmetry_percentile", str(sym),
            "--y_diff_percentile", str(y_diff)
        ]
        logger.info("Running preprocessing...")
        ret_pre = subprocess.run(cmd_pre, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret_pre.returncode != 0:
            logger.error(f"Preprocessing failed: {ret_pre.stderr}")
            return 0.0

        # 2. 学習
        cmd_train = [PYTHON_TRAIN, "train_for_filter_search.py"]
        logger.info("Running training...")
        ret_train = subprocess.run(cmd_train, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret_train.returncode != 0:
            logger.error(f"Training failed: {ret_train.stderr}")
            return 0.0

        # 3. スコア抽出
        if match:
            score = float(match.group(1))
            logger.info(f"Result: Score = {score}")
            
            # キャッシュ保存
            cache = load_cache() # リロードして競合低減
            cache[cache_key] = score
            save_cache(cache)
            
            return score
        else:
            logger.error("Score not found in training output.")
            return 0.0

    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return 0.0

def optimize_single_param(target_name, current_params, min_val=0, max_val=50):
    """
    1つのパラメータを「粗い探索 -> 細かい探索」で最適化する
    """
    logger.info(f"\n>>> Optimizing {target_name} (Range: {min_val}-{max_val}) <<<")
    
    best_val = current_params[target_name]
    best_score = -1.0
    
    # 探索済みキャッシュ
    history = {}

    def evaluate_wrapper(val):
        if val in history:
            return history[val]
        
        # パラメータ設定
        params = current_params.copy()
        params[target_name] = val
        
        score = run_trial(params['pitch'], params['sym'], params['y_diff'])
        history[val] = score
        return score

    # --- Step 1: 粗い探索 (0, 25, 50) ---
    points = [min_val, (min_val + max_val) // 2, max_val]
    logger.info(f"Step 1: Coarse search at {points}")
    
    scores = {}
    for p in points:
        scores[p] = evaluate_wrapper(p)
    
    # 最良の点を見つける
    best_coarse_val = max(scores, key=scores.get)
    best_coarse_score = scores[best_coarse_val]
    
    logger.info(f"Best coarse value: {best_coarse_val} (Score: {best_coarse_score})")

    # --- Step 2: リファインメント (上位2点の中間探索) ---
    logger.info("Step 2: Binary Search Refinement")
    
    while True:
        # スコア順にソート (降順)
        sorted_history = sorted(history.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_history) < 2:
            break
            
        best1_val, best1_score = sorted_history[0]
        best2_val, best2_score = sorted_history[1]
        
        # 中間値を計算 (整数、切り捨て)
        mid_val = int((best1_val + best2_val) / 2)
        
        # 収束判定: 中間値が既に評価済みなら終了
        if mid_val in history:
            logger.info(f"Refinement converged at {mid_val}. (Already evaluated)")
            break
            
        logger.info(f"Refining: Best1={best1_val}({best1_score:.4f}), Best2={best2_val}({best2_score:.4f}) -> Next: {mid_val}")
        
        # 新しい値を評価
        scores[mid_val] = evaluate_wrapper(mid_val)
            
    # 現時点でのベスト
    best_val = max(scores, key=scores.get)
    best_score = scores[best_val]
    
    logger.info(f"Finished optimizing {target_name}. Best: {best_val} (Score: {best_score})")
    return best_val, best_score

def main():
    logger.info("Starting Sequential Optimization")
    
    # 初期パラメータ
    current_params = {'pitch': 0, 'sym': 0, 'y_diff': 0}
    
    # 1. Pitch の最適化
    best_pitch, _ = optimize_single_param('pitch', current_params)
    current_params['pitch'] = best_pitch
    
    # 2. Symmetry の最適化 (Pitchは最適値で固定)
    best_sym, _ = optimize_single_param('sym', current_params)
    current_params['sym'] = best_sym
    
    # 3. Y-Diff の最適化 (Pitch, Symは最適値で固定, max=75)
    best_ydiff, final_score = optimize_single_param('y_diff', current_params, max_val=75)
    current_params['y_diff'] = best_ydiff
    
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Final Best Params: {current_params}")
    logger.info(f"Final Score: {final_score}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
