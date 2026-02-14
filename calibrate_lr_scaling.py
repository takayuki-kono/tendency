"""
LRスケーリング指数キャリブレーション

フィルタによりデータ量が変わった際の学習率調整:
  adjusted_lr = base_lr / ((ratio^2)^exponent)
             = base_lr / (ratio^(2 * exponent))

各フィルタレベル (0/5/40/80%) で LR を epoch 10 でベストになるようキャリブレーションし、
生スコアを epoch差で減衰した実効スコアが最も高い exponent を採用する:
  weighted_score = raw_score / ((abs(best_epoch - target_epoch) + 1) ** exponent)

探索範囲は exponent = 0.5〜0.8 を二分探索する。

結果は outputs/lr_scaling_config.json に保存され、
optimize_sequential.py が読み込んで使用する。
"""
import subprocess
import sys
import re
import os
import logging
import json
import math
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
RATIO_BASE_POWER = 2.0


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

    ratio = saved_images / total_images if total_images > 0 else 1.0
    logger.info(f"  Total={total_images}, Saved={saved_images}, Ratio={ratio:.4f}")
    return total_images, saved_images


def run_training_trial(lr, model_name='EfficientNetV2B0', epochs=20):
    """指定LRで学習を実行してスコアとbest_epochを返す"""
    best_params = load_best_train_params()

    cmd = [PYTHON_TRAIN, "components/train_multitask_trial.py", "--model_name", model_name]
    for k, v in best_params.items():
        if k not in ['model_name', 'fine_tune', 'epochs', 'learning_rate', 'auto_lr_target_epoch', 'seed']:
            cmd.extend([f"--{k}", str(v)])
    cmd.extend(["--learning_rate", str(lr)])
    cmd.extend(["--epochs", str(epochs)])
    cmd.extend(["--fine_tune", "False"])

    logger.info(f"  Training with LR={lr:.8f}...")

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

    match_epoch = re.search(r"BEST_EPOCH:\s*(\d+)", full_output)
    best_epoch = int(match_epoch.group(1)) if match_epoch else epochs

    logger.info(f"  Score: {score:.4f}, BestEpoch: {best_epoch}/{epochs}")
    return score, best_epoch


def distance_weighted_score(raw_score, best_epoch, target_epoch, exponent):
    """best_epoch が target_epoch に近いほど高くなる重み付きスコア"""
    distance = abs(best_epoch - target_epoch)
    return raw_score / ((distance + 1.0) ** exponent)


def calibrate_lr_for_target(initial_lr, target_epoch=10, model_name='EfficientNetV2B0',
                            cal_epochs=20, max_iter=3, distance_exponent=0.5,
                            epoch_scale_exponent=0.75):
    """
    target_epochでベストになるLRをキャリブレーションし、(lr, score, best_epoch)を返す。
    候補選択は epoch差を加味した重み付きスコアを優先する。
    """
    current_lr = initial_lr
    best_candidate = None  # (-weighted_score, distance, -raw_score, lr, best_epoch, raw_score, weighted_score)

    for i in range(max_iter):
        score, best_epoch = run_training_trial(current_lr, model_name, cal_epochs)
        distance = abs(best_epoch - target_epoch)
        weighted = distance_weighted_score(score, best_epoch, target_epoch, distance_exponent)
        candidate = (-weighted, distance, -score, current_lr, best_epoch, score, weighted)
        if best_candidate is None or candidate < best_candidate:
            best_candidate = candidate

        logger.info(
            f"  [Cal #{i+1}] LR={current_lr:.8f}, BestEpoch={best_epoch}/{cal_epochs}, "
            f"Score={score:.4f}, WeightedScore={weighted:.4f}"
        )

        if best_epoch == target_epoch:
            break

        # LRスケーリング: epoch比のべき指数
        raw_scale = best_epoch / target_epoch
        scale = raw_scale ** epoch_scale_exponent if raw_scale > 0 else 0.5
        scale = max(0.5, min(scale, 2.0))
        new_lr = current_lr * scale
        if abs(new_lr - current_lr) / max(current_lr, 1e-12) < 0.01:
            break
        current_lr = new_lr

    _, _, _, chosen_lr, chosen_epoch, chosen_score, chosen_weighted_score = best_candidate
    logger.info(
        f"  Calibrated: LR={chosen_lr:.8f}, BestEpoch={chosen_epoch}, "
        f"Score={chosen_score:.4f}, WeightedScore={chosen_weighted_score:.4f}"
    )
    return chosen_lr, chosen_score, chosen_epoch


def main():
    logger.info("=" * 60)
    logger.info("LR Scaling Exponent Calibration")
    logger.info("=" * 60)

    model_name = 'EfficientNetV2B0'
    target_epoch = 10
    cal_epochs = 20

    # --- Step 1: ベースライン（フィルタなし）---
    logger.info("\n>>> Step 1: Baseline (no filter, calibrate to epoch 10) <<<")

    # キャリブレーション済みLRキャッシュを確認
    cal_cache_file = "outputs/cache/calibrated_lr.json"
    base_lr = None
    base_score = None

    if os.path.exists(cal_cache_file):
        try:
            with open(cal_cache_file, 'r', encoding='utf-8') as f:
                cal_cache = json.load(f)
            base_lr = cal_cache['lr']
            base_score = cal_cache['score']
            logger.info(f"LR Calibration Cache Hit! LR={base_lr:.8f}, Score={base_score:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load LR calibration cache: {e}")

    if base_lr is None:
        run_preprocess(0)
        best_params = load_best_train_params()
        initial_lr = best_params.get('learning_rate', 0.0001)
        base_lr, base_score, _ = calibrate_lr_for_target(
            initial_lr, target_epoch, model_name, cal_epochs
        )
        logger.info(f"Baseline: LR={base_lr:.8f}, Score={base_score:.4f}")

    if base_score <= 0:
        logger.error("Baseline score is 0. Aborting.")
        return

    logger.info(f"Base LR: {base_lr:.8f}, Base Score: {base_score:.4f}")

    # --- Step 2: フィルタレベル準備 ---
    logger.info("\n>>> Step 2: Prepare filter levels <<<")
    filter_percentiles = [0, 5, 40, 80]
    filter_levels = []  # (filter_pct, saved_ratio)

    for pct in filter_percentiles:
        total, saved = run_preprocess(pct)
        if total > 0 and saved > 0:
            ratio = saved / total
            filter_levels.append((pct, ratio))
        else:
            logger.warning(f"Filter {pct}% produced no data, skipping.")

    if len(filter_levels) == 0:
        logger.error("No valid filter levels. Aborting.")
        return

    logger.info(f"Filter levels: {filter_levels}")

    # --- Step 3: 二分探索でベスト指数を探す ---
    logger.info("\n>>> Step 3: Binary search for best exponent (epoch-proximity weighted score) <<<")
    lo, hi = 0.5, 0.8
    history = []

    def evaluate_exponent(exp):
        """指数を評価し、全フィルタの平均重み付きスコアを返す"""
        logger.info(f"\n--- Exponent = {exp:.4f} ---")
        total_weighted_score = 0.0
        total_raw_score = 0.0
        level_results = []
        for filter_pct, ratio in filter_levels:
            run_preprocess(filter_pct)
            effective_ratio_power = RATIO_BASE_POWER * exp
            initial_lr = base_lr / (ratio ** effective_ratio_power)
            logger.info(
                f"  Filter={filter_pct}%, Ratio={ratio:.4f}, "
                f"Power=({RATIO_BASE_POWER:.1f}*{exp:.4f})={effective_ratio_power:.4f}, "
                f"InitialLR={initial_lr:.8f}"
            )
            cal_lr, cal_score, cal_epoch = calibrate_lr_for_target(
                initial_lr, target_epoch, model_name, cal_epochs, max_iter=3,
                distance_exponent=exp, epoch_scale_exponent=exp
            )
            distance = abs(cal_epoch - target_epoch)
            weighted_score = distance_weighted_score(cal_score, cal_epoch, target_epoch, exp)
            total_weighted_score += weighted_score
            total_raw_score += cal_score
            level_results.append({
                'filter_pct': filter_pct, 'ratio': ratio,
                'initial_lr': initial_lr, 'calibrated_lr': cal_lr,
                'ratio_base_power': RATIO_BASE_POWER,
                'effective_ratio_power': effective_ratio_power,
                'score': cal_score, 'best_epoch': cal_epoch,
                'distance': distance, 'weighted_score': weighted_score
            })
            logger.info(
                f"  -> Score={cal_score:.4f}, BestEpoch={cal_epoch}, "
                f"Distance={distance}, WeightedScore={weighted_score:.4f}"
            )
        avg_weighted_score = total_weighted_score / len(filter_levels)
        avg_raw_score = total_raw_score / len(filter_levels)
        logger.info(
            f"Exponent={exp:.4f}: AvgWeightedScore={avg_weighted_score:.4f}, "
            f"AvgRawScore={avg_raw_score:.4f}"
        )
        history.append({
            'exponent': exp,
            'avg_score': avg_weighted_score,
            'avg_raw_score': avg_raw_score,
            'details': level_results
        })
        return avg_weighted_score

    # 初回: lo, mid, hi の3点評価
    mid = (lo + hi) / 2
    scores = {}
    for exp in [lo, mid, hi]:
        scores[exp] = evaluate_exponent(exp)

    # 二分探索: スコア最高の2点の間を狭める（3回追加）
    for iteration in range(3):
        sorted_by_score = sorted(scores.items(), key=lambda x: -x[1])  # スコア降順
        top1_exp, top1_score = sorted_by_score[0]
        top2_exp, top2_score = sorted_by_score[1]

        # 上位2点の中間を次の探索点にする
        new_exp = (top1_exp + top2_exp) / 2

        # 既存の点に近すぎたらスキップ
        if any(abs(h['exponent'] - new_exp) < 0.02 for h in history):
            logger.info(f"  Skip: {new_exp:.4f} too close to existing point")
            # 代わりにベスト付近の反対側を試す
            if top1_exp > top2_exp:
                new_exp = top1_exp + (top1_exp - top2_exp) * 0.5
            else:
                new_exp = top1_exp - (top2_exp - top1_exp) * 0.5
            new_exp = max(lo, min(hi, new_exp))
            if any(abs(h['exponent'] - new_exp) < 0.02 for h in history):
                logger.info(f"  Skip iteration {iteration+1}: no new point to try")
                continue

        logger.info(f"\n[Binary Search #{iteration+1}] Trying {new_exp:.4f} "
                     f"(between {top1_exp:.4f}={top1_score:.4f} and {top2_exp:.4f}={top2_score:.4f})")
        scores[new_exp] = evaluate_exponent(new_exp)

    # --- Step 4: ベスト選択 ---
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION COMPLETE")
    logger.info("=" * 60)

    history_sorted = sorted(history, key=lambda x: -x['avg_score'])  # 重み付きスコア降順
    for h in history_sorted:
        logger.info(
            f"  Exp={h['exponent']:.4f}: AvgWeightedScore={h['avg_score']:.4f}, "
            f"AvgRawScore={h.get('avg_raw_score', 0.0):.4f}"
        )

    best = history_sorted[0]
    best_exp = best['exponent']
    best_avg_score = best['avg_score']

    logger.info(f"\nBest Exponent: {best_exp:.4f} (AvgScore={best_avg_score:.4f})")
    logger.info(
        f"Formula: adjusted_lr = base_lr / ((ratio^{RATIO_BASE_POWER:.1f})^{best_exp:.4f}) "
        f"= base_lr / (ratio^{RATIO_BASE_POWER * best_exp:.4f})"
    )

    # 保存
    result = {
        'exponent': best_exp,
        'ratio_base_power': RATIO_BASE_POWER,
        'effective_ratio_power': RATIO_BASE_POWER * best_exp,
        'base_lr': base_lr,
        'base_score': base_score,
        'best_avg_score': best_avg_score,
        'target_epoch': target_epoch,
        'search_range': [lo, hi],
        'score_metric': 'raw_score / ((abs(best_epoch-target_epoch)+1)^exponent)',
        'lr_formula': 'base_lr / ((ratio^2)^exponent) = base_lr / (ratio^(2*exponent))',
        'history': [
            {
                'exponent': h['exponent'],
                'avg_score': h['avg_score'],
                'avg_raw_score': h.get('avg_raw_score', 0.0)
            }
            for h in history_sorted
        ],
    }
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    logger.info(f"Result saved to {OUTPUT_FILE}")

    # フィルタなしに戻す
    run_preprocess(0)


if __name__ == "__main__":
    main()
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
