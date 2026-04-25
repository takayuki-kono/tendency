import subprocess
import sys
import re
import os
import logging
import math
import time
import json
import hashlib
import winsound

# --- 設定 ---
# Python実行環境のパス
PYTHON_PREPROCESS = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe"
PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
DATA_SOURCE_DIR = "train" 

# 出力ディレクトリ
LOG_DIR = "outputs/logs"
CACHE_DIR = "outputs/cache"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "filter_opt_cache.json")
BEST_TRAIN_PARAMS_FILE = "outputs/best_train_params.json"

# 効率ベースの最適化設定
# 効率 = 精度向上 / フィルタリング枚数
# 効率が低いパラメータは自動的にスケールダウンまたは除外される
MIN_EFFICIENCY_THRESHOLD = 0.00001  # 最小効率閾値（これ以下は除外候補）

# Validation クラス最小サンプル数ガード（small-sample ノイズ対策）
# 前処理後の validation で「いずれかのタスク×クラス」の画像数がこの値未満なら、
# その候補は学習せず score=0.0 固定で採点失敗扱い（キャッシュも固定）。
MIN_VAL_PER_CLASS = 20
# 旧形式 3-tuple キャッシュ（val_min_cnt 未記録）に対するヒューリスティック。
# saved_images = total - filtered がこの値未満なら val<20 の可能性が高いとみなして
# キャッシュヒット時に score=0.0 に上書きして無効化する。
# 係数 10 は "train/val/test 3 split × クラス数 × MIN_VAL_PER_CLASS" のおおよその下限を想定。
LEGACY_CACHE_MIN_SAVED = MIN_VAL_PER_CLASS * 10

# LRキャリブレーション結果（main()で設定される）
CALIBRATED_BASE_LR = None
LR_SCALING_CONFIG_FILE = "outputs/lr_scaling_config.json"
LR_SCALING_EXP = 0.65  # 単一exponent（relative_ratio ** exp でLRスケーリング）
BASE_RATIO = 1.0
LR_INDIVIDUAL_EXPONENTS = {}  # パラメータ別 exponent (param_key -> float)

# パスが存在しない場合はデフォルトを使用
if not os.path.exists(PYTHON_PREPROCESS): PYTHON_PREPROCESS = "python"
if not os.path.exists(PYTHON_TRAIN): PYTHON_TRAIN = "python"

# ログ設定
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_filename = f"sequential_opt_log_{timestamp}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from components.lr_adjustment import (
    LR_TARGET_EPOCH, LR_ACCEPTABLE_MIN, LR_ACCEPTABLE_MAX,
    LR_MAX_ADJUSTMENTS, LR_LAST_ACCU_EPS,
    compute_lr_adjustment_ratio, lr_adjustment_decision, lr_calibration_should_stop,
)


def count_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        count += len(files)
    return count


def _min_val_class_count(val_dir):
    """validation ディレクトリ配下の「各タスク×各クラス」の画像合計枚数の最小値を返す。

    フォルダ名の文字長が全て同じで 1 文字超なら multitask とみなし、各文字位置を
    別タスクとしてクラス集計する。1 文字または長さが揃わない場合は single-task
    とみなしフォルダ名をそのまま 1 クラスとして集計する。

    Returns:
        tuple: (min_count, min_desc, per_task_counts)
            - min_count: 最小枚数（int）。val_dir 不在 or 空なら None
            - min_desc: 最小を記録したタスク/クラスの説明文字列
            - per_task_counts: [dict(class_char -> count), ...] のリスト
    """
    if not os.path.isdir(val_dir):
        return None, None, None

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    subdirs = [d for d in os.listdir(val_dir)
               if os.path.isdir(os.path.join(val_dir, d))]
    if not subdirs:
        return 0, "(no class folders)", []

    combined_counts = {}
    for d in subdirs:
        p = os.path.join(val_dir, d)
        cnt = 0
        for root, _, files in os.walk(p):
            for f in files:
                if f.lower().endswith(img_exts):
                    cnt += 1
        combined_counts[d] = cnt

    lens = set(len(d) for d in subdirs)
    if len(lens) == 1 and next(iter(lens)) > 1:
        n_tasks = next(iter(lens))
        per_task_counts = [dict() for _ in range(n_tasks)]
        for label, cnt in combined_counts.items():
            for i, ch in enumerate(label):
                per_task_counts[i][ch] = per_task_counts[i].get(ch, 0) + cnt
    else:
        per_task_counts = [dict(combined_counts)]

    min_cnt, min_desc = None, None
    for i, cls_cnts in enumerate(per_task_counts):
        for c, cnt in cls_cnts.items():
            if min_cnt is None or cnt < min_cnt:
                min_cnt = cnt
                min_desc = f"task{i}={c} ({cnt})"
    return min_cnt, min_desc, per_task_counts

def _reset_cache_file(current_file_count: int) -> dict:
    """キャッシュJSONを初期化してディスクへ即時書き込む。"""
    fresh = {'__file_count__': current_file_count}
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(fresh, f, indent=4)
    except Exception as e:
        logger.warning(f"Failed to reset cache file on disk: {e}")
    return fresh

def load_cache():
    current_file_count = count_files("train") + count_files("validation")
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            # 壊れたキャッシュは破棄して新規化（ディスクも即時更新）
            return _reset_cache_file(current_file_count)

        prev_count = cache.get('__file_count__')
        # ファイル数が変わっていたらキャッシュクリア（ディスクも即時クリア）
        if prev_count != current_file_count:
            logger.info(
                f"Data changed ({prev_count} -> {current_file_count}). Clearing cache file on disk."
            )
            return _reset_cache_file(current_file_count)
        return cache
    return _reset_cache_file(current_file_count)

def load_best_train_params():
    """最適化された学習パラメータを読み込む"""
    if os.path.exists(BEST_TRAIN_PARAMS_FILE):
        try:
            with open(BEST_TRAIN_PARAMS_FILE, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"Loaded best training params: {params}")
            return params
        except Exception as e:
            logger.warning(f"Failed to load best train params: {e}")
    else:
        logger.info("Best train params not found. Using defaults.")
    return {}

def _get_head_lr_from_best(best_params: dict, default: float) -> float:
    # optimize_sequential は fine_tune=False（head-only）の評価が中心。
    # head のLRと本体(backbone/nohead)のLRは乖離する想定のため、head 側のみ参照する。
    # learning_rate_nohead は body 側の LR で head calibration では使わない。
    for k in ("learning_rate_head", "warmup_lr", "learning_rate"):
        if k in best_params:
            try:
                return float(best_params[k])
            except Exception:
                pass
    return float(default)

def save_best_train_params(params: dict) -> None:
    """最適化された学習パラメータを書き戻す（存在しない場合は新規作成）。"""
    os.makedirs(os.path.dirname(BEST_TRAIN_PARAMS_FILE), exist_ok=True)
    with open(BEST_TRAIN_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)

def save_cache(cache):
    # ファイル数も保存
    cache['__file_count__'] = count_files("train") + count_files("validation")
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

def run_calibration_trial(model_name, lr, cal_epochs=5):
    """
    LRキャリブレーション用: 前処理済みデータで学習のみ実行し、BEST_EPOCHを返す。
    preprocessed_multitask が既に準備されている前提で呼ぶこと。
    
    Returns:
        tuple: (best_epoch, score)
    """
    best_params = load_best_train_params()
    
    cmd_train = [PYTHON_TRAIN, "components/train_multitask_trial.py", "--model_name", model_name]
    # learning_rate_nohead/head/ft は optimize 側のメタ情報で train_multitask_trial には存在しない引数
    _skip_keys = {
        'model_name', 'fine_tune', 'epochs', 'learning_rate', 'auto_lr_target_epoch',
        'learning_rate_nohead', 'learning_rate_head', 'learning_rate_ft',
    }
    for k, v in best_params.items():
        if k not in _skip_keys:
            cmd_train.extend([f"--{k}", str(v)])
    cmd_train.extend(["--learning_rate", str(lr)])
    cmd_train.extend(["--epochs", str(cal_epochs)])
    cmd_train.extend(["--fine_tune", "False"])
    cmd_train.extend(["--enable_early_stopping", "False"])
    # Conditional Extension は train_multitask_trial 側で必要時のみ発動させる
    
    logger.info(f"[Calibration] Running {cal_epochs} epochs with LR={lr:.8f}...")
    
    process = subprocess.Popen(
        cmd_train, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    
    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        if any(kw in line for kw in ['Epoch ', 'BEST_EPOCH', 'MinClassAcc', 'Avg=']):
            logger.info(f"  [Cal] {line}")
    
    rc = process.wait()
    full_output = "\n".join(output_lines)
    if rc != 0:
        logger.error(
            f"[Calibration] train_multitask_trial 異常終了 (returncode={rc}). "
            f"先頭2KB:\n{full_output[:2000]}"
        )

    # BEST_EPOCH抽出
    match_epoch = re.search(r"BEST_EPOCH:\s*(\d+)", full_output)
    best_epoch = int(match_epoch.group(1)) if match_epoch else cal_epochs

    # スコア抽出
    match_score = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
    score = float(match_score.group(1)) if match_score else 0.0
    if match_score is None:
        logger.warning("[Calibration] FINAL_VAL_ACCURACY not found in output (score=0). Last 15 lines:")
        for line in output_lines[-15:]:
            logger.warning("  %s", line.rstrip())
    # 最終エポックの精度（trainのLR調整終了条件と合わせるため）
    epoch_scores = re.findall(r"MinClassAcc=([\d.]+)", full_output)
    last_epoch_accu = float(epoch_scores[-1]) if epoch_scores else 0.0

    logger.info(f"[Calibration] Result: BestEpoch={best_epoch}/{cal_epochs}, Score={score:.4f}")
    return best_epoch, score, last_epoch_accu


def calibrate_base_lr(model_name, initial_lr, cal_epochs=10, target_best_epoch=None):
    """
    cal_epochs の学習を繰り返し、target_best_epoch でベストになるLRを二分探索で探す。

    探索方式（train_sequential.py と統一）:
    - best_epoch < target_min → LR高すぎ → lr_high を current_lr で更新
    - best_epoch > target_max → LR低すぎ → lr_low を current_lr で更新
    - 両境界が揃ったら log 空間の中点（幾何平均 sqrt(lr_low * lr_high)）で次 LR 決定
      （LR は乗算スケールで効くため、算術平均より幾何平均の方が収束が速い）
    - 片側のみの場合は raw ratio (best_epoch/target) でスケーリング（クランプなし）

    target_best_epoch:
    - None             : cal_epochs//2 を target とする
    - int              : 単一 target（target_min=target_max）
    - tuple (min,max)  : 許容帯。帯内に落ちたらベスト候補として採用

    Returns:
        tuple: (calibrated_lr, final_score)
    """
    if target_best_epoch is None:
        target_min = target_max = float(max(1, cal_epochs // 2))
    elif isinstance(target_best_epoch, tuple):
        target_min = float(target_best_epoch[0])
        target_max = float(target_best_epoch[1])
    else:
        target_min = target_max = float(target_best_epoch)

    current_lr = initial_lr
    target_mid = (target_min + target_max) / 2.0

    logger.info(f"\n{'='*50}")
    logger.info(f"LR Calibration Start (bisection, log-space)")
    if target_min == target_max:
        logger.info(f"  cal_epochs={cal_epochs}, target_best_epoch={target_min:.0f}")
    else:
        logger.info(f"  cal_epochs={cal_epochs}, target_best_epoch=[{target_min:.0f}, {target_max:.0f}]")
    logger.info(f"  initial_lr = {initial_lr:.8f}")
    logger.info(f"{'='*50}")

    best_candidate = None  # (distance, -score, lr, best_epoch, score)
    lr_low = None   # best_epoch >= target_min になった最大LR（LR低すぎ or 帯内）
    lr_high = None  # best_epoch <  target_min になった最小LR（LR高すぎ）
    max_iterations = LR_MAX_ADJUSTMENTS + 1

    for iteration in range(max_iterations):
        best_epoch, score, last_epoch_accu = run_calibration_trial(model_name, current_lr, cal_epochs)

        if best_epoch < target_min:
            distance = target_min - best_epoch
        elif best_epoch > target_max:
            distance = best_epoch - target_max
        else:
            distance = 0.0

        # 距離最優先、同率なら score 高い方を採用
        candidate = (distance, -score, current_lr, best_epoch, score)
        if best_candidate is None or candidate < best_candidate:
            best_candidate = candidate

        logger.info(
            f"Calibration #{iteration+1}: LR={current_lr:.8f}, "
            f"BestEpoch={best_epoch}/{cal_epochs}, Score={score:.4f}, Distance={distance:.0f}"
        )

        should_stop, stop_msg = lr_calibration_should_stop(best_epoch, last_epoch_accu, score)
        if should_stop and stop_msg:
            logger.info(stop_msg)
            break
        if iteration >= LR_MAX_ADJUSTMENTS:
            logger.info(f"Reached max LR adjustments ({LR_MAX_ADJUSTMENTS}). Stopping calibration.")
            break

        # 境界更新
        if best_epoch < target_min:
            if lr_high is None or current_lr < lr_high:
                lr_high = current_lr
        else:
            # best_epoch >= target_min（帯内含む） → LR 低すぎ寄り
            if lr_low is None or current_lr > lr_low:
                lr_low = current_lr

        # 次のLR決定
        if lr_low is not None and lr_high is not None:
            # 両境界あり → 幾何平均（log空間の中点）
            new_lr = math.sqrt(lr_low * lr_high)
            logger.info(f"  Bisection (geom): low={lr_low:.8f}, high={lr_high:.8f}, mid={new_lr:.8f}")
        else:
            # 片側のみ → raw ratio scaling（クランプなし）
            # クランプを外した理由: 二分探索に入ると振動せず単調収束するため、
            # 片側探索でも大胆に動いた方が早く反対側境界を踏める。
            scale = compute_lr_adjustment_ratio(best_epoch, target_epoch=int(target_mid), total_epochs=cal_epochs)
            new_lr = current_lr * scale
            logger.info(
                f"  Ratio scaling (clamp 0.5-2.0): scale={scale:.4f}"
            )

        logger.info(f"  LR: {current_lr:.8f} -> {new_lr:.8f}")

        # 収束判定: LR変化が十分小さい
        if abs(new_lr - current_lr) / max(current_lr, 1e-12) < 0.02:
            logger.info(f"  LR change too small. Stopping.")
            break

        current_lr = new_lr

    if best_candidate is not None:
        _, _, chosen_lr, chosen_epoch, chosen_score = best_candidate
        logger.info(
            f"Calibration selected best candidate: LR={chosen_lr:.8f}, "
            f"BestEpoch={chosen_epoch}/{cal_epochs}, Score={chosen_score:.4f}"
        )
        current_lr, score = chosen_lr, chosen_score

    logger.info(f"\nCalibrated Base LR: {current_lr:.8f}, Final Score: {score:.4f}")
    logger.info(f"{'='*50}")
    return current_lr, score


def run_trial(pitch, sym, y_diff, mouth_open, eb_eye_high, eb_eye_low, sharpness_low, sharpness_high, face_size_low=0, face_size_high=0, retouching=0, mask=0, glasses=0, grayscale=False, model_name='EfficientNetV2B0', active_param_name=None):
    """
    指定されたパラメータとモデルで前処理と学習を実行し、スコアを返す
    
    Returns:
        tuple: (raw_score, total_images, filtered_count)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating: Model={model_name}, Pitch={pitch}%, Sym={sym}%, Y-Diff={y_diff}%, Mouth-Open={mouth_open}%, Eb-High={eb_eye_high}%, Eb-Low={eb_eye_low}%, Sharp-L={sharpness_low}%, Sharp-H={sharpness_high}%, FaceSize-L={face_size_low}%, FaceSize-H={face_size_high}%, Retouch={retouching}%, Mask={mask}%, Glasses={glasses}%, Grayscale={grayscale}")
    
    file_count = count_files("train") + count_files("validation")
    # キャッシュキーにLR情報を含める（キャリブレーション後はLRが変わるため）
    lr_tag = f"_clr={CALIBRATED_BASE_LR:.8f}" if CALIBRATED_BASE_LR else ""
    
    # 動的加重平均されたExponentを事前計算してキャッシュキーに含める
    mapping = {
        'y_diff': 'y_diff_percentile', 'sym': 'symmetry_percentile', 'pitch': 'pitch_percentile',
        'sharpness_low': 'sharpness_percentile_low', 'sharpness_high': 'sharpness_percentile_high',
        'eb_eye_high': 'eyebrow_eye_percentile_high', 'eb_eye_low': 'eyebrow_eye_percentile_low',
        'face_size_low': 'face_size_percentile_low', 'face_size_high': 'face_size_percentile_high',
        'mouth_open': 'mouth_open_percentile', 'retouching': 'retouching_percentile',
        'mask': 'mask_percentile', 'glasses': 'glasses_percentile'
    }
    current_vals = {
        'pitch': pitch, 'sym': sym, 'y_diff': y_diff, 'mouth_open': mouth_open,
        'eb_eye_high': eb_eye_high, 'eb_eye_low': eb_eye_low, 
        'sharpness_low': sharpness_low, 'sharpness_high': sharpness_high,
        'face_size_low': face_size_low, 'face_size_high': face_size_high,
        'retouching': retouching, 'mask': mask, 'glasses': glasses
    }
    
    total_weighted_exp = 0.0
    total_power = 0.0
    
    for param_key, power in current_vals.items():
        try:
            power_val = float(power)
        except ValueError:
            power_val = 0.0
        
        if power_val > 0:
            mapped_key = mapping.get(param_key) or param_key
            if not mapped_key.endswith('_percentile') and not mapped_key.endswith('_low') and not mapped_key.endswith('_high'):
                 mapped_key += '_percentile'
                 
            if mapped_key in LR_INDIVIDUAL_EXPONENTS:
                p_exp = float(LR_INDIVIDUAL_EXPONENTS[mapped_key])
                total_weighted_exp += p_exp * power_val
                total_power += power_val
                
    cache_exp_tag = f"_exp={total_weighted_exp/total_power:.4f}" if total_power > 0 else f"_exp_def"
    
    # 互換性のため、以前の形式のキャッシュキーも生成
    cache_key_legacy = f"model={model_name}_pitch={pitch}_sym={sym}_ydiff={y_diff}_mouth={mouth_open}_ebh={eb_eye_high}_ebl={eb_eye_low}_sharplow={sharpness_low}_sharphigh={sharpness_high}_fsl={face_size_low}_fsh={face_size_high}_retouch={retouching}_mask={mask}_glasses={glasses}_gray={grayscale}{lr_tag}_cnt={file_count}"
    
    # 新しい形式のキャッシュキー (exp値加味)
    cache_key = cache_key_legacy + cache_exp_tag
    
    cache = load_cache()
    
    # 新キー -> 旧キー の順で検索
    hit_key = None
    if cache_key in cache:
        hit_key = cache_key
    elif cache_key_legacy in cache:
        hit_key = cache_key_legacy
        logger.info(f"Using legacy cache key (no _exp tag)")
        
    if hit_key:
        cached = cache[hit_key]
        # キャッシュ値のフォーマット:
        #   4-tuple: (raw_score, total, filtered, val_min_cnt)  ← 新形式（val ガード付き）
        #   3-tuple: (raw_score, total, filtered)               ← 旧形式（val_min_cnt 未記録）
        #   2-tuple: (raw_score, filtering_rate)                ← 最古形式
        if isinstance(cached, (list, tuple)) and len(cached) >= 2:
            cached_val_min = None
            if len(cached) >= 4:
                raw_score, total_images, filtered_count, cached_val_min = cached[0], cached[1], cached[2], cached[3]
            elif len(cached) == 3:
                raw_score, total_images, filtered_count = cached
            else:
                raw_score, filtering_rate = cached
                total_images = file_count
                filtered_count = int(total_images * filtering_rate)

            # --- Validation クラス最小枚数ガード（キャッシュ経路）---
            if cached_val_min is not None:
                # 新形式: 記録済み val_min_cnt で厳密判定
                if cached_val_min < MIN_VAL_PER_CLASS:
                    logger.warning(
                        f"Cache Hit but INVALIDATED (val undersize): "
                        f"val_min={cached_val_min} < {MIN_VAL_PER_CLASS}. "
                        f"Returning score=0.0 (Total={total_images}, Filtered={filtered_count})"
                    )
                    return (0.0, total_images, filtered_count)
            else:
                # 旧形式 (3-tuple): val_min_cnt 未記録のため saved_images ヒューリスティックで推定
                saved_images_cached = total_images - filtered_count
                if total_images > 0 and saved_images_cached < LEGACY_CACHE_MIN_SAVED:
                    logger.warning(
                        f"Cache Hit but INVALIDATED (legacy 3-tuple, saved={saved_images_cached} "
                        f"< {LEGACY_CACHE_MIN_SAVED}): likely val<{MIN_VAL_PER_CLASS}. "
                        f"Returning score=0.0 (Total={total_images}, Filtered={filtered_count})"
                    )
                    return (0.0, total_images, filtered_count)

            logger.info(f"Cache Hit! RawScore={raw_score:.4f}, Total={total_images}, Filtered={filtered_count}")
            return (raw_score, total_images, filtered_count)
        else:
            # 2-tuple 未満の最古互換: val_min_cnt も saved も不明のためそのまま返す
            logger.info(f"Cache Hit (legacy format)! Score: {cached}")
            return (cached, file_count, 0)

    try:
        # Preprocess
        cmd_pre = [
            PYTHON_PREPROCESS,
            "preprocess_multitask.py",
            "--out_dir", "preprocessed_multitask",
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
        
        if ret_pre.stdout:
            logger.info(f"Preprocessing STDOUT:\n{ret_pre.stdout}")
        if ret_pre.stderr:
            logger.warning(f"Preprocessing STDERR:\n{ret_pre.stderr}")

        if ret_pre.returncode != 0:
            logger.error(f"Preprocessing failed with return code {ret_pre.returncode}")
            return (0.0, 0, 0)

        # フィルタリング枚数を計算（前処理の出力からTotal/Savedを取得）
        # logger.infoはstderrに出力されるため、stdout + stderr の両方を検索
        preprocess_output = (ret_pre.stdout or "") + "\n" + (ret_pre.stderr or "")
        total_images = 0
        saved_images = 0
        for line in preprocess_output.split('\n'):
            # "Processed {src_root}: Total={total}, Saved={saved_count}, Skipped={skipped_count}"
            match_stats = re.search(r'Total=(\d+), Saved=(\d+)', line)
            if match_stats:
                total_images += int(match_stats.group(1))
                saved_images += int(match_stats.group(2))
        
        filtered_count = total_images - saved_images
        if total_images > 0:
            logger.info(f"Filtering Stats: Total={total_images}, Saved={saved_images}, Filtered={filtered_count} ({filtered_count/total_images*100:.1f}%)")

        # --- Validation クラス最小サンプル数ガード ---
        # preprocess 後の validation で「いずれかのタスク×クラス」が MIN_VAL_PER_CLASS 未満なら、
        # 学習せず score=0.0 固定で採点失敗扱い（キャッシュも固定して再評価を防止）。
        val_dir = os.path.join("preprocessed_multitask", "validation")
        val_min_cnt, val_min_desc, val_per_task = _min_val_class_count(val_dir)
        if val_min_cnt is not None and val_min_cnt < MIN_VAL_PER_CLASS:
            logger.warning(
                f"SKIP (val undersize): min class size={val_min_cnt} < {MIN_VAL_PER_CLASS} "
                f"[{val_min_desc}]. per_task={val_per_task}"
            )
            result = (0.0, total_images, filtered_count)
            cache = load_cache()
            # 4-tuple で保存 (val_min_cnt を記録)
            cache[cache_key] = (0.0, total_images, filtered_count, int(val_min_cnt))
            save_cache(cache)
            return result
        elif val_min_cnt is not None:
            logger.info(
                f"Val class size OK: min={val_min_cnt} [{val_min_desc}] (threshold={MIN_VAL_PER_CLASS})"
            )

        # Train with specific Model and Optimized Params
        train_script = "components/train_multitask_trial.py"
        cmd_train = [PYTHON_TRAIN, train_script, "--model_name", model_name]
        
        # Load best params and append to command
        # epochs, fine_tune, model_name, learning_rate は明示的に設定するので除外
        # learning_rate_nohead/head/ft は optimize 側のメタ情報で train_multitask_trial には存在しない引数
        best_params = load_best_train_params()
        _skip_keys = {
            'model_name', 'fine_tune', 'epochs', 'learning_rate', 'auto_lr_target_epoch',
            'learning_rate_nohead', 'learning_rate_head', 'learning_rate_ft',
        }
        for k, v in best_params.items():
            if k not in _skip_keys:
                cmd_train.extend([f"--{k}", str(v)])
        
        # 学習率の設定 (2026-02-14 改良):
        if CALIBRATED_BASE_LR is not None and total_images > 0 and saved_images > 0:
            ratio = saved_images / total_images
            safe_ratio = max(ratio, 0.001)
            
            # Use relative ratio (ratio / BASE_RATIO)
            relative_ratio = safe_ratio / BASE_RATIO if BASE_RATIO > 0 else safe_ratio

            # 単一exponent: パラメータ別に設定されていれば重み付き平均、なければ LR_SCALING_EXP
            mapping = {
                'y_diff': 'y_diff_percentile',
                'sym': 'symmetry_percentile',
                'pitch': 'pitch_percentile',
                'sharpness_low': 'sharpness_percentile_low',
                'sharpness_high': 'sharpness_percentile_high',
                'eb_eye_high': 'eyebrow_eye_percentile_high',
                'eb_eye_low': 'eyebrow_eye_percentile_low',
                'face_size_low': 'face_size_percentile_low',
                'face_size_high': 'face_size_percentile_high',
                'mouth_open': 'mouth_open_percentile',
                'retouching': 'retouching_percentile',
                'mask': 'mask_percentile',
                'glasses': 'glasses_percentile'
            }
            
            current_vals = {
                'pitch': pitch, 'sym': sym, 'y_diff': y_diff, 'mouth_open': mouth_open,
                'eb_eye_high': eb_eye_high, 'eb_eye_low': eb_eye_low, 
                'sharpness_low': sharpness_low, 'sharpness_high': sharpness_high,
                'face_size_low': face_size_low, 'face_size_high': face_size_high,
                'retouching': retouching, 'mask': mask, 'glasses': glasses
            }
            
            total_weighted_exp = 0.0
            total_power = 0.0
            
            for param_key, power in current_vals.items():
                try:
                    power_val = float(power)
                except ValueError:
                    power_val = 0.0
                
                if power_val > 0:
                    mapped_key = mapping.get(param_key) or param_key
                    if not mapped_key.endswith('_percentile') and not mapped_key.endswith('_low') and not mapped_key.endswith('_high'):
                         mapped_key += '_percentile'
                         
                    if mapped_key in LR_INDIVIDUAL_EXPONENTS:
                        p_exp = float(LR_INDIVIDUAL_EXPONENTS[mapped_key])
                        total_weighted_exp += p_exp * power_val
                        total_power += power_val
            
            if total_power > 0:
                exponent = total_weighted_exp / total_power
                logger.info(f"Using Weighted Average Exponent: {exponent:.4f} (Total Power: {total_power:.1f})")
            else:
                exponent = LR_SCALING_EXP
                logger.info(f"Using Default Exponent: {exponent:.4f} (No active filters)")
            
            # y = 1.0 (Linear Scaling) -> Updated
            x = relative_ratio
            
            # adjusted_lr = base_lr * (ratio ** exp)
            scale_factor = relative_ratio ** exponent
            adjusted_lr = CALIBRATED_BASE_LR * scale_factor
            
            logger.info(
                f"LR Scaling (Dynamic): Base={CALIBRATED_BASE_LR:.8f}, Ratio={safe_ratio:.4f} (BaseRatio={BASE_RATIO:.4f}), "
                f"RelRatio={relative_ratio:.4f} -> Exp={exponent:.4f} -> LR={adjusted_lr:.8f}"
            )
        elif CALIBRATED_BASE_LR is not None:
            adjusted_lr = CALIBRATED_BASE_LR
            logger.info(f"LR (Calibrated): {adjusted_lr:.8f}")
        else:
            adjusted_lr = _get_head_lr_from_best(best_params, default=0.0001)
            logger.info(f"LR (Fallback): {adjusted_lr:.8f}")
        
        cmd_train.extend(["--learning_rate", str(adjusted_lr)])
        
        # 評価用なのでFine-tuningはOff、Epochs=20
        cmd_train.extend(["--epochs", "20"])
        cmd_train.extend(["--fine_tune", "False"])
        cmd_train.extend(["--auto_lr_target_epoch", "0"])  # 内部自動スケーリングを無効化
        # Conditional Extension は train_multitask_trial 側で必要時のみ発動させる

        # train_sequential.py と同じ LR 再調整条件（モジュール定数で共用）
        training_epochs = 20
        logger.info(f"Running training with {model_name} (epochs={training_epochs}, fine_tune=False)...")

        current_training_lr = adjusted_lr
        best_trial_score = 0.0
        best_trial_output = None

        for adj_iter in range(LR_MAX_ADJUSTMENTS + 1):
            if adj_iter > 0:
                logger.info(f"  [LR Adjust #{adj_iter}] LR={current_training_lr:.8f}...")

            # LR更新: cmd_trainの--learning_rateを差し替え
            retry_cmd = []
            skip_next = False
            for ci, arg in enumerate(cmd_train):
                if skip_next:
                    skip_next = False
                    continue
                if arg == "--learning_rate":
                    retry_cmd.extend(["--learning_rate", str(current_training_lr)])
                    skip_next = True
                else:
                    retry_cmd.append(arg)

            # Popenでリアルタイム出力 + スコア抽出
            process = subprocess.Popen(
                retry_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace'
            )

            train_output = []
            for line in process.stdout:
                line = line.rstrip()
                train_output.append(line)
                if any(kw in line for kw in ['Epoch ', 'FINAL_VAL_ACCURACY', 'TASK_', 'MinClassAcc', 'Avg=', 'BEST_EPOCH']):
                    logger.info(f"  [Train] {line}")

            process.wait()

            if process.returncode != 0:
                logger.error(f"Training failed (returncode={process.returncode})")
                return (0.0, 0, 0)

            full_output = "\n".join(train_output)

            # BEST_EPOCH抽出
            match_epoch = re.search(r"BEST_EPOCH:\s*(\d+)", full_output)
            best_epoch = int(match_epoch.group(1)) if match_epoch else training_epochs

            # スコア抽出
            match_score = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
            trial_score = float(match_score.group(1)) if match_score else 0.0

            # 各エポックのMinClassAccスコアを抽出してlast epoch accuを取得
            epoch_scores = re.findall(r"MinClassAcc=([\d.]+)", full_output)
            last_epoch_accu = float(epoch_scores[-1]) if epoch_scores else 0.0

            # ベストスコア更新
            if trial_score > best_trial_score:
                best_trial_score = trial_score
                best_trial_output = full_output

            logger.info(f"  BestEpoch={best_epoch}/{training_epochs}, Score={trial_score:.4f}, LastEpochAccu={last_epoch_accu:.4f}")
            should_exit, log_msg, need_adjust, effective_epoch = lr_adjustment_decision(
                best_epoch, last_epoch_accu, trial_score, training_epochs
            )
            if should_exit and log_msg:
                logger.info(log_msg)
                break
            if need_adjust and adj_iter < LR_MAX_ADJUSTMENTS and effective_epoch is not None:
                ratio = compute_lr_adjustment_ratio(effective_epoch, target_epoch=LR_TARGET_EPOCH, total_epochs=training_epochs)
                current_training_lr *= ratio
                logger.info(f"  [LR Adjust] effective_epoch={effective_epoch} -> ratio={ratio:.4f} -> LR={current_training_lr:.8f}")
            else:
                break
        
        # ベスト結果を使用
        if best_trial_output is not None:
            full_output = best_trial_output
        raw_score = best_trial_score
        
        # Extract Score
        match = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", full_output)
        if match:
            raw_score = max(raw_score, float(match.group(1)))
            
            # 各タスクのスコアを抽出してログに出力
            for char_code in range(ord('A'), ord('Z') + 1):
                task_label = chr(char_code)
                match_task = re.search(rf"TASK_{task_label}_ACCURACY:\s*([\d.]+)", full_output)
                if match_task:
                    logger.info(f"  Task {task_label}: {float(match_task.group(1)):.4f}")

            # 詳細なクラス別精度をログに転記
            details_match = re.search(r"--- Task [A-Z] Details.*", full_output, re.DOTALL)
            if details_match:
                logger.info("\n[Detailed Class Accuracy]")
                logger.info(details_match.group(0).strip())

            logger.info(f"Result: RawScore={raw_score:.4f} (Average), Total={total_images}, Filtered={filtered_count}")
            
            # キャッシュに保存 (4-tuple: val_min_cnt を併せて記録してガード再判定可能にする)
            cache = load_cache()
            if val_min_cnt is not None:
                cache[cache_key] = (raw_score, total_images, filtered_count, int(val_min_cnt))
            else:
                # val_dir が無い等で未算出: 3-tuple で保存しヒット時はヒューリスティックに委ねる
                cache[cache_key] = (raw_score, total_images, filtered_count)
            save_cache(cache)
            return (raw_score, total_images, filtered_count)
        else:
            logger.error("Score not found in training output.")
            logger.error(f"STDOUT:\n{full_output}")
            return (0.0, 0, 0)

    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return (0.0, 0, 0)

def optimize_single_param(target_name, current_params, model_name, baseline_score, baseline_filtered, points=[0, 5, 25, 50, 75]):
    """
    1つのパラメータを最適化する。
    
    Returns:
        tuple: (best_val, best_score, improvement, filtered_diff, efficiency)
        - best_val: 最適なパラメータ値
        - best_score: 最高スコア
        - improvement: ベースラインからの精度向上
        - filtered_diff: ベースラインからのフィルタリング枚数増加
        - efficiency: 効率 = improvement / (filtered_diff + 1)
    """
    logger.info(f"\n>>> Optimizing {target_name} [Model: {model_name}] <<<")
    logger.info(f"Baseline: Score={baseline_score:.4f}, Filtered={baseline_filtered}")
    
    def evaluate_wrapper(val):
        # Independent Optimization: Always start from all-zero params
        test_params = {
            'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
            'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0, 'sharpness_high': 0,
            'face_size_low': 0, 'face_size_high': 0, 'retouching': 0, 'mask': 0, 'glasses': 0
        }
        test_params[target_name] = val
        
        # run_trialは(raw_score, total_images, filtered_count)を返す
        result = run_trial(
            test_params['pitch'], test_params['sym'], test_params['y_diff'], test_params['mouth_open'],
            test_params['eb_eye_high'], test_params['eb_eye_low'],
            test_params['sharpness_low'], test_params['sharpness_high'],
            test_params['face_size_low'], test_params['face_size_high'],
            test_params['retouching'], test_params['mask'], test_params['glasses'],
            model_name=model_name,
            active_param_name=target_name
        )
        return result  # (raw_score, total_images, filtered_count)

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
    # --- 二分探索 Refinement (1%単位まで探索) ---
    # Initial searchで改善が見られた場合のみ実行
    if best_score > baseline_score:
        logger.info("  [Refinement] Starting binary search refinement to 1% resolution...")
        
        # 既存のスコア辞書 (scores) を使用して探索を継続
        # 上位2点間を埋めていく方針
        
        def calc_eff(score_info):
            raw_s, _, f_count = score_info
            imp = raw_s - baseline_score
            f_diff = f_count - baseline_filtered
            return imp / (f_diff + 1) if f_diff >= 0 else imp

        refinement_iter = 0
        MAX_REFINEMENT_ITER = 20 # 0-50の範囲なら十分収束する
        
        while refinement_iter < MAX_REFINEMENT_ITER:
            # スコア順・効率順にソート (降順)
            sorted_by_score = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            sorted_by_eff = sorted(scores.items(), key=lambda x: calc_eff(x[1]), reverse=True)
            
            if len(sorted_by_score) < 2:
                break
                
            s_top1_val, s_top2_val = sorted_by_score[0][0], sorted_by_score[1][0]
            e_top1_val, e_top2_val = sorted_by_eff[0][0], (sorted_by_eff[1][0] if len(sorted_by_eff) > 1 else None)
            
            candidates = []
            
            # Score Top2 中間点
            if abs(s_top1_val - s_top2_val) > 1:
                mid_s = (s_top1_val + s_top2_val) // 2
                if mid_s not in scores:
                    candidates.append(('Score', mid_s))
                    
            # Efficiency Top2 中間点
            if e_top2_val is not None and abs(e_top1_val - e_top2_val) > 1:
                mid_e = (e_top1_val + e_top2_val) // 2
                if mid_e not in scores and mid_e not in [c[1] for c in candidates]:
                    candidates.append(('Efficiency', mid_e))
                    
            if not candidates:
                logger.info(f"  [Refinement] Converged. No valid midpoints left to explore.")
                break
                
            improved_any = False
            for cand_type, mid_val in candidates:
                logger.info(f"  [Refinement #{refinement_iter+1}] Testing mid={mid_val} (based on {cand_type} Top 2)...")
                raw_score, total_images, filtered_count = evaluate_wrapper(mid_val)
                logger.info(f"  {target_name}={mid_val} -> Score: {raw_score:.4f}, Filtered: {filtered_count}")
                scores[mid_val] = (raw_score, total_images, filtered_count)
                
                if raw_score > best_score:
                    best_score = raw_score
                    best_val = mid_val
                    best_filtered = filtered_count
                    logger.info(f"  [REFINED BEST] {target_name}={mid_val} (Score: {raw_score:.4f})")
                    
                # 確認: この中間点が新たなTop2 (Score or Efficiency) に入ったか？
                new_sorted_s = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
                new_sorted_e = sorted(scores.items(), key=lambda x: calc_eff(x[1]), reverse=True)
                
                s_top2_vals = [new_sorted_s[0][0], new_sorted_s[1][0]] if len(new_sorted_s) > 1 else [new_sorted_s[0][0]]
                e_top2_vals = [new_sorted_e[0][0], new_sorted_e[1][0]] if len(new_sorted_e) > 1 else [new_sorted_e[0][0]]
                
                if mid_val in s_top2_vals or mid_val in e_top2_vals:
                    logger.info(f"  [Refinement] Midpoint {mid_val} stays in Top 2 (Score or Eff). Will continue.")
                    improved_any = True
                else:
                    logger.info(f"  [Refinement] Midpoint {mid_val} dropped out of Top 2 for both Score and Efficiency.")
            
            if not improved_any:
                logger.info(f"  [Refinement] None of the explored midpoints reached Top 2. Stopping binary search.")
                break
                
            refinement_iter += 1
    else:
        logger.info("  [Refinement] Skipped (No improvement over baseline in initial search).")
    
    # --- Collect Candidates (Best Score & Best Efficiency) ---
    # すべての探索点（Refinement含む）から候補を選定
    best_score_candidate = None
    best_eff_candidate = None
    max_score_val = -1.0
    max_eff_val = -float('inf')
    
    for p, res in scores.items():
        r_score, r_total, r_filtered = res
        
        # Improvement / Efficiency vs Baseline
        imp = r_score - baseline_score
        f_diff = r_filtered - baseline_filtered
        eff = imp / (f_diff + 1) if f_diff >= 0 else imp # Safety
        
        # Valid only if improvement is positive (or at least non-negative?)
        # User constraint: "score上昇" -> improvement > 0
        if imp > 0:
            # Update Best Score Candidate
            if r_score > max_score_val:
                max_score_val = r_score
                best_score_candidate = {
                    'val': p, 'score': r_score, 'improvement': imp, 
                    'filtered': r_filtered, 'efficiency': eff
                }
            
            # Update Best Efficiency Candidate
            if eff > max_eff_val:
                max_eff_val = eff
                best_eff_candidate = {
                    'val': p, 'score': r_score, 'improvement': imp, 
                    'filtered': r_filtered, 'efficiency': eff
                }

    logger.info(f"Finished optimizing {target_name}.")
    
    candidates = []
    if best_score_candidate:
        candidates.append(best_score_candidate)
        logger.info(f"  Best Score Param: {best_score_candidate['val']} (Score={best_score_candidate['score']:.4f}, Eff={best_score_candidate['efficiency']:.6f})")
        
    if best_eff_candidate:
        # Avoid duplicate object if same point is both best score and best eff
        if not best_score_candidate or best_eff_candidate['val'] != best_score_candidate['val']:
            candidates.append(best_eff_candidate)
            logger.info(f"  Best Eff Param:   {best_eff_candidate['val']} (Score={best_eff_candidate['score']:.4f}, Eff={best_eff_candidate['efficiency']:.6f})")

    if not candidates:
        logger.info("  No valid candidates found (no improvement).")

    return candidates, scores

def main():
    global CALIBRATED_BASE_LR, LR_SCALING_EXP, BASE_RATIO, LR_INDIVIDUAL_EXPONENTS
    logger.info("Starting Sequential Optimization (Efficiency-Based)")

    # LRスケーリング設定を読み込む（exponent は単一値）
    LR_SCALING_CONFIG_FILE = "outputs/lr_scaling_config.json"
    if os.path.exists(LR_SCALING_CONFIG_FILE):
        try:
            with open(LR_SCALING_CONFIG_FILE, 'r', encoding='utf-8') as f:
                lr_config = json.load(f)
            BASE_RATIO = 1.0  # 毎回動的に計算するため固定値は使わない
            if 'exponent' in lr_config:
                LR_SCALING_EXP = float(lr_config['exponent'])
            if 'individual_exponents' in lr_config:
                raw = lr_config['individual_exponents']
                LR_INDIVIDUAL_EXPONENTS = {k: float(v) for k, v in raw.items()}
                logger.info(f"Loaded {len(LR_INDIVIDUAL_EXPONENTS)} individual exponents.")
            logger.info(f"Loaded LR scaling config: exponent={LR_SCALING_EXP:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load {LR_SCALING_CONFIG_FILE}: {e}. Using defaults.")
    else:
        logger.info(f"LR scaling config {LR_SCALING_CONFIG_FILE} not found. Using defaults.")
    
    # Initial Params
    current_params = {
        'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
        'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0, 'sharpness_high': 0,
        'face_size_low': 0, 'face_size_high': 0, 'retouching': 0, 'mask': 0, 'glasses': 0
    }
    
    # 効率情報を記録する辞書
    param_efficiency = {}
    
    # --- LR Calibration: 毎回キャリブレーション実行（キャッシュなし） ---
    logger.info("\n>>> LR Calibration: Finding optimal base learning rate <<<")
    
    # まずフィルタなしで前処理
    cmd_pre_cal = [
        PYTHON_PREPROCESS, "preprocess_multitask.py",
        "--out_dir", "preprocessed_multitask",
        "--pitch_percentile", "0", "--symmetry_percentile", "0",
        "--y_diff_percentile", "0", "--mouth_open_percentile", "0",
        "--eyebrow_eye_percentile_high", "0", "--eyebrow_eye_percentile_low", "0",
        "--sharpness_percentile_low", "0", "--sharpness_percentile_high", "0",
        "--face_size_percentile_low", "0", "--face_size_percentile_high", "0",
        "--retouching_percentile", "0", "--mask_percentile", "0",
        "--glasses_percentile", "0"
    ]
    logger.info("Preprocessing for LR calibration (no filters)...")
    subprocess.run(cmd_pre_cal, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    # LRキャリブレーション実行
    best_params = load_best_train_params()
    initial_lr = _get_head_lr_from_best(best_params, default=0.0005)
    CALIBRATED_BASE_LR, cal_score = calibrate_base_lr(
        'EfficientNetV2B0',
        initial_lr,
        cal_epochs=20,
        target_best_epoch=13,
    )
    logger.info(f"Using Calibrated Base LR: {CALIBRATED_BASE_LR:.8f}, Score: {cal_score:.4f}")
    # base_lr が確定した時点で、次回の初期値として head 側LRのみを更新する。
    # body (learning_rate_nohead) / FT (learning_rate_ft) は別条件で乖離するため、ここでは触らない。
    try:
        best_params["learning_rate_head"] = float(CALIBRATED_BASE_LR)
        save_best_train_params(best_params)
        logger.info(f"Updated best_train_params learning_rate_head -> {CALIBRATED_BASE_LR:.8f}")
    except Exception as e:
        logger.warning(f"Failed to persist learning_rate_head to {BEST_TRAIN_PARAMS_FILE}: {e}")
    
    # --- Step 0: Model Architecture Selection & Baseline ---
    # キャリブレーション最終結果をB0のベースラインとして流用
    logger.info("\n>>> Step 0: Model Architecture Selection & Baseline <<<")
    best_model = 'EfficientNetV2B0'
    best_model_score = cal_score
    baseline_filtered = 0  # キャリブレーションはフィルタなし
    logger.info(f"B0 Baseline from calibration: Score={cal_score:.4f}")
    
    # 他のモデル候補をテスト
    other_models = ['EfficientNetV2S']
    for m in other_models:
        logger.info(f"Testing Model: {m}")
        raw_score, total_images, filtered_count = run_trial(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, model_name=m)
        if raw_score > best_model_score:
            best_model_score = raw_score
            best_model = m
            baseline_filtered = filtered_count
            
    logger.info(f"Best Model Selected: {best_model}")
    logger.info(f"Baseline: Score={best_model_score:.4f}, Filtered={baseline_filtered}")
    # train_sequential が起動時に参照する best_train_params.json へバックボーンを保存
    try:
        _btp = load_best_train_params()
        _btp["model_name"] = best_model
        save_best_train_params(_btp)
        logger.info(
            f"Updated {BEST_TRAIN_PARAMS_FILE} model_name -> {best_model} (for train_sequential)"
        )
    except Exception as e:
        logger.warning(f"Failed to persist model_name to {BEST_TRAIN_PARAMS_FILE}: {e}")

    baseline_score = best_model_score
    
    # --- Parameter Optimization (recording efficiency) ---
    param_names = [
        'pitch', 'sym', 'y_diff', 'mouth_open',
        'eb_eye_high', 'eb_eye_low', 'sharpness_low', 'sharpness_high',
        'face_size_low', 'face_size_high', 'retouching', 'mask', 'glasses'
    ]
    
    # Phase 1 loop
    # Phase 1 loop
    global_best_score = baseline_score
    global_best_params = {k: 0 for k in current_params}
    global_best_desc = "Baseline (No Filter)"
    
    # 全候補リスト (Best Score, Best Eff mixed)
    all_candidates = []

    all_param_scores = {}  # パラメータごとの全スコア辞書（タイブレーカー用）

    for param_name in param_names:
        # Returns (candidates, scores_dict)
        candidates, param_scores = optimize_single_param(
            param_name, current_params, best_model, baseline_score, baseline_filtered
        )
        all_param_scores[param_name] = param_scores
        
        for cand in candidates:
            cand['param_name'] = param_name
            all_candidates.append(cand)
            
            # Check for Global Single Best
            if cand['score'] > global_best_score:
                global_best_score = cand['score']
                global_best_params = {k: 0 for k in current_params}
                global_best_params[param_name] = cand['val']
                global_best_desc = f"Single Best ({param_name}={cand['val']})"
                logger.info(f"  [Global Best Update] New best found in single trial: {global_best_desc}, Score: {cand['score']:.4f}")

    # --- Phase 1 Tiebreaker: 同率パラメータの再評価 ---
    logger.info("\n>>> Phase 1 Tiebreaker: Checking for tied parameters <<<")
    tied_found = False
    
    for param_name, param_scores in all_param_scores.items():
        if not param_scores:
            continue
        # このパラメータのベストスコアを取得
        best_param_score = max(s[0] for s in param_scores.values())
        if best_param_score <= baseline_score:
            continue
        # 同率の値を検出
        tied_values = [v for v, s in param_scores.items() if abs(s[0] - best_param_score) < 1e-6]
        if len(tied_values) <= 1:
            continue
        
        # 同率あり → キャッシュ削除して再評価
        tied_found = True
        logger.info(f"  [Tie] {param_name}: values {tied_values} tied at {best_param_score:.4f}. Re-evaluating...")
        
        # 該当キャッシュエントリを削除
        cache = load_cache()
        file_count = count_files(DATA_SOURCE_DIR)
        keys_to_delete = []
        for tv in tied_values:
            # run_trialのキャッシュキーパターンに合わせて検索
            param_pattern = f"{param_name.replace('_', '')}" if '_' not in param_name else param_name
            for ck in list(cache.keys()):
                # パラメータ値が含まれるキーを探す
                # 各パラメータのキャッシュキーフォーマットに合致するか簡易チェック
                param_key_map = {
                    'pitch': f'pitch={tv}', 'sym': f'sym={tv}', 'y_diff': f'ydiff={tv}',
                    'mouth_open': f'mouth={tv}', 'eb_eye_high': f'ebh={tv}', 'eb_eye_low': f'ebl={tv}',
                    'sharpness_low': f'sharplow={tv}', 'sharpness_high': f'sharphigh={tv}',
                    'face_size_low': f'fsl={tv}', 'face_size_high': f'fsh={tv}',
                    'retouching': f'retouch={tv}', 'mask': f'mask={tv}', 'glasses': f'glasses={tv}'
                }
                if param_name in param_key_map and param_key_map[param_name] in ck:
                    keys_to_delete.append(ck)
        
        for dk in set(keys_to_delete):
            if dk in cache:
                del cache[dk]
                logger.info(f"    Cache cleared: {dk[:80]}...")
        save_cache(cache)
        
        # 再評価
        best_retry_score = -1.0
        best_retry_val = tied_values[0]
        for tv in tied_values:
            eval_params = {k: 0 for k in current_params}
            eval_params[param_name] = tv
            res = run_trial(
                eval_params['pitch'], eval_params['sym'], eval_params['y_diff'], eval_params['mouth_open'],
                eval_params['eb_eye_high'], eval_params['eb_eye_low'],
                eval_params['sharpness_low'], eval_params['sharpness_high'],
                eval_params['face_size_low'], eval_params['face_size_high'],
                eval_params['retouching'], eval_params['mask'], eval_params['glasses'],
                grayscale=False, model_name=best_model, active_param_name=param_name
            )
            retry_score = res[0]
            logger.info(f"    {param_name}={tv} -> Score: {retry_score:.4f}")
            if retry_score > best_retry_score:
                best_retry_score = retry_score
                best_retry_val = tv
        
        logger.info(f"  [Tie Resolved] {param_name}: winner={best_retry_val} (Score={best_retry_score:.4f})")
        
        # all_candidates内の該当パラメータの候補を更新
        for cand in all_candidates:
            if cand['param_name'] == param_name and cand['val'] in tied_values:
                if cand['val'] == best_retry_val:
                    cand['score'] = best_retry_score
                    cand['improvement'] = best_retry_score - baseline_score
                    # efficiency再計算
                    f_diff = cand['filtered'] - baseline_filtered
                    cand['efficiency'] = cand['improvement'] / (f_diff + 1) if f_diff >= 0 else cand['improvement']
        
        # Global Best更新チェック
        if best_retry_score > global_best_score:
            global_best_score = best_retry_score
            global_best_params = {k: 0 for k in current_params}
            global_best_params[param_name] = best_retry_val
            global_best_desc = f"Single Best ({param_name}={best_retry_val})"
            logger.info(f"  [Global Best Update] {global_best_desc}, Score: {best_retry_score:.4f}")
    
    if not tied_found:
        logger.info("  No tied parameters found.")

    # --- 精度上昇効率一覧 (Phase 1 結果サマリー) ---
    logger.info("\n" + "=" * 70)
    logger.info("精度上昇効率一覧 (Phase 1 Summary)")
    logger.info("=" * 70)
    logger.info(f"Baseline Score: {baseline_score:.4f}, Baseline Filtered: {baseline_filtered}")
    logger.info("-" * 70)
    logger.info(f"{'Param':<18} {'Type':<10} {'Val':>5} {'Score':>8} {'Improv':>8} {'Filtered':>8} {'Efficiency':>12}")
    logger.info("-" * 70)
    
    # パラメータごとに Best Score / Best Efficiency をグループ化して表示
    for param_name in param_names:
        param_cands = [c for c in all_candidates if c['param_name'] == param_name]
        if not param_cands:
            logger.info(f"{param_name:<18} {'---':<10} {'':>5} {'N/A':>8} {'':>8} {'':>8} {'':>12}")
            continue
        
        # Best Score 候補
        best_score_cand = max(param_cands, key=lambda x: x['score'])
        logger.info(
            f"{param_name:<18} {'BestScore':<10} {best_score_cand['val']:>5} "
            f"{best_score_cand['score']:>8.4f} {best_score_cand['improvement']:>+8.4f} "
            f"{best_score_cand['filtered']:>8} {best_score_cand['efficiency']:>12.6f}"
        )
        
        # Best Efficiency 候補 (Best Scoreと異なる場合のみ表示)
        best_eff_cand = max(param_cands, key=lambda x: x['efficiency'])
        if best_eff_cand['val'] != best_score_cand['val']:
            logger.info(
                f"{'':>18} {'BestEff':<10} {best_eff_cand['val']:>5} "
                f"{best_eff_cand['score']:>8.4f} {best_eff_cand['improvement']:>+8.4f} "
                f"{best_eff_cand['filtered']:>8} {best_eff_cand['efficiency']:>12.6f}"
            )
    
    logger.info("-" * 70)
    logger.info(f"Total candidates: {len(all_candidates)}")
    if all_candidates:
        overall_best_score_cand = max(all_candidates, key=lambda x: x['score'])
        overall_best_eff_cand = max(all_candidates, key=lambda x: x['efficiency'])
        logger.info(
            f"Overall Best Score:      {overall_best_score_cand['param_name']}={overall_best_score_cand['val']} "
            f"(Score={overall_best_score_cand['score']:.4f}, Eff={overall_best_score_cand['efficiency']:.6f})"
        )
        logger.info(
            f"Overall Best Efficiency: {overall_best_eff_cand['param_name']}={overall_best_eff_cand['val']} "
            f"(Score={overall_best_eff_cand['score']:.4f}, Eff={overall_best_eff_cand['efficiency']:.6f})"
        )
    logger.info("=" * 70)

    # Phase 2: Efficiency-Based Greedy Integration (Grayscale無しで実行)
    logger.info("\n>>> Phase 2: Efficiency-Based Greedy Integration <<<")
    
    # 効率順にソート (降順)
    # 単体スコア（Score）ではなく、上昇効率（Efficiency）の高い順に統合を試す
    sorted_candidates = sorted(all_candidates, key=lambda x: x['efficiency'], reverse=True)
    
    current_greedy_params = {k: 0 for k in current_params}
    
    # ベースライン（Grayscale=False）のスコア計測
    base_res = run_trial(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        grayscale=False, model_name=best_model
    )
    current_best_score = base_res[0]
    logger.info(f"Base Score (No filters): {current_best_score:.4f}")
    
    greedy_history = []
    
    for cand in sorted_candidates:
        p_name = cand['param_name']
        p_val = cand['val']
        p_eff = cand['efficiency']
        p_score_single = cand['score']
        
        logger.info(f"Trying candidate: {p_name}={p_val} (Eff: {p_eff:.6f}, SingleScore: {p_score_single:.4f})...")
        
        # 既にセットされている値がある場合、それと比較して更新するか判断
        prev_val = current_greedy_params[p_name]
        
        if prev_val == p_val:
            logger.info(f"  -> Already set to {p_val}. Skipping.")
            continue
            
        # 一時的にパラメータ適用
        temp_params = current_greedy_params.copy()
        temp_params[p_name] = p_val
        
        res = run_trial(
            temp_params['pitch'], temp_params['sym'], temp_params['y_diff'], temp_params['mouth_open'],
            temp_params['eb_eye_high'], temp_params['eb_eye_low'],
            temp_params['sharpness_low'], temp_params['sharpness_high'],
            temp_params['face_size_low'], temp_params['face_size_high'],
            temp_params['retouching'], temp_params['mask'], temp_params['glasses'],
            grayscale=False, model_name=best_model
        )
        score, total, filtered = res
        
        # 採用基準: スコアが現状以上 (>=) なら採用
        # (効率が良い順に試しているので、同じスコアなら効率が良い方が先に採用されているはずだが、
        #  後から来た「効率は低いがスコアが高い候補」がさらにスコアを上げるなら更新する)
        if score >= current_best_score:
            diff = score - current_best_score
            logger.info(f"  -> Accepted (Score: {score:.4f} >= {current_best_score:.4f}, Diff: +{diff:.4f})")
            current_best_score = score
            current_greedy_params[p_name] = p_val
            greedy_history.append((p_name, p_val, score))
        else:
            logger.info(f"  -> Skipped (Score: {score:.4f} < {current_best_score:.4f})")
            
    greedy_score = current_best_score
    greedy_params = current_greedy_params.copy()

    # 元のパラメータでもスコアを確認（比較用、Grayscale無し）
    logger.info("\n--- Testing Original Parameters (no scaling, no grayscale) ---")
    original_result = run_trial(
        current_params['pitch'], current_params['sym'], current_params['y_diff'], current_params['mouth_open'],
        current_params['eb_eye_high'], current_params['eb_eye_low'],
        current_params['sharpness_low'], current_params['sharpness_high'],
        current_params['face_size_low'], current_params['face_size_high'],
        current_params['retouching'], current_params['mask'], current_params['glasses'],
        grayscale=False, model_name=best_model
    )
    original_score, original_total, original_filtered = original_result

    # --- Final Selection Phase (Grayscale適用前) ---
    logger.info("\n>>> Final Selection Phase (without Grayscale) <<<")
    
    candidates = []
    
    # Greedy Integration
    candidates.append({'name': "Greedy Integration", 'params': greedy_params, 'score': greedy_score})
    logger.info(f"Strategy: {'Greedy':<15} Score={greedy_score:.4f}")
    
    # Original (all Phase 1 params applied)
    candidates.append({
        'name': "Original (No Scale)",
        'params': {k: v for k, v in current_params.items()},
        'score': original_score,
    })
    logger.info(f"Strategy: {'Original':<15} Score={original_score:.4f}")
    
    # Single Best
    logger.info(f"\n--- Testing Single Best ({global_best_desc}) ---")
    sb_res = run_trial(
        global_best_params['pitch'], global_best_params['sym'], global_best_params['y_diff'], global_best_params['mouth_open'],
        global_best_params['eb_eye_high'], global_best_params['eb_eye_low'],
        global_best_params['sharpness_low'], global_best_params['sharpness_high'],
        global_best_params['face_size_low'], global_best_params['face_size_high'],
        global_best_params['retouching'], global_best_params['mask'], global_best_params['glasses'],
        grayscale=False, model_name=best_model
    )
    sb_score = sb_res[0]
    candidates.append({
        'name': f"Single Best ({global_best_desc})",
        'params': global_best_params.copy(),
        'score': sb_score,
    })
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
        grayscale=False, model_name=best_model
    )
    res_gray = run_trial(
        final_params['pitch'], final_params['sym'], final_params['y_diff'], final_params['mouth_open'],
        final_params['eb_eye_high'], final_params['eb_eye_low'],
        final_params['sharpness_low'], final_params['sharpness_high'],
        final_params['face_size_low'], final_params['face_size_high'],
        final_params['retouching'], final_params['mask'], final_params['glasses'],
        grayscale=True, model_name=best_model
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

    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Selected Strategy: {final_desc}")
    logger.info(f"Final Best Params: {final_params}")
    logger.info(f"Final Best Model: {best_model}")
    logger.info(f"Final Score: {final_score:.4f}")
    logger.info(f"Baseline Score: {baseline_score:.4f}")
    logger.info(f"Total Improvement: +{final_score - baseline_score:.4f}")
    logger.info("="*50)

    # 結果を自動記録 (Legacy logger removed, using JSON analysis)
    # --- Save Optimization Analysis Data ---
    analysis_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'baseline_score': baseline_score,
        'candidates': all_candidates,
        'greedy_order': [
            {'param': c['param_name'], 'val': c['val'], 'efficiency': c['efficiency'], 'single_score': c['score']} 
            for c in sorted_candidates
        ],
        'greedy_process': [
            {'param': p, 'val': v, 'score_after': s} for p, v, s in greedy_history
        ],
        'final_selection': {
            'strategy': final_desc,
            'score': final_score,
            'params': final_params,
            'model': best_model
        }
    }
    
    analysis_file = os.path.join("outputs", "optimization_analysis.json")
    try:
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=4)
        logger.info(f"Optimization analysis saved to: {analysis_file}")
    except Exception as e:
        logger.error(f"Failed to save analysis data: {e}")

    # Generate and print the final preprocessing command
    cmd_parts = [PYTHON_PREPROCESS, "preprocess_multitask.py"]
    cmd_parts.extend(["--out_dir", "preprocessed_multitask"])
    
    # Add all params
    for k, v in final_params.items():
        if k == 'grayscale':
            pass # handled separately
        else:
            # percentile params
            arg_map = {
                'pitch': '--pitch_percentile',
                'sym': '--symmetry_percentile',
                'y_diff': '--y_diff_percentile',
                'mouth_open': '--mouth_open_percentile',
                'eb_eye_high': '--eyebrow_eye_percentile_high',
                'eb_eye_low': '--eyebrow_eye_percentile_low',
                'sharpness_low': '--sharpness_percentile_low',
                'sharpness_high': '--sharpness_percentile_high',
                'face_size_low': '--face_size_percentile_low',
                'face_size_high': '--face_size_percentile_high',
                'retouching': '--retouching_percentile',
                'mask': '--mask_percentile',
                'glasses': '--glasses_percentile'
            }
            if k in arg_map:
                cmd_parts.extend([arg_map[k], str(v)])
    
    if final_params.get('grayscale', False):
        cmd_parts.append("--grayscale")
        
    final_cmd = " ".join(cmd_parts)
    logger.info("\nRun this command to apply the best filters:")
    logger.info(final_cmd)
    logger.info(f"\nRecommended Model for Training: {best_model}")
    logger.info("="*50)

    # Save command to batch file for easy execution
    bat_file = "run_optimized_preprocess.bat"
    with open(bat_file, 'w', encoding='utf-8') as f:
        f.write(f"@echo off\n")
        f.write(f"echo Running Optimized Preprocessing...\n")
        f.write(f"{final_cmd}\n")
        f.write(f"pause\n")
    logger.info(f"Command saved to {bat_file}")

if __name__ == "__main__":
    main()
    # 処理完了通知音
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
