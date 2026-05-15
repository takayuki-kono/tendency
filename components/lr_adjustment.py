# LR再調整・キャリブレーション共通定数と判定ロジック（train_sequential / optimize_sequential で共用）

import math

LR_TARGET_EPOCH = 13
LR_ACCEPTABLE_MIN = 11
LR_ACCEPTABLE_MAX = 15  # early-stop 帯の上限（lr_calibration_should_stop / should_exit）
LR_MAX_ADJUSTMENTS = 6
# calibrate_base_lr（optimize / train_sequential）の試行回数上限。run_trial の LR 再調整回数とは独立。
LR_CALIBRATION_MAX_ITERATIONS = 10
# calibrate_base_lr の「新規」探索開始 LR（モデル・データ数・head/FT のいずれかが前回と異なるとき）
LR_CALIBRATION_INITIAL = 0.01
# calibrate_base_lr: 連続試行で LR の相対変化がこれ未満なら打ち止め（二分・ratio 更新後）
LR_CALIB_MIN_RELATIVE_CHANGE = 0.05
# best_train_params.json に保存する LR キャリブ文脈のキー（model / data_file_count / mode / base_lr）
LR_CALIB_CONTEXT_JSON_KEY = "lr_calib_context"
# JSON のみ／親スクリプト用で train_multitask_trial の argparse に存在しないキー（子プロセスへ渡さない）
TRAIN_MULTITASK_META_JSON_KEYS_ONLY = frozenset(
    (
        LR_CALIB_CONTEXT_JSON_KEY,
        "finish_mode",
        "score_step_3_9_head",
        "score_step_3_5_ft_calib",
        "score_head_only_phase_best",
        "ft_calib_carryover_selected",
        "score_step_3_5_ft_calib_carry",
        "lr_step_3_5_ft_calib_carry",
        "score_step_3_5_ft_calib_warmup",
        "lr_step_3_5_ft_calib_warmup",
    )
)
# train_sequential 互換で JSON に残り得る旧 LR フィールドも CLI に載せない
TRAIN_MULTITASK_LEGACY_LR_JSON_KEYS = frozenset(
    ("learning_rate_nohead", "learning_rate_head", "learning_rate_ft")
)
TRAIN_MULTITASK_CLI_EXCLUDE_KEYS = TRAIN_MULTITASK_META_JSON_KEYS_ONLY | TRAIN_MULTITASK_LEGACY_LR_JSON_KEYS
LR_LAST_ACCU_EPS = 0.01  # 最終epoch精度とベストスコアの差がこれ以上で「last≠best」とみなす
# 学習（optimizer / LR スケジューラ）に乗せる絶対域。極小 LR は .8f ログで 0 表示になり実質停止、
# 極大は設定ミス時の数値破綻を防ぐ。再調整「比」クランプとは独立。
# 目安: Adam + 224 系 CNN の head/FT でよく使う 1e-4〜1e-2 の帯より広く、探索を殺さない範囲に上限。
LR_TRAIN_ABSOLUTE_MIN = 1e-7
LR_TRAIN_ABSOLUTE_MAX = 0.1


def clip_learning_rate_for_training(lr):
    """
    train_multitask_trial から optimizer / scheduler に渡す直前に適用する。
    戻り値は常に [LR_TRAIN_ABSOLUTE_MIN, LR_TRAIN_ABSOLUTE_MAX]。
    """
    try:
        x = float(lr)
    except (TypeError, ValueError):
        return LR_TRAIN_ABSOLUTE_MIN
    if x != x:  # NaN
        return LR_TRAIN_ABSOLUTE_MIN
    if x > 1e100:  # +inf
        return LR_TRAIN_ABSOLUTE_MAX
    if x < -1e100:  # -inf
        return LR_TRAIN_ABSOLUTE_MIN
    if x < LR_TRAIN_ABSOLUTE_MIN:
        return LR_TRAIN_ABSOLUTE_MIN
    if x > LR_TRAIN_ABSOLUTE_MAX:
        return LR_TRAIN_ABSOLUTE_MAX
    return x


def compute_lr_adjustment_ratio(best_epoch, target_epoch=10, total_epochs=20):
    """`best_epoch / target_epoch` を返す。比の乗算クランプは行わない。`total_epochs` は互換用。実 LR は `clip_learning_rate_for_training` 適用。"""
    if target_epoch <= 0:
        return 1.0
    return best_epoch / target_epoch


def lr_bisect_update_bounds_and_next_raw(
    best_epoch: int,
    current_lr: float,
    cal_epochs: int,
    target_min: float,
    target_max: float,
    lr_low: float | None,
    lr_high: float | None,
) -> tuple[float | None, float | None, float, bool, float | None]:
    """
    `calibrate_base_lr` と `run_trial` の LR 再調整で共通。
    試行の `best_epoch` で lr_low / lr_high を更新し、clip 適用前の次試行 LR を返す。

    Returns:
        (lr_low, lr_high, new_lr_raw, used_geom_bisection, ratio_scale_or_none)
    """
    lo, hi = lr_low, lr_high
    if best_epoch < target_min:
        if hi is None or current_lr < hi:
            hi = current_lr
    elif best_epoch > target_max:
        if lo is None or current_lr > lo:
            lo = current_lr

    target_mid = (target_min + target_max) / 2.0
    if lo is not None and hi is not None:
        new_lr_raw = math.sqrt(lo * hi)
        return lo, hi, new_lr_raw, True, None

    scale = compute_lr_adjustment_ratio(
        best_epoch, target_epoch=int(target_mid), total_epochs=cal_epochs
    )
    new_lr_raw = current_lr * scale
    return lo, hi, new_lr_raw, False, scale


def lr_adjustment_decision(best_epoch, last_epoch_accu, trial_score, training_epochs):
    """
    互換用の判定（現行の `run_trial` は `lr_calibration_should_stop` + `lr_bisect_update_bounds_and_next_raw` を直接使用）。
    戻り値: (should_exit: bool, log_message: str|None, need_adjust: bool, effective_epoch: int|None)
    """
    # 許容帯内かつ last≠best なら再調整ループ終了（lr_calibration_should_stop と同一）
    if LR_ACCEPTABLE_MIN <= best_epoch <= LR_ACCEPTABLE_MAX and abs(last_epoch_accu - trial_score) >= LR_LAST_ACCU_EPS:
        return (True, f"  BestEpoch {best_epoch} in [{LR_ACCEPTABLE_MIN}-{LR_ACCEPTABLE_MAX}] and last_accu≠best. Done.", False, None)

    need_adjust = False
    effective_epoch = best_epoch
    if best_epoch <= 10:
        need_adjust = True
    elif best_epoch == training_epochs:
        need_adjust = True
        effective_epoch = training_epochs
    elif abs(last_epoch_accu - trial_score) < LR_LAST_ACCU_EPS:
        need_adjust = True
    return (False, None, need_adjust, effective_epoch if need_adjust else None)


def lr_calibration_should_stop(best_epoch, last_epoch_accu, score, *, acceptable_band=None):
    """
    calibrate_base_lr の試行ごとの早期終了（満足打ち切り）。
    run_trial の lr_adjustment_decision の should_exit と同一。
    `acceptable_band` が None のときは `LR_ACCEPTABLE_MIN`〜`LR_ACCEPTABLE_MAX`（既定 11〜15）。
    `target_best_epoch` が帯タプル `(lo, hi)` のキャリブでは呼び出し側が同じ帯を渡し、
    Step 4.7 の 10〜15 が 11〜15 固定で打ち切られないようにする。
    戻り値: (should_stop: bool, log_message: str|None)
    """
    if acceptable_band is None:
        lo, hi = LR_ACCEPTABLE_MIN, LR_ACCEPTABLE_MAX
    else:
        lo, hi = acceptable_band
    if lo <= best_epoch <= hi and abs(last_epoch_accu - score) >= LR_LAST_ACCU_EPS:
        return (True, f"BestEpoch {best_epoch} in [{lo}-{hi}] and last_accu≠best. Stopping calibration.")
    return (False, None)


def lr_calib_mode_from_fine_tune(fine_tune_val) -> str:
    """head-only → 'head'、fine_tune 有効 → 'ft'。"""
    s = str(fine_tune_val).strip().lower()
    return "ft" if s in ("true", "1", "yes") else "head"


def parse_lr_calib_context(blob) -> dict | None:
    """
    JSON の lr_calib_context オブジェクトを検証して正規化 dict にする。
    戻り値: {"model_name", "data_file_count", "mode", "base_lr"} または None。
    """
    if not blob or not isinstance(blob, dict):
        return None
    try:
        mn = blob.get("model_name")
        dc = int(blob["data_file_count"])
        mode = blob.get("mode")
        blr = float(blob["base_lr"])
        if mn is None or mode not in ("head", "ft"):
            return None
        return {
            "model_name": str(mn),
            "data_file_count": dc,
            "mode": str(mode),
            "base_lr": clip_learning_rate_for_training(blr),
        }
    except (KeyError, TypeError, ValueError):
        return None


def lr_calib_triple_match(ctx: dict, model_name: str, data_file_count: int, mode: str) -> bool:
    if not ctx:
        return False
    return (
        str(ctx.get("model_name")) == str(model_name)
        and int(ctx["data_file_count"]) == int(data_file_count)
        and str(ctx.get("mode")) == str(mode)
    )


def resolve_calib_initial_lr(
    model_name: str,
    data_file_count: int,
    mode: str,
    *,
    last_ctx: dict | None,
    persisted_ctx: dict | None,
    fresh_initial: float = LR_CALIBRATION_INITIAL,
) -> tuple[float, str]:
    """
    calibrate_base_lr の initial_lr を決める。
    - 同一実行内の直前キャリブと (model, data_file_count, mode) が一致 → その base_lr から再キャリブ
    - そうでなければディスク上の lr_calib_context と一致 → 保存 base_lr から再キャリブ
    - いずれでもなければ fresh_initial（通常 0.01）
    """
    mode = str(mode)
    if last_ctx is not None and lr_calib_triple_match(last_ctx, model_name, data_file_count, mode):
        lr = clip_learning_rate_for_training(float(last_ctx["base_lr"]))
        return lr, "同一ラン直前のキャリブと model/data_count/head|ft 一致 → 引き継ぎ base_lr でキャリブ"
    if persisted_ctx is not None and lr_calib_triple_match(
        persisted_ctx, model_name, data_file_count, mode
    ):
        lr = clip_learning_rate_for_training(float(persisted_ctx["base_lr"]))
        return lr, "保存 lr_calib_context と model/data_count/head|ft 一致 → 保存 base_lr でキャリブ"
    lr0 = clip_learning_rate_for_training(float(fresh_initial))
    return lr0, "model・データ数・head|ft のいずれかが不一致または初回 → initial_lr=新規探索（既定0.01）でキャリブ"


def make_lr_calib_context(model_name: str, data_file_count: int, mode: str, base_lr: float) -> dict:
    return {
        "model_name": str(model_name),
        "data_file_count": int(data_file_count),
        "mode": str(mode),
        "base_lr": float(clip_learning_rate_for_training(base_lr)),
    }
