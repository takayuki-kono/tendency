# LR再調整・キャリブレーション共通定数と判定ロジック（train_sequential / optimize_sequential で共用）

LR_TARGET_EPOCH = 13
LR_ACCEPTABLE_MIN = 11
LR_ACCEPTABLE_MAX = 15  # 許容範囲・ピーク後下降で終了する上限（両方に使用）
LR_MAX_ADJUSTMENTS = 6
# calibrate_base_lr（optimize / train_sequential）の試行回数上限。run_trial の LR 再調整回数とは独立。
LR_CALIBRATION_MAX_ITERATIONS = 10
# calibrate_base_lr の「新規」探索開始 LR（モデル・データ数・head/FT のいずれかが前回と異なるとき）
LR_CALIBRATION_INITIAL = 0.01
# best_train_params.json に保存する LR キャリブ文脈のキー（model / data_file_count / mode / base_lr）
LR_CALIB_CONTEXT_JSON_KEY = "lr_calib_context"
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


def lr_adjustment_decision(best_epoch, last_epoch_accu, trial_score, training_epochs):
    """
    run_trial 内のLR再調整ループで使う判定。
    戻り値: (should_exit: bool, log_message: str|None, need_adjust: bool, effective_epoch: int|None)
    """
    # 許容範囲内かつ last≠best なら調整完了
    if LR_ACCEPTABLE_MIN <= best_epoch <= LR_ACCEPTABLE_MAX and abs(last_epoch_accu - trial_score) >= LR_LAST_ACCU_EPS:
        return (True, f"  BestEpoch {best_epoch} in [{LR_ACCEPTABLE_MIN}-{LR_ACCEPTABLE_MAX}] and last_accu≠best. Done.", False, None)
    # 許容範囲内で last < best（ピーク後に下降）なら再調整しないで終了
    if LR_ACCEPTABLE_MIN <= best_epoch <= LR_ACCEPTABLE_MAX and last_epoch_accu < trial_score:
        return (True, f"  BestEpoch {best_epoch} in [{LR_ACCEPTABLE_MIN}-{LR_ACCEPTABLE_MAX}] and last_accu < best (peaked then declined). Done.", False, None)

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


def lr_calibration_should_stop(best_epoch, last_epoch_accu, score):
    """
    キャリブレーションの終了条件（run_trial の lr_adjustment_decision と同一条件に揃える）。
    戻り値: (should_stop: bool, log_message: str|None)
    """
    if LR_ACCEPTABLE_MIN <= best_epoch <= LR_ACCEPTABLE_MAX and abs(last_epoch_accu - score) >= LR_LAST_ACCU_EPS:
        return (True, f"BestEpoch {best_epoch} in [{LR_ACCEPTABLE_MIN}-{LR_ACCEPTABLE_MAX}] and last_accu≠best. Stopping calibration.")
    if LR_ACCEPTABLE_MIN <= best_epoch <= LR_ACCEPTABLE_MAX and last_epoch_accu < score:
        return (True, f"BestEpoch {best_epoch} in [{LR_ACCEPTABLE_MIN}-{LR_ACCEPTABLE_MAX}] and last_accu < best (peaked then declined). Stopping calibration.")
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
