# LR再調整・キャリブレーション共通定数と判定ロジック（train_sequential / optimize_sequential で共用）

LR_TARGET_EPOCH = 13
LR_ACCEPTABLE_MIN = 11
LR_ACCEPTABLE_MAX = 15  # 許容範囲・ピーク後下降で終了する上限（両方に使用）
LR_MAX_ADJUSTMENTS = 6
LR_LAST_ACCU_EPS = 0.01  # 最終epoch精度とベストスコアの差がこれ以上で「last≠best」とみなす
# 学習（optimizer / LR スケジューラ）に乗せる絶対域。極小 LR は .8f ログで 0 表示になり実質停止、
# 極大は設定ミス時の数値破綻を防ぐ。再調整「比」クランプとは独立。
# 目安: Adam + 224 系 CNN の head/FT でよく使う 1e-4〜1e-2 の帯より広く、探索を殺さない範囲に上限。
LR_TRAIN_ABSOLUTE_MIN = 1e-7
LR_TRAIN_ABSOLUTE_MAX = 0.1


def clip_learning_rate_for_training(lr):
    """
    train_multitask_trial から optimizer / scheduler / 延長学習に渡す直前に適用する。
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
