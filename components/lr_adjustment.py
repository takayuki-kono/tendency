# LR再調整・キャリブレーション共通定数と判定ロジック（train_sequential / optimize_sequential で共用）

LR_TARGET_EPOCH = 13
LR_ACCEPTABLE_MIN = 11
LR_ACCEPTABLE_MAX = 15  # 許容範囲・ピーク後下降で終了する上限（両方に使用）
LR_MAX_ADJUSTMENTS = 6
LR_LAST_ACCU_EPS = 0.01  # 最終epoch精度とベストスコアの差がこれ以上で「last≠best」とみなす
# 時間軸比 raw = best_epoch / target_epoch のクランプ（run_trial 再調整 & calibrate 片側比）
LR_ADJUST_RATIO_MIN = 0.5
LR_ADJUST_RATIO_MAX = 2.0


def compute_lr_adjustment_ratio(best_epoch, target_epoch=10, total_epochs=20, min_lr_ratio=0.05):
    """
    時間軸でLR調整比率を計算する。raw = best_epoch / target_epoch を
    [LR_ADJUST_RATIO_MIN, LR_ADJUST_RATIO_MAX] にクランプし、new_lr = current_lr * ratio。
    best_epoch が target より小さいほど raw は小さく（LR 下げ）、大きいほど raw は大きく（LR 上げ）。
    """
    if target_epoch <= 0:
        return 1.0
    raw = best_epoch / target_epoch
    return max(LR_ADJUST_RATIO_MIN, min(raw, LR_ADJUST_RATIO_MAX))


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
