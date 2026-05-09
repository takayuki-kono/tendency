"""
学習時の前処理で確定したフィルタ実数閾値をデプロイ用にモデル旁へ複製する。

`preprocess_multitask.py` が `preprocessed_multitask/filter_threshold_manifest.json`
を出力する前提で、`train_multitask_trial.py` / `train_sequential.py` が
`.keras` 保存ディレクトリへ同ファイル名でコピーする。
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_RELPATH = os.path.join("preprocessed_multitask", "filter_threshold_manifest.json")
MANIFEST_FILENAME = "filter_threshold_manifest.json"


def copy_filter_manifest_beside_model(model_path: str, manifest_src: Optional[str] = None) -> None:
    """
    model_path と同じディレクトリに filter_threshold_manifest.json を置く。
    manifest_src が無い・存在しない場合は WARNING のみ（学習自体は継続）。
    """
    src = manifest_src or DEFAULT_MANIFEST_RELPATH
    if not os.path.isfile(src):
        logger.warning("Filter manifest not found (%s); skip copy beside %s", src, model_path)
        return
    dst_dir = os.path.dirname(os.path.abspath(model_path))
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, MANIFEST_FILENAME)
    try:
        shutil.copy2(src, dst)
        logger.info("Copied filter threshold manifest -> %s", dst)
    except OSError as exc:
        logger.warning("Could not copy filter manifest to %s: %s", dst, exc)
