"""
train_opt_cache.json から Step 3.9 以降に相当するエントリを削除する（train_sequential は変更しない）。

キーは MD5(params_json + _count=file_count) のため、元 params が保存されていないキーは特定不可。
本スクリプトは outputs/best_train_params.json から再現できる組み合わせを列挙し、
file_count を 1..5000 まで走査して一致キーを削除する。

用法: python scripts/prune_train_opt_cache_post_step39.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(REPO_ROOT, "outputs", "cache", "train_opt_cache.json")
BEST_JSON = os.path.join(REPO_ROOT, "outputs", "best_train_params.json")
DATA_SOURCE_DIR = os.path.join(REPO_ROOT, "preprocessed_multitask", "train")
N_FINAL_RUNS = 3
FINAL_EPOCHS = 20


def count_files(directory: str) -> int:
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        count += len(files)
    return count


def cache_key(params: dict, file_count: int) -> str:
    params_str = json.dumps(params, sort_keys=True)
    key_src = f"{params_str}_count={file_count}"
    return hashlib.md5(key_src.encode("utf-8")).hexdigest()


def strip_meta(d: dict) -> dict:
    out = dict(d)
    for k in (
        "finish_mode",
        "score_step_3_9_head",
        "score_step_3_5_ft_calib",
    ):
        out.pop(k, None)
    return out


def build_param_variants(raw: dict) -> list[dict]:
    """Step 3.9 以降の run_trial で起こりうる params の代表集合。"""
    variants: list[dict] = []

    def add(d: dict) -> None:
        variants.append(dict(d))

    for run_idx in range(N_FINAL_RUNS):
        seed = 42 + run_idx
        p_ft = dict(raw)
        p_ft["epochs"] = FINAL_EPOCHS
        p_ft["seed"] = seed
        p_ft.pop("export_model_path", None)
        p_ft["fine_tune"] = "True"
        add(p_ft)

        p_h = dict(raw)
        p_h["epochs"] = FINAL_EPOCHS
        p_h["seed"] = seed
        p_h["fine_tune"] = "False"
        lr_head = float(
            p_h.get("learning_rate")
            or p_h.get("learning_rate_head")
            or p_h.get("learning_rate_nohead")
            or 1e-3
        )
        p_h["learning_rate"] = lr_head
        p_h["learning_rate_ft"] = lr_head
        p_h["export_model_path"] = os.path.join(
            "outputs", "models", f"model_seed{seed}.keras"
        )
        add(p_h)

    for ul in (20, 40, 60, 999):
        for lr_source in ("learning_rate", "learning_rate_ft"):
            if lr_source not in raw:
                continue
            lr_val = float(raw[lr_source])
            for with_seed in (False, True):
                for seed_try in (42, 43, 44):
                    p4 = dict(raw)
                    p4["unfreeze_layers"] = ul
                    p4["fine_tune"] = "True"
                    p4["epochs"] = 20
                    p4["learning_rate"] = lr_val
                    p4["learning_rate_ft"] = lr_val
                    p4.pop("export_model_path", None)
                    if with_seed:
                        p4["seed"] = seed_try
                    else:
                        p4.pop("seed", None)
                    add(p4)
    return variants


def expand_optional_key_variants(params_list: list[dict]) -> list[dict]:
    """メタキーを落とした版も追加（キャッシュ生成時の dict 形の揺れに対応）。"""
    opt = (
        "learning_rate_nohead",
        "learning_rate_head",
        "learning_rate_ft",
        "save_best_head_model_path",
        "save_best_head_weights_path",
    )
    out: list[dict] = []
    for p in params_list:
        out.append(dict(p))
        for k in opt:
            if k in p:
                q = dict(p)
                q.pop(k, None)
                out.append(q)
        # Windows / POSIX のパス表記差
        if p.get("init_weights_path"):
            q = dict(p)
            q["init_weights_path"] = str(q["init_weights_path"]).replace("\\", "/")
            out.append(q)
    # 重複除去（同一 json 表現）
    seen: set[str] = set()
    uniq: list[dict] = []
    for p in out:
        s = json.dumps(p, sort_keys=True)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def main() -> int:
    os.chdir(REPO_ROOT)
    if not os.path.isfile(CACHE_FILE):
        print(f"No cache file: {CACHE_FILE}", file=sys.stderr)
        return 1

    fc_now = count_files(DATA_SOURCE_DIR)

    if not os.path.isfile(BEST_JSON):
        print(f"Error: {BEST_JSON} missing — cannot derive removal keys.", file=sys.stderr)
        return 1

    with open(BEST_JSON, encoding="utf-8") as f:
        bp = json.load(f)
    base = strip_meta(bp)

    candidates = expand_optional_key_variants(build_param_variants(base))

    bak = CACHE_FILE + ".bak"
    shutil.copy2(CACHE_FILE, bak)
    print(f"Backup: {bak}")

    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)

    before = len(cache)
    keys_in_cache = set(cache.keys())
    removed = 0

    for fc in range(1, 5001):
        for p in candidates:
            h = cache_key(p, fc)
            if h in keys_in_cache:
                del cache[h]
                keys_in_cache.discard(h)
                removed += 1

    print(f"current_data_file_count={fc_now}")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4)

    print(
        f"unique_param_variants={len(candidates)}, fc_range=1..5000, "
        f"removed={removed}, entries_before={before}, after={len(cache)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
