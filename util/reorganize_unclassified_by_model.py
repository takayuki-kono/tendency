import os
import sys
import json
import shutil
import argparse
from glob import glob
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf
import cv2

# 親ディレクトリをパスに追加して components からインポート可能にする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 4タスクのラベル定義を最新実装に合わせる
    from components.train_for_filter_search import ALL_TASK_LABELS
except Exception:
    # 最低限のフォールバック（想定: a/b, d/e, f/g, h/i）
    ALL_TASK_LABELS = [['a', 'b'], ['d', 'e'], ['f', 'g'], ['h', 'i']]


IMG_SIZE = 224
BATCH_SIZE = 32
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def _imread_jp(path: str):
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def preprocess_image(path: str):
    img = _imread_jp(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def find_latest_keras_model(models_dir: str = os.path.join("outputs", "models")) -> str | None:
    if not os.path.exists(models_dir):
        return None
    candidates = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.lower().endswith(".keras")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_model_for_inference(model_path: str):
    # 予測のみなので compile=False を基本にする（カスタムメトリクス不要化）
    return tf.keras.models.load_model(model_path, compile=False)


def predict_label_strings(model, image_paths: list[str], batch_size: int = BATCH_SIZE):
    """
    Returns:
        list[tuple[path, label_str, conf]]:
            label_str: 4タスクのargmaxを結合した文字列（例: adfh）
            conf: 各タスクの最大確率の min（保守的）
    """
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []
        valid_paths = []
        for p in batch_paths:
            img = preprocess_image(p)
            if img is None:
                continue
            batch_imgs.append(img)
            valid_paths.append(p)

        if not batch_imgs:
            continue

        preds = model.predict(np.asarray(batch_imgs), verbose=0)
        # multi-output: list[np.ndarray] expected
        if isinstance(preds, dict):
            # Keras can return dict outputs; keep stable order by key name
            preds = [preds[k] for k in sorted(preds.keys())]
        elif isinstance(preds, np.ndarray):
            preds = [preds]

        if not isinstance(preds, list) or not preds:
            continue

        # Adapt to model output count (latest model may be single-task)
        task_label_sets = ALL_TASK_LABELS[: len(preds)]

        for j, p in enumerate(valid_paths):
            chars = []
            confs = []
            for task_idx, task_labels in enumerate(task_label_sets):
                task_probs = preds[task_idx]
                if task_probs is None or len(task_probs) <= j:
                    continue
                task_probs = task_probs[j]
                cls = int(np.argmax(task_probs))
                if cls < 0 or cls >= len(task_labels):
                    # output dimension mismatch; cannot reliably map to labels
                    return []
                chars.append(task_labels[cls])
                confs.append(float(task_probs[cls]))
            if not chars:
                continue
            label_str = "".join(chars)
            conf = float(min(confs)) if confs else 0.0
            results.append((p, label_str, conf))

    return results


def safe_move_dir(src_dir: str, dest_dir: str):
    """
    dest_dir が既に存在する場合は suffix を付けて衝突回避して move する。
    Returns: actual_dest_dir
    """
    base = dest_dir
    if not os.path.exists(base):
        os.makedirs(os.path.dirname(base), exist_ok=True)
        shutil.move(src_dir, base)
        return base

    k = 2
    while True:
        cand = f"{base}__dup{k}"
        if not os.path.exists(cand):
            os.makedirs(os.path.dirname(cand), exist_ok=True)
            shutil.move(src_dir, cand)
            return cand
        k += 1


def main():
    parser = argparse.ArgumentParser(
        description="Classify folders under master_data/未分類 using the latest model and move them into master_data/[label]/"
    )
    parser.add_argument("--unclassified_dir", type=str, default=os.path.join("master_data", "未分類"))
    parser.add_argument("--master_data_dir", type=str, default="master_data")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--min_images", type=int, default=10, help="Skip folder if fewer images than this")
    parser.add_argument("--dry_run", action="store_true", help="Do not move anything, only print/report")
    parser.add_argument("--report_path", type=str, default=os.path.join("outputs", "logs", "unclassified_reorg_report.json"))
    args = parser.parse_args()

    unclassified_dir = args.unclassified_dir
    if not os.path.exists(unclassified_dir):
        print(f"Unclassified directory not found: {unclassified_dir}")
        return

    model_path = args.model.strip() or find_latest_keras_model()
    if not model_path:
        print("Model not found. Provide --model or place a .keras under outputs/models/")
        return

    print(f"Using model: {model_path}")
    model = load_model_for_inference(model_path)

    # 対象: 未分類 直下の各フォルダ
    person_dirs = [
        os.path.join(unclassified_dir, d)
        for d in sorted(os.listdir(unclassified_dir))
        if os.path.isdir(os.path.join(unclassified_dir, d))
    ]
    if not person_dirs:
        print(f"No folders under: {unclassified_dir}")
        return

    report = {
        "model": model_path,
        "unclassified_dir": unclassified_dir,
        "moved": [],
        "skipped": [],
        "errors": [],
    }

    for person_dir in person_dirs:
        person_name = os.path.basename(person_dir)
        image_paths = []
        for ext in VALID_EXTS:
            image_paths.extend(glob(os.path.join(person_dir, "**", f"*{ext}"), recursive=True))
            image_paths.extend(glob(os.path.join(person_dir, "**", f"*{ext.upper()}"), recursive=True))
        image_paths = sorted(set(image_paths))

        if len(image_paths) < args.min_images:
            msg = f"skip (too few images): {person_name} ({len(image_paths)})"
            print(msg)
            report["skipped"].append({"person": person_name, "reason": "too_few_images", "count": len(image_paths)})
            continue

        try:
            preds = predict_label_strings(model, image_paths)
            if not preds:
                msg = f"skip (no readable images): {person_name}"
                print(msg)
                report["skipped"].append({"person": person_name, "reason": "no_readable_images", "count": len(image_paths)})
                continue

            counts = Counter([lab for _, lab, _ in preds])
            if not counts:
                msg = f"skip (no predictions): {person_name}"
                print(msg)
                report["skipped"].append({"person": person_name, "reason": "no_predictions", "count": len(image_paths)})
                continue
            top_label, top_count = counts.most_common(1)[0]
            # tie-break: avg confidence
            top_candidates = [lab for lab, c in counts.items() if c == top_count]
            if len(top_candidates) > 1:
                avg_conf = defaultdict(list)
                for _, lab, conf in preds:
                    if lab in top_candidates:
                        avg_conf[lab].append(conf)
                top_label = max(top_candidates, key=lambda lab: float(np.mean(avg_conf[lab]) if avg_conf[lab] else 0.0))

            dest_dir = os.path.join(args.master_data_dir, top_label, person_name)
            summary = {
                "person": person_name,
                "images_total": len(image_paths),
                "predicted_total": len(preds),
                "label_counts": dict(counts),
                "selected_label": top_label,
                "dest": dest_dir,
            }
            print(f"{person_name} -> {top_label} (n={counts[top_label]}/{len(preds)})")

            if args.dry_run:
                report["moved"].append({**summary, "dry_run": True})
                continue

            actual_dest = safe_move_dir(person_dir, dest_dir)
            report["moved"].append({**summary, "actual_dest": actual_dest})

        except Exception as e:
            print(f"error: {person_name}: {e}")
            report["errors"].append({"person": person_name, "error": str(e)})

    # report write
    try:
        os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report saved: {args.report_path}")
    except Exception as e:
        print(f"Failed to write report: {e}")


if __name__ == "__main__":
    main()
