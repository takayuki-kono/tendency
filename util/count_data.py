import os
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", encoding='utf-8', force=True)
logger = logging.getLogger(__name__)

# ディレクトリのパス
DIRS_TO_COUNT = [
    ("Master Data", "master_data"),
    ("Root Train", "train"),
    ("Root Validation", "validation"),
    ("Preprocessed Train", os.path.join("preprocessed_multitask", "train")),
    ("Preprocessed Validation", os.path.join("preprocessed_multitask", "validation")),
]

def count_images_in_directory(base_dir):
    """指定されたベースディレクトリ内の各サブディレクトリの画像数をカウントする (再帰的)"""
    if not os.path.exists(base_dir):
        return {}, 0

    counts = {}
    total_images = 0
    
    # サブディレクトリ（ラベル）ごとにループ
    for label_name in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        image_count = 0
        # ラベルディレクトリ内を再帰的に探索
        for root, dirs, files in os.walk(label_path):
            for fn in files:
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    image_count += 1
        
        counts[label_name] = image_count
        total_images += image_count
            
    return counts, total_images

def analyze_task_distribution(counts, total_images, dataset_name):
    """タスクごとのクラス分布を分析"""
    if total_images == 0:
        return

    # タスク定義
    TASK_A_LABELS = ['a', 'b', 'c']
    TASK_B_LABELS = ['d', 'e']
    TASK_C_LABELS = ['f', 'g']
    TASK_D_LABELS = ['h', 'i']
    ALL_TASKS = [
        ('Task A', TASK_A_LABELS),
        ('Task B', TASK_B_LABELS),
        ('Task C', TASK_C_LABELS),
        ('Task D', TASK_D_LABELS)
    ]
    
    logger.info(f"\n=== {dataset_name} Per-Task Distribution ===")
    
    for task_name, task_labels in ALL_TASKS:
        task_idx = ALL_TASKS.index((task_name, task_labels))
        
        # 各ラベルのカウントを集計
        label_counts = {label: 0 for label in task_labels}
        for folder_name, count in counts.items():
            if len(folder_name) > task_idx:
                char = folder_name[task_idx]
                if char in label_counts:
                    label_counts[char] += count
        
        task_total = sum(label_counts.values())
        
        logger.info(f"\n--- {task_name} ---")
        for label in task_labels:
            cnt = label_counts[label]
            pct = (cnt / task_total * 100) if task_total > 0 else 0
            bar = "█" * int(pct / 5)  # 5%ごとに1ブロック
            logger.info(f"  {label}: {cnt:>5} ({pct:>5.1f}%) {bar}")


def main():
    logger.info("--- Starting Image Count Script ---")

    all_summaries = []

    for name, path in DIRS_TO_COUNT:
        counts, total = count_images_in_directory(path)
        if total > 0:
            logger.info(f"\n--- {name} Summary ({path}) ---")
            for label, count in sorted(counts.items()):
                logger.info(f"  {label}: {count} images")
            logger.info(f"Total {name} Images: {total}")
            
            # タスクごとの分布を表示
            analyze_task_distribution(counts, total, name)
            all_summaries.append((name, total))
        else:
            if os.path.exists(path):
                logger.info(f"\n--- {name} Summary ({path}) ---")
                logger.info(f"  No images found or directory is empty.")
            else:
                logger.info(f"\n--- {name} Summary ({path}) ---")
                logger.info(f"  Directory does not exist.")

    logger.info("\n" + "="*40)
    logger.info("OVERALL SUMMARY")
    logger.info("="*40)
    for name, total in all_summaries:
        logger.info(f"{name:<25}: {total:>6} images")
    logger.info("="*40)

    logger.info("\n--- Image Count Script Finished ---")

if __name__ == "__main__":
    main()
