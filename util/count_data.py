import os
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", encoding='utf-8', force=True)
logger = logging.getLogger(__name__)

# ディレクトリのパス
PREPRO_DIR = "preprocessed_multitask"
PREPRO_TRAIN = os.path.join(PREPRO_DIR, "train")
PREPRO_VALID = os.path.join(PREPRO_DIR, "validation")

def count_images_in_directory(base_dir):
    """指定されたベースディレクトリ内の各サブディレクトリの画像数をカウントする"""
    if not os.path.exists(base_dir):
        logger.warning(f"Directory not found: {base_dir}")
        return {}

    counts = {}
    total_images = 0
    
    logger.info(f"Counting images in: {base_dir}")

    for label_name in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        image_count = 0
        for fn in os.listdir(label_path):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_count += 1
        
        counts[label_name] = image_count
        total_images += image_count
        logger.info(f"  {label_name}: {image_count} images")
            
    logger.info(f"Total images in {base_dir}: {total_images}")
    return counts, total_images

def analyze_task_distribution(counts, total_images, dataset_name):
    """タスクごとのクラス分布を分析"""
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

    train_counts, total_train = count_images_in_directory(PREPRO_TRAIN)
    logger.info("\n--- Train Data Summary ---")
    for label, count in sorted(train_counts.items()):
        logger.info(f"  {label}: {count} images")
    logger.info(f"Total Train Images: {total_train}")
    
    # タスクごとの分布を表示
    analyze_task_distribution(train_counts, total_train, "Train")

    val_counts, total_val = count_images_in_directory(PREPRO_VALID)
    logger.info("\n--- Validation Data Summary ---")
    for label, count in sorted(val_counts.items()):
        logger.info(f"  {label}: {count} images")
    logger.info(f"Total Validation Images: {total_val}")
    
    # タスクごとの分布を表示
    analyze_task_distribution(val_counts, total_val, "Validation")

    logger.info("\n--- Image Count Script Finished ---")

if __name__ == "__main__":
    main()
