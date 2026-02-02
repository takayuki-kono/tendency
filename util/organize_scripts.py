"""
メインパイプライン以外のスクリプトを util/ フォルダに移動するスクリプト
"""
import os
import shutil

PROJECT_ROOT = r"d:\tendency"
UTIL_DIR = os.path.join(PROJECT_ROOT, "util")

os.makedirs(UTIL_DIR, exist_ok=True)

# メインパイプラインスクリプト（移動しない）
MAIN_SCRIPTS = [
    "download_and_filter_faces.py",
    "reorganize_by_label.py",
    "create_person_split.py",
    "preprocess_multitask.py",
    "optimize_sequential.py",
    "train_sequential.py",
    "train_sequential_task_a.py",  # Task A用の学習スクリプト
]

# 整理用スクリプト（移動しない）
ORGANIZE_SCRIPTS = [
    "organize_outputs.py",
    "organize_scripts.py",
    "reorganize_master_data.py",
]

# 除外するスクリプト（移動しない）
EXCLUDE_SCRIPTS = MAIN_SCRIPTS + ORGANIZE_SCRIPTS

def main():
    moved = []
    
    for filename in os.listdir(PROJECT_ROOT):
        filepath = os.path.join(PROJECT_ROOT, filename)
        
        # ディレクトリはスキップ
        if os.path.isdir(filepath):
            continue
        
        # .py ファイルのみ対象
        if not filename.endswith(".py"):
            continue
        
        # メインスクリプトはスキップ
        if filename in EXCLUDE_SCRIPTS:
            continue
        
        # util/ に移動
        dest_path = os.path.join(UTIL_DIR, filename)
        try:
            shutil.move(filepath, dest_path)
            print(f"Moved: {filename} -> util/")
            moved.append(filename)
        except Exception as e:
            print(f"[ERROR] Failed to move {filename}: {e}")
    
    print(f"\nDone. Moved {len(moved)} scripts to util/")

if __name__ == "__main__":
    main()
