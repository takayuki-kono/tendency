import os
import shutil

TARGET_FILES = [
    "train_sequential_single.py",
    "run_optimization_single.bat",
    "components/train_single_trial.py",
    "components/train_single_trial_task_a.py",
    "components/train_filter_single_trial.py",
    "util/analyze_errors_single.py",
    "optimize_sequential_single.py",
    "preprocess_single.py"
]

TARGET_DIRS = [
    "preprocessed_single",
    "error_analysis_single"
]

DEST_DIR = "archive_single"

def move_items():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created directory: {DEST_DIR}")

    # Move Files
    for file_path in TARGET_FILES:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(DEST_DIR, file_name)
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path) # Overwrite
                shutil.move(file_path, dest_path)
                print(f"Moved: {file_path} -> {dest_path}")
            except Exception as e:
                print(f"Error moving {file_path}: {e}")
        else:
            print(f"Not found (skipped): {file_path}")

    # Move Directories
    for dir_path in TARGET_DIRS:
        if os.path.exists(dir_path):
            dir_name = os.path.basename(dir_path)
            dest_path = os.path.join(DEST_DIR, dir_name)
            try:
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path) # Overwrite
                shutil.move(dir_path, dest_path)
                print(f"Moved directory: {dir_path} -> {dest_path}")
            except Exception as e:
                print(f"Error moving directory {dir_path}: {e}")
        else:
            print(f"Directory not found (skipped): {dir_path}")

if __name__ == "__main__":
    move_items()
