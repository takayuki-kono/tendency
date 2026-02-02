"""
プロジェクト直下の出力ファイルを outputs/ 以下に整理するスクリプト
"""
import os
import shutil

PROJECT_ROOT = r"d:\tendency"

# 出力先ディレクトリ
LOGS_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
CACHE_DIR = os.path.join(PROJECT_ROOT, "outputs", "cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 移動ルール
RULES = {
    "logs": [".txt", ".log"],           # ログファイル
    "cache": [".json"],                  # キャッシュ/設定ファイル
    "models": [".keras", ".h5", ".tflite"]  # モデルファイル
}

# 除外するファイル（移動しない）
EXCLUDE_FILES = [
    "requirements.txt",
    "env.txt",
    "_workdir.txt",
    "kokone.txt",
    "n.txt",
]

def main():
    moved_count = {"logs": 0, "cache": 0, "models": 0}
    
    for filename in os.listdir(PROJECT_ROOT):
        filepath = os.path.join(PROJECT_ROOT, filename)
        
        # ディレクトリはスキップ
        if os.path.isdir(filepath):
            continue
        
        # 除外ファイルはスキップ
        if filename in EXCLUDE_FILES:
            continue
        
        # 拡張子で分類
        ext = os.path.splitext(filename)[1].lower()
        
        dest_dir = None
        category = None
        
        if ext in RULES["models"]:
            dest_dir = MODELS_DIR
            category = "models"
        elif ext in RULES["cache"]:
            dest_dir = CACHE_DIR
            category = "cache"
        elif ext in RULES["logs"]:
            dest_dir = LOGS_DIR
            category = "logs"
        
        if dest_dir:
            dest_path = os.path.join(dest_dir, filename)
            try:
                shutil.move(filepath, dest_path)
                print(f"[{category}] {filename} -> {dest_dir}")
                moved_count[category] += 1
            except Exception as e:
                print(f"[ERROR] Failed to move {filename}: {e}")
    
    # .bat ファイルを batch/ フォルダに移動
    BATCH_DIR = os.path.join(PROJECT_ROOT, "batch")
    os.makedirs(BATCH_DIR, exist_ok=True)
    moved_bat = 0
    for filename in os.listdir(PROJECT_ROOT):
        if filename.endswith(".bat"):
            filepath = os.path.join(PROJECT_ROOT, filename)
            dest_path = os.path.join(BATCH_DIR, filename)
            try:
                shutil.move(filepath, dest_path)
                print(f"[batch] {filename} -> batch/")
                moved_bat += 1
            except Exception as e:
                print(f"[ERROR] Failed to move {filename}: {e}")
    
    print(f"\nDone. Moved: logs={moved_count['logs']}, cache={moved_count['cache']}, models={moved_count['models']}, batch={moved_bat}")

if __name__ == "__main__":
    main()
