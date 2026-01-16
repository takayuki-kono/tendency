import os
import shutil

# Root directory of image data
BASE_DIR = "master_data"

# Mapping: {Actress Name: Label String (e.g., 'bdgh')}
# 人物名とラベルのマッピング定義
LABEL_MAP = {
    "前田敦子": "bdgh",
    "坂井真紀": "bdgh",
    "森川葵": "bdgi",
    "真木よう子": "bdfh",
    "蓮佛美沙子": "bdgi",
    "ソニン": "bdfh",
    "山田杏奈": "befh",
    "新垣結衣": "befh",
    "大原麗子": "bdfi",
    "玉城ティナ": "befi",
    "南沙良": "adfh",
    "桜田ひより": "befi",
    "長谷川京子": "adfh",
    "桜庭ななみ": "adfi",
    "瀧本美織": "bdfi",
    "木南晴夏": "adgh",
    "りょう": "aegh",
    "臼田あさ美": "aefh",
    "井川遥": "begh",
    "米倉涼子": "aefh",
    "中谷美紀": "adgh",
    "木村文乃": "begh",
    "内田理央": "aegh",
    "吉高由里子": "begi",
    "中条あやみ": "adfi",
    "薬師丸ひろ子": "begi"
}

def reorganize():
    if not os.path.exists(BASE_DIR):
        print(f"Base directory '{BASE_DIR}' does not exist.")
        return

    print("Starting reorganization...")
    
    for person_name, label in LABEL_MAP.items():
        # Source directory for the actress
        src_path = os.path.join(BASE_DIR, person_name)
        
        # If source does not exist, skip
        if not os.path.exists(src_path):
            continue
            
        # Destination directory: master_data/label/person_name/
        dest_parent = os.path.join(BASE_DIR, label)
        dest_path = os.path.join(dest_parent, person_name)
        
        if os.path.normpath(src_path) == os.path.normpath(dest_path):
            print(f"Already in place: {person_name}")
            continue

        try:
            # Create label directory if needed
            os.makedirs(dest_parent, exist_ok=True)
            
            # Move the entire person directory to the new location
            if os.path.exists(dest_path):
                print(f"Destination {dest_path} already exists. Merging/Overwriting...")
                # If destination exists, move files individually
                for item in os.listdir(src_path):
                    s = os.path.join(src_path, item)
                    d = os.path.join(dest_path, item)
                    if os.path.isdir(s):
                        # Simple directory move/copy not handled in deep merge for now
                        # but normally flat structure expected inside person dir
                        pass 
                    else:
                        if os.path.exists(d): os.remove(d) # Overwrite
                        shutil.move(s, d)
                os.rmdir(src_path) # Remove old empty dir
            else:
                shutil.move(src_path, dest_path)
            
            print(f"Moved: {person_name} -> {label}/{person_name}")
            
        except Exception as e:
            print(f"Error moving {person_name}: {e}")

    print("Reorganization complete.")

if __name__ == "__main__":
    reorganize()
