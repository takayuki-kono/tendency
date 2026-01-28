"""
simple_split.py
master_data内の各ラベルフォルダごとに、人物フォルダをTrain/Validationに振り分けるスクリプト

ロジック:
    1. master_data直下の各ラベルフォルダ（adfh, bdgh等）を走査
    2. ラベルフォルダ内の人物フォルダをリストアップ
    3. 人物リストをシャッフルし、50:50でTrain/Validationに配分
    4. フォルダをコピー（移動元: master_data/label/person -> 移動先: train/label/person）
"""
import os
import shutil
import random

# パス設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MASTER_DATA_DIR = os.path.join(PROJECT_ROOT, "master_data")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "validation")

def copy_person(person_path, dest_root, label, person_name):
    """人物フォルダを指定先にコピー"""
    dest_path = os.path.join(dest_root, label, person_name)
    
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
            
        shutil.copytree(person_path, dest_path)
        print(f"    -> コピー: {dest_path}")
        return True
    except Exception as e:
        print(f"    !! エラー: {e}")
        return False

def main():
    print("="*60)
    print("ラベル別 Train/Validation 分割スクリプト (人物単位50:50)")
    print("="*60)
    print(f"ソース: {MASTER_DATA_DIR}")
    
    if not os.path.exists(MASTER_DATA_DIR):
        print("エラー: master_data ディレクトリが見つかりません。")
        return

    labels = [d for d in os.listdir(MASTER_DATA_DIR) 
              if os.path.isdir(os.path.join(MASTER_DATA_DIR, d))]
    
    if not labels:
        print("ラベルフォルダが見つかりませんでした。")
        return

    total_train = 0
    total_val = 0
    
    for label in labels:
        label_path = os.path.join(MASTER_DATA_DIR, label)
        print(f"\n[{label}] 処理中...")
        
        people = [p for p in os.listdir(label_path) 
                  if os.path.isdir(os.path.join(label_path, p))]
        
        if not people:
            print("  人物フォルダなし")
            continue
            
        print(f"  発見: {len(people)} 人")
        
        random.shuffle(people)
        
        mid = (len(people) + 1) // 2
        train_group = people[:mid]
        val_group = people[mid:]
        
        print(f"  配分: Train={len(train_group)}, Validation={len(val_group)}")
        
        if train_group:
            print("  -- Trainへコピー --")
            for person in train_group:
                src = os.path.join(label_path, person)
                copy_person(src, TRAIN_DIR, label, person)
                total_train += 1
                
        if val_group:
            print("  -- Validationへコピー --")
            for person in val_group:
                src = os.path.join(label_path, person)
                copy_person(src, VAL_DIR, label, person)
                total_val += 1
                
    print("\n" + "="*60)
    print(f"完了しました。")
    print(f"Train: {total_train} 人")
    print(f"Validation: {total_val} 人")

if __name__ == "__main__":
    main()
