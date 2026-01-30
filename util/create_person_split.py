"""
create_person_split.py
master_data内の人物フォルダをtrain/validationに振り分けるスクリプト

使用方法:
    python util/create_person_split.py

動作:
    1. master_data/ラベル/人物名 の構造をスキャン
    2. 各ラベルごとに、見つかった人物を交互にtrain/validationに振り分け
"""
import os
import shutil


# スクリプトの場所からプロジェクトルートを特定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ディレクトリ設定
MASTER_DATA_DIR = os.path.join(PROJECT_ROOT, "master_data")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "validation")

def find_all_people():
    """master_data内の全人物を検索し、(人物名, ラベル, フルパス)のリストを返す"""
    people = []
    
    if not os.path.exists(MASTER_DATA_DIR):
        print(f"エラー: master_dataが見つかりません: {MASTER_DATA_DIR}")
        return people
    
    # master_data/ラベル/人物名 の構造をスキャン
    for label_dir in os.listdir(MASTER_DATA_DIR):
        label_path = os.path.join(MASTER_DATA_DIR, label_dir)
        
        if not os.path.isdir(label_path):
            continue
            
        # ラベルフォルダ内の人物フォルダを列挙
        for person_name in os.listdir(label_path):
            person_path = os.path.join(label_path, person_name)
            
            if os.path.isdir(person_path):
                people.append({
                    "name": person_name,
                    "label": label_dir,
                    "path": person_path
                })
    
    return people

def copy_person(person_info, dest_root):
    """人物フォルダを指定先にコピー"""
    src = person_info["path"]
    label = person_info["label"]
    name = person_info["name"]
    
    # コピー先: dest_root/label/person_name
    dest = os.path.join(dest_root, label, name)
    
    # 親ディレクトリを作成
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # 既存があれば削除
    if os.path.exists(dest):
        shutil.rmtree(dest)
    
    # コピー実行
    shutil.copytree(src, dest)
    print(f"  コピー: {name} -> {dest}")

def create_split():
    print("=" * 50)
    print("Train/Validation 分割スクリプト (ラベルごとの交互振り分け版)")
    print("=" * 50)
    print(f"プロジェクトルート: {PROJECT_ROOT}")
    print(f"データソース: {MASTER_DATA_DIR}")
    print()
    
    # 全人物を検索
    people = find_all_people()
    
    if not people:
        print("人物フォルダが見つかりませんでした。")
        return
    
    # ラベルごとにグループ化
    people_by_label = {}
    for p in people:
        label = p['label']
        if label not in people_by_label:
            people_by_label[label] = []
        people_by_label[label].append(p)

    train_group = []
    val_group = []
    
    print(f"検出した人物数: {len(people)}")
    print("ラベルごとの振り分け:")
    
    for label, group in sorted(people_by_label.items()):
        # 名前順でソートして順序を固定
        group.sort(key=lambda x: x['name'])
        print(f"  [{label}] Total: {len(group)}")
        
        for i, p in enumerate(group):
            if i % 2 == 0:
                train_group.append(p)
                print(f"    - {p['name']} -> Train")
            else:
                val_group.append(p)
                print(f"    - {p['name']} -> Validation")
    print()
    
    print(f"Train グループ ({len(train_group)}人)")
    print(f"Validation グループ ({len(val_group)}人)")
    print()
    
    # コピー実行
    print("--- Train へのコピー ---")
    for p in train_group:
        copy_person(p, TRAIN_DIR)
    
    print()
    print("--- Validation へのコピー ---")
    for p in val_group:
        copy_person(p, VAL_DIR)
    
    print()
    print("分割完了！")
    print(f"Train: {TRAIN_DIR}")
    print(f"Validation: {VAL_DIR}")

if __name__ == "__main__":
    create_split()
