"""
create_single_split.py
master_data内の人物フォルダをtrain/validationに振り分けるスクリプト

使用方法:
    python util/create_single_split.py

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
    """master_data内の全人物を検索し、(人物名, ラベル, フルパス, 画像数)のリストを返す"""
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
                # 画像数をカウント
                valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                image_count = len([
                    f for f in os.listdir(person_path) 
                    if f.lower().endswith(valid_extensions)
                ])
                
                people.append({
                    "name": person_name,
                    "label": label_dir,
                    "path": person_path,
                    "image_count": image_count
                })
    
    return people

def copy_person(person_info, dest_root):
    """人物フォルダを指定先にコピー"""
    src = person_info["path"]
    label = person_info["label"]
    name = person_info["name"]
    count = person_info["image_count"]
    
    # コピー先: dest_root/label/person_name
    dest = os.path.join(dest_root, label, name)
    
    # 親ディレクトリを作成
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # 既存があれば削除
    if os.path.exists(dest):
        shutil.rmtree(dest)
    
    # コピー実行
    shutil.copytree(src, dest)
    print(f"  コピー: {name} ({count}枚) -> {dest}")

def create_split():
    print("=" * 50)
    print("Train/Validation 分割スクリプト (画像数上位50% Train版)")
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
        # 画像数が多い順にソート
        group.sort(key=lambda x: x['image_count'], reverse=True)
        
        total_people = len(group)
        # 上位50%を計算（切り上げでTrainを多めにするか、ユーザー指定通りバッサリ50%か。
        # 「上位50%」という言葉通りなら、全体の半分。
        # 10人 -> 5人。11人 -> 5.5人 (5人 or 6人)
        # ここでは math.ceil を使って「少なくとも半数はTrain」にするか、
        # シンプルに半分で切るか。
        # Pythonのスライス [:n] は n個要素を取る。
        # n = total_people // 2 だと、11 // 2 = 5。残り6人がValidation。
        # Trainが多いほうが一般的には嬉しいが、「上位50%」という表現には忠実。
        # 今回は割り算(商)を使います。
        split_idx = total_people // 2
        
        # もし1人しかいない場合はTrainに入れる (0になってしまうのを防ぐため例外処理する？)
        # いや、1 // 2 = 0 なのでTrain 0人, Val 1人になってしまう。
        # 1人しかいないならTrainに入れたいのが人情だが、「上位50%」なら0になる。
        # ここは「最低1人はTrain」というロジックを入れるか、
        # あるいは「上位50% (端数切り上げ)」で (total + 1) // 2 にするか。
        # (11+1)//2 = 6 (Train 6, Val 5)
        # (1+1)//2 = 1 (Train 1, Val 0)
        # こっちのほうが安全そう。
        
        split_idx = (total_people + 1) // 2
        
        train_subset = group[:split_idx]
        val_subset = group[split_idx:]
        
        print(f"  [{label}] Total: {total_people}")
        
        for p in train_subset:
            train_group.append(p)
            print(f"    - {p['name']} ({p['image_count']}枚) -> Train")
            
        for p in val_subset:
            val_group.append(p)
            print(f"    - {p['name']} ({p['image_count']}枚) -> Validation")
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
