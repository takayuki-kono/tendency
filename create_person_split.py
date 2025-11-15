import os
import shutil
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", encoding='utf-8', force=True)
logger = logging.getLogger(__name__)

# --- 設定 ---
# 元データが入っている親フォルダ
MASTER_DIR = "master_data"

# 出力先のフォルダ
TRAIN_DIR = "train"
VALIDATION_DIR = "validation"
# ---

def create_person_split():
    """
    MASTER_DIRから人物ベースでデータを読み込み、trainとvalidationに分割する。
    
    ルール：
    - 各クラスフォルダ内の人物フォルダが2つの場合 -> 1人目をtrain、2人目をvalidationへ。
    - 各クラスフォルダ内の人物フォルダが1つの場合 -> 1人目をtrainのみへ（検証データなし）。
    """
    
    # --- 1. 出力先フォルダをクリーンアップ ---
    for dir_path in [TRAIN_DIR, VALIDATION_DIR]:
        if os.path.exists(dir_path):
            logger.info(f"既存のフォルダを削除します: {dir_path}")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        logger.info(f"フォルダを作成しました: {dir_path}")

    if not os.path.exists(MASTER_DIR):
        logger.error(f"マスターデータフォルダが見つかりません: {MASTER_DIR}")
        logger.error("事前に 'master_data' フォルダを作成し、'master_data/<クラス名>/<人物名>/' の構造で画像を配置してください。")
        return

    logger.info(f"--- '{MASTER_DIR}' からの人物ベース分割を開始します ---")

    # --- 2. 各クラスを処理 ---
    for class_name in sorted(os.listdir(MASTER_DIR)):
        class_path = os.path.join(MASTER_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        person_folders = sorted([d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))])
        
        # --- 3. 人物フォルダの数に応じて分割 ---
        if len(person_folders) == 0:
            logger.warning(f"[{class_name}] には人物フォルダがありません。スキップします。")
            continue
        
        # --- 2人いる場合 -> train/validationに分割 ---
        if len(person_folders) >= 2:
            person_train = person_folders[0]
            person_val = person_folders[1]
            
            # 学習データへコピー
            src_train = os.path.join(class_path, person_train)
            dst_train = os.path.join(TRAIN_DIR, class_name, person_train)
            shutil.copytree(src_train, dst_train)
            
            # 検証データへコピー
            src_val = os.path.join(class_path, person_val)
            dst_val = os.path.join(VALIDATION_DIR, class_name, person_val)
            shutil.copytree(src_val, dst_val)
            
            logger.info(f"[{class_name}]: {person_train} -> train, {person_val} -> validation に分割しました。")
            
            if len(person_folders) > 2:
                logger.warning(f"[{class_name}] には3人以上の人物がいますが、最初の2人のみ使用しました。")

        # --- 1人しかいない場合 -> trainのみ ---
        elif len(person_folders) == 1:
            person_train = person_folders[0]
            
            # 学習データへコピー
            src_train = os.path.join(class_path, person_train)
            dst_train = os.path.join(TRAIN_DIR, class_name, person_train)
            shutil.copytree(src_train, dst_train)
            
            logger.info(f"[{class_name}]: {person_train} -> train のみ追加しました。(検証データなし)")

    logger.info("--- すべての分割処理が完了しました ---")


if __name__ == "__main__":
    create_person_split()
