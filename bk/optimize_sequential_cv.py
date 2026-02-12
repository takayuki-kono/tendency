import subprocess
import sys
import re
import os
import logging
import time
import json
import hashlib
import winsound
import shutil
import random
from collections import defaultdict
import glob

# --- 設定 ---
PYTHON_PREPROCESS = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe"
PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"

# 元データ
SOURCE_TRAIN_DIR = "train"
SOURCE_VAL_DIR = "validation"

# CV用一時ディレクトリ
CV_RAW_DIR = "cv_temp_raw"
CV_PREPRO_DIR = "cv_temp_preprocessed"
CV_WORK_DIR = "cv_temp_work" # 学習ごとのTrain/Val配置場所

# ログ
LOG_DIR = "outputs/logs"
CACHE_DIR = "outputs/cache"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "filter_opt_cv_cache.json")
BEST_TRAIN_PARAMS_FILE = "outputs/best_train_params.json"

K_FOLDS = 3  # 3-Fold CV

# パス確認
if not os.path.exists(PYTHON_PREPROCESS): PYTHON_PREPROCESS = "python"
if not os.path.exists(PYTHON_TRAIN): PYTHON_TRAIN = "python"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'sequential_opt_cv_log.txt'), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def prepare_cv_raw_data():
    """TrainとValidationを結合してCV用のRawデータを作成"""
    if os.path.exists(CV_RAW_DIR):
        # 簡易チェック: 空でなければOKとするか、毎回作り直すか
        # データ変更の可能性を考えて毎回同期（更新分のみ）が理想だが、
        # ここでは安全のため、一旦削除して作り直す（時間がかかるようなら考える）
        # しかし画像コピーは重いので、rsync的な動きが望ましい。
        # 単純に cp -u (update) 的なことをする
        pass
    else:
        os.makedirs(CV_RAW_DIR)

    logger.info("Preparing CV Raw Dataset (Merging Train + Validation)...")
    
    # helper to copy
    def merge_dirs(src_root, dst_root):
        if not os.path.exists(src_root): return
        for class_dir in os.listdir(src_root): # adgh, etc.
            src_class = os.path.join(src_root, class_dir)
            if not os.path.isdir(src_class): continue
            
            dst_class = os.path.join(dst_root, class_dir)
            os.makedirs(dst_class, exist_ok=True)
            
            for person_dir in os.listdir(src_class):
                src_person = os.path.join(src_class, person_dir)
                dst_person = os.path.join(dst_class, person_dir)
                
                if os.path.isdir(src_person):
                    if not os.path.exists(dst_person):
                        shutil.copytree(src_person, dst_person)
                    else:
                        # 既に存在する場合、中のファイルを確認して足りないものをコピー
                        for f in os.listdir(src_person):
                            src_f = os.path.join(src_person, f)
                            dst_f = os.path.join(dst_person, f)
                            if not os.path.exists(dst_f):
                                shutil.copy2(src_f, dst_f)

    merge_dirs(SOURCE_TRAIN_DIR, CV_RAW_DIR)
    merge_dirs(SOURCE_VAL_DIR, CV_RAW_DIR)
    logger.info(f"CV Raw Dataset ready at {CV_RAW_DIR}")

def get_person_split(raw_dir, k=3):
    """
    クラスごとに人物ディレクトリをリストアップし、K分割する
    Returns: List of K splits. Each split is a dict {class_name: [person_list]}
    """
    splits = [defaultdict(list) for _ in range(k)]
    
    for class_name in os.listdir(raw_dir):
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_path): continue
        
        persons = [p for p in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, p))]
        random.shuffle(persons) # ランダムシャッフル
        
        # 分割
        for i, person in enumerate(persons):
            split_idx = i % k
            splits[split_idx][class_name].append(person)
            
    return splits

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

def load_best_train_params():
    if os.path.exists(BEST_TRAIN_PARAMS_FILE):
        try:
            with open(BEST_TRAIN_PARAMS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: pass
    return {}

def run_cv_trial(pitch, sym, y_diff, mouth_open, eb_eye_high, eb_eye_low, sharpness_low, sharpness_high, 
                 face_size_low, face_size_high, retouching, mask, glasses, grayscale, model_name='EfficientNetV2B0'):
    
    # 1. キャッシュ確認
    cache_key = f"CV_model={model_name}_pitch={pitch}_sym={sym}_ydiff={y_diff}_mouth={mouth_open}_ebh={eb_eye_high}_ebl={eb_eye_low}_sl={sharpness_low}_sh={sharpness_high}_fsl={face_size_low}_fsh={face_size_high}_ret={retouching}_mask={mask}_glass={glasses}_gray={grayscale}_k={K_FOLDS}"
    
    cache = load_cache()
    if cache_key in cache:
        logger.info(f"Cache Hit! Score: {cache[cache_key]}")
        return cache[cache_key]

    logger.info(f"Running CV Trial ({K_FOLDS}-Fold)... Params: {cache_key}")

    # 2. 全データを一括前処理 (CV_RAW_DIR -> CV_PREPRO_DIR/train)
    # preprocess_multitask.py は train_dir と val_dir を別々に処理するが、
    # ここでは CV_RAW_DIR を train_dir として渡し、val_dir はダミー(空)または無視させる対応が必要だが
    # validation dir が必須なら、適当な空フォルダを指定してエラー回避する。
    dummy_val = "dummy_val_for_prepro"
    os.makedirs(dummy_val, exist_ok=True)
    
    cmd_pre = [
        PYTHON_PREPROCESS,
        "preprocess_multitask.py",
        "--train_dir", CV_RAW_DIR,
        "--val_dir", dummy_val, # ダミー
        "--out_dir", CV_PREPRO_DIR,
        "--pitch_percentile", str(pitch),
        "--symmetry_percentile", str(sym),
        "--y_diff_percentile", str(y_diff),
        "--mouth_open_percentile", str(mouth_open),
        "--eyebrow_eye_percentile_high", str(eb_eye_high),
        "--eyebrow_eye_percentile_low", str(eb_eye_low),
        "--sharpness_percentile_low", str(sharpness_low),
        "--sharpness_percentile_high", str(sharpness_high),
        "--face_size_percentile_low", str(face_size_low),
        "--face_size_percentile_high", str(face_size_high),
        "--retouching_percentile", str(retouching),
        "--mask_percentile", str(mask),
        "--glasses_percentile", str(glasses)
    ]
    if grayscale:
        cmd_pre.append("--grayscale")
        
    logger.info("Running Batch Preprocessing...")
    ret_pre = subprocess.run(cmd_pre, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if ret_pre.returncode != 0:
        logger.error(f"Preprocessing failed: {ret_pre.stderr}")
        return 0.0

    # 3. K-Fold Loop
    preprocessed_pool_dir = os.path.join(CV_PREPRO_DIR, "train") # preprocess_multitaskの出力先
    splits = get_person_split(preprocessed_pool_dir, k=K_FOLDS)
    
    scores = []
    
    for k in range(K_FOLDS):
        logger.info(f"--- Fold {k+1}/{K_FOLDS} ---")
        
        # ワークスペース作成
        fold_train_dir = os.path.join(CV_WORK_DIR, "train")
        fold_val_dir = os.path.join(CV_WORK_DIR, "validation")
        
        if os.path.exists(CV_WORK_DIR): shutil.rmtree(CV_WORK_DIR)
        os.makedirs(fold_train_dir)
        os.makedirs(fold_val_dir)
        
        # データ配置 (人物単位でリンク/コピー)
        # Validation = splits[k]
        # Train = others
        
        # 全クラス走査
        for class_name in os.listdir(preprocessed_pool_dir):
            if not os.path.isdir(os.path.join(preprocessed_pool_dir, class_name)): continue
            
            src_class = os.path.join(preprocessed_pool_dir, class_name)
            
            # 宛先クラスフォルダ作成
            os.makedirs(os.path.join(fold_train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(fold_val_dir, class_name), exist_ok=True)
            
            val_persons = splits[k][class_name]
            
            for person_name in os.listdir(src_class):
                src_person = os.path.join(src_class, person_name)
                
                if person_name in val_persons:
                    # Validationへ
                    dst_person = os.path.join(fold_val_dir, class_name, person_name)
                else:
                    # Trainへ
                    dst_person = os.path.join(fold_train_dir, class_name, person_name)
                
                # コピー (シンボリックリンクだと権限エラーが出る場合があるが、早さを取るならジャンクション推奨。ここでは安全にcopytree)
                # 画像だけのフォルダなのでcopytreeでOK
                shutil.copytree(src_person, dst_person)
        
        # 学習実行 (パスは CV_WORK_DIR 以下の train/validation がハードコードされていると困る)
        # train_multitask_trial.py はディレクトリ設定がハードコードされている ("preprocessed_multitask/train" etc)
        # なので、引数でディレクトリを指定できるようにするか、一時的にディレクトリ名を合わせる必要がある。
        # train_multitask_trial.py を確認すると...
        # PREPROCESSED_TRAIN_DIR = 'preprocessed_multitask/train' (ハードコード)
        # これをオーバーライドする機能がない。
        # 解決策: train_multitask_trial.py を改造するのも手だが、
        # ここでは一時フォルダ名を "preprocessed_multitask" にリネームして使うハックを行うか、
        # あるいはそもそも CV_PREPRO_DIR の名前を工夫するか。
        # => train_multitask_trial.py を引数対応にするのが一番正しい。
        # しかし今からツール修正は手間。
        # 代わりに、シンボリックリンク（ジャンクション）で 'preprocessed_multitask' を CV_WORK_DIR に向ける。
        
        # 既存の 'preprocessed_multitask' を退避
        if os.path.exists("preprocessed_multitask"):
            if os.path.exists("preprocessed_multitask_backup"):
                shutil.rmtree("preprocessed_multitask_backup")
            os.rename("preprocessed_multitask", "preprocessed_multitask_backup")
        
        try:
            # CV_WORK_DIR を preprocessed_multitask としてシンボリックリンクまたはリネーム
            # リネームが早い
            os.rename(CV_WORK_DIR, "preprocessed_multitask")
            
            # 学習コマンド
            train_script = "components/train_multitask_trial.py"
            cmd_train = [PYTHON_TRAIN, train_script, "--model_name", model_name]
            
            best_params = load_best_train_params()
            for key, val in best_params.items():
                if key not in ['model_name', 'fine_tune']:
                    cmd_train.extend([f"--{key}", str(val)])
            
            # 高速化のためEpoch少なめ (CVなのである程度信頼できるが、毎回Fullは重い)
            # User指示が「精度求める」なのでEpoch 5 くらいは確保
            if "--epochs" not in cmd_train:
                cmd_train.extend(["--epochs", "5"])
            cmd_train.extend(["--fine_tune", "False"])
            
            ret_train = subprocess.run(cmd_train, capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            # 結果取得
            match = re.search(r"FINAL_VAL_ACCURACY:\s*([\d.]+)", ret_train.stdout)
            if match:
                score = float(match.group(1))
                logger.info(f"  Fold {k+1} Score: {score}")
                scores.append(score)
            else:
                logger.error(f"  Fold {k+1} Failed to get score.")
                logger.warning(ret_train.stdout)
                scores.append(0.0)
                
        finally:
            # 復元
            if os.path.exists("preprocessed_multitask"):
                # これは今回使ったTemp
                if os.path.exists(CV_WORK_DIR): shutil.rmtree(CV_WORK_DIR) # 安全のため
                os.rename("preprocessed_multitask", CV_WORK_DIR)
            
            if os.path.exists("preprocessed_multitask_backup"):
                os.rename("preprocessed_multitask_backup", "preprocessed_multitask")

    # 平均スコア
    avg_score = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"CV Result: {avg_score} (Scores: {scores})")
    
    # save cache
    cache = load_cache()
    cache[cache_key] = avg_score
    save_cache(cache)
    
    return avg_score

def optimize_single_param(target_name, current_params, model_name, points=[0, 10, 30]):
    # CVは重いので探索点を絞る
    logger.info(f"\n>>> Optimizing {target_name} [CV Mode] <<<")
    best_val = 0
    best_score = -1.0
    
    for p in points:
        test_params = current_params.copy()
        test_params[target_name] = p
        
        # 他のパラメータも渡す
        score = run_cv_trial(
            test_params['pitch'], test_params['sym'], test_params['y_diff'], test_params['mouth_open'],
            test_params['eb_eye_high'], test_params['eb_eye_low'],
            test_params['sharpness_low'], test_params['sharpness_high'],
            test_params['face_size_low'], test_params['face_size_high'],
            test_params['retouching'], test_params['mask_percentile'], test_params['glasses_percentile'],
            grayscale=test_params.get('grayscale', False),
            model_name=model_name
        )
        
        if score > best_score:
            best_score = score
            best_val = p
    
    logger.info(f"Best {target_name}: {best_val} (Score: {best_score})")
    return best_val

def main():
    logger.info("Starting Cross-Validation Optimization")
    prepare_cv_raw_data()
    
    # 初期パラメータ
    current_params = {
        'pitch': 0, 'sym': 0, 'y_diff': 0, 'mouth_open': 0,
        'eb_eye_high': 0, 'eb_eye_low': 0, 'sharpness_low': 0, 'sharpness_high': 0,
        'face_size_low': 0, 'face_size_high': 0, 'retouching': 0,
        'mask_percentile': 0, 'glasses_percentile': 0,
        'grayscale': False
    }
    
    best_model = 'EfficientNetV2B0' # 固定または探索
    
    # 段階的最適化 (重要なものから)
    # ポイントは少し荒く設定 (CVが重いため)
    
    # 1. Pitch
    current_params['pitch'] = optimize_single_param('pitch', current_params, best_model, points=[0, 10, 30])
    
    # 2. Symmetry
    current_params['sym'] = optimize_single_param('sym', current_params, best_model, points=[0, 10, 30])
    
    # 3. Y-Diff
    current_params['y_diff'] = optimize_single_param('y_diff', current_params, best_model, points=[0, 10])
    
    # 4. Mouth Open
    current_params['mouth_open'] = optimize_single_param('mouth_open', current_params, best_model, points=[0, 30])
    
    # 5. Eyebrow High/Low
    current_params['eb_eye_high'] = optimize_single_param('eb_eye_high', current_params, best_model, points=[0, 10])
    current_params['eb_eye_low'] = optimize_single_param('eb_eye_low', current_params, best_model, points=[0, 10])
    
    # 6. Sharpness Low/High
    current_params['sharpness_low'] = optimize_single_param('sharpness_low', current_params, best_model, points=[0, 5])
    current_params['sharpness_high'] = optimize_single_param('sharpness_high', current_params, best_model, points=[0, 50]) # 100=Off for high cut? No, 0=Off. preprocess says "Filter top X%". So 0 is keep all.
    
    # 7. Face Size Low/High
    current_params['face_size_low'] = optimize_single_param('face_size_low', current_params, best_model, points=[0, 10])
    current_params['face_size_high'] = optimize_single_param('face_size_high', current_params, best_model, points=[0, 10])

    # 8. Retouching
    current_params['retouching'] = optimize_single_param('retouching', current_params, best_model, points=[0, 30, 50])
    
    # 9. Mask
    current_params['mask_percentile'] = optimize_single_param('mask_percentile', current_params, best_model, points=[0, 20, 50])
    
    # 10. Glasses
    current_params['glasses_percentile'] = optimize_single_param('glasses_percentile', current_params, best_model, points=[0, 20, 50])
    
    # 11. Grayscale Check
    logger.info("Checking Grayscale...")
    score_color = run_cv_trial(**{**current_params, 'grayscale': False}, model_name=best_model)
    score_gray = run_cv_trial(**{**current_params, 'grayscale': True}, model_name=best_model)
    if score_gray > score_color:
        current_params['grayscale'] = True
        logger.info("Grayscale Selected")
    
    logger.info("="*50)
    logger.info(f"Final Optimized Params: {current_params}")
    logger.info("="*50)
    
    # 結果コマンド生成
    final_cmd = (
        f"{PYTHON_PREPROCESS} preprocess_multitask.py --out_dir preprocessed_multitask "
        f"--pitch_percentile {current_params['pitch']} "
        f"--symmetry_percentile {current_params['sym']} "
        f"--y_diff_percentile {current_params['y_diff']} "
        f"--mouth_open_percentile {current_params['mouth_open']} "
        f"--eyebrow_eye_percentile_high {current_params['eb_eye_high']} "
        f"--eyebrow_eye_percentile_low {current_params['eb_eye_low']} "
        f"--sharpness_percentile_low {current_params['sharpness_low']} "
        f"--sharpness_percentile_high {current_params['sharpness_high']} "
        f"--face_size_percentile_low {current_params['face_size_low']} "
        f"--face_size_percentile_high {current_params['face_size_high']} "
        f"--retouching_percentile {current_params['retouching']} "
        f"--mask_percentile {current_params['mask_percentile']} "
        f"--glasses_percentile {current_params['glasses_percentile']} "
    )
    if current_params['grayscale']:
        final_cmd += "--grayscale "
    logger.info(final_cmd)

if __name__ == "__main__":
    main()
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
