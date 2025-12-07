import optuna
import subprocess
import sys
import re
import os
import logging

# Optunaのログ設定
file_handler = logging.FileHandler('optuna_log.txt', mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.get_logger("optuna").addHandler(file_handler)

# 標準出力もファイルに書き込むためのラッパー
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode, encoding='utf-8', buffering=1)
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

sys.stdout = Tee('script_output.txt', 'w')

# --- 設定項目 ---
# 最適化の試行回数
N_TRIALS = 30 
# 並列実行数 (-1を指定すると利用可能な全CPUコアを使用)
N_JOBS = 1 
# 既存の探索を継続するか (True: 継続, False: 新規・上書き)
CONTINUE_OPTIMIZATION = True

# Python実行環境のパス
PYTHON_PREPROCESS = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe"
PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
# パスが存在しない場合は、システムのデフォルト 'python' を使用
if not os.path.exists(PYTHON_PREPROCESS): PYTHON_PREPROCESS = "python"
if not os.path.exists(PYTHON_TRAIN): PYTHON_TRAIN = "python"

# OptunaのStudy名と永続化設定
STUDY_NAME = "filter_optimization_study"
STORAGE_NAME = "sqlite:///filter_optimization.db"

# --- 目的関数 ---
def objective(trial: optuna.Trial):
    """
    1回の試行を評価する関数
    """
    # 1. ハイパーパラメータの提案（すべてパーセンタイル）
    pitch_percentile = trial.suggest_int('pitch_percentile', 0, 50, step=5)
    symmetry_percentile = trial.suggest_int('symmetry_percentile', 0, 50, step=5)
    y_diff_percentile = trial.suggest_int('y_diff_percentile', 0, 75, step=5)
    
    trial_number = trial.number
    print(f"\n{'='*60}")
    print(f"Trial {trial_number}: Start | Pitch={pitch_percentile}%, Symmetry={symmetry_percentile}%, Y-Diff={y_diff_percentile}%")
    print(f"{'='*60}")

    try:
        # 2. 前処理スクリプトの実行
        cmd_pre = [
            PYTHON_PREPROCESS,
            "preprocess_multitask.py",
            "--pitch_percentile", str(pitch_percentile),
            "--symmetry_percentile", str(symmetry_percentile),
            "--y_diff_percentile", str(y_diff_percentile)
        ]
        print(f"Trial {trial_number}: Running preprocessing...")
        ret_pre = subprocess.run(cmd_pre, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret_pre.returncode != 0:
            print(f"Trial {trial_number}: Preprocessing failed. Stderr: {ret_pre.stderr}")
            # 失敗は探索に影響させない（環境エラーの可能性があるため）
            raise optuna.TrialPruned()

        # 3. 軽量学習スクリプトの実行
        cmd_train = [PYTHON_TRAIN, "train_for_filter_search.py"]
        print(f"Trial {trial_number}: Running evaluation training...")
        
        # subprocess.run を使ってシンプルに実行
        ret_train = subprocess.run(cmd_train, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if ret_train.returncode != 0:
            print(f"Trial {trial_number}: Training failed. Stderr: {ret_train.stderr}")
            raise optuna.TrialPruned()

        # 4. スコアの抽出
        output = ret_train.stdout
        # print(output) # デバッグ用に全出力を表示したい場合
        match = re.search(r"FINAL_SCORE:\s*(\d+\.\d+)", output)
        
        if match:
            score = float(match.group(1))
            print(f"--- Trial {trial_number}: Finished. Score: {score} ---")
            return score
        else:
            print(f"--- Trial {trial_number}: Score not found in output. ---")
            raise optuna.TrialPruned()
            
    except optuna.TrialPruned:
        raise  # TrialPruned は再スロー
    except Exception as e:
        print(f"An error occurred in trial {trial_number}: {e}")
        raise optuna.TrialPruned()  # その他のエラーも探索に影響させない

# --- メイン処理 ---
def main():
    """
    最適化プロセス全体を管理する関数
    """
    # 新規開始の場合は既存のDBを削除
    if not CONTINUE_OPTIMIZATION and os.path.exists("filter_optimization.db"):
        try:
            os.remove("filter_optimization.db")
            print("Deleted existing database for fresh start.")
        except PermissionError:
            print("Could not delete database. It might be in use.")

    # 効率的なサンプラーの設定
    # n_startup_trials: ランダム探索の回数を減らし、早期にベイズ最適化へ移行
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5,  # 最初の5回はランダム、その後はベイズ最適化
        seed=42  # 再現性のため
    )

    # Studyオブジェクトの作成（方向を最大化に設定）
    # 永続化: SQLiteを使い、中断・再開が可能
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        load_if_exists=True, # 常にTrueにしておき、リセットはDB削除で対応する
        direction='maximize',
        sampler=sampler
    )

    # 最適化の実行
    print(f"Starting optimization with {N_TRIALS} trials and {N_JOBS} parallel jobs...")
    study.optimize(
        objective, 
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        # 例外発生時に試行を失敗として記録し、最適化を続行する
        catch=(Exception,)
    ) 

    # 結果の表示
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"Best trial:")
    print(f"  Value (Score): {best_trial.value}")
    print(f"  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print("="*60)

    # リファインメント探索（2分探索での絞り込み）
    refine_search(study, objective)

def refine_search(study, objective_func):
    """
    上位2件のパラメータの中間値を探索し、精度を向上させる
    """
    print("\n" + "="*60)
    print("STARTING BINARY SEARCH REFINEMENT")
    print("="*60)
    
    max_refine_trials = 10  # リファインメントの最大回数
    
    for i in range(max_refine_trials):
        # 完了した試行のみを取得してソート
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed_trials.sort(key=lambda t: t.value, reverse=True)
        
        if len(completed_trials) < 2:
            print("Not enough trials to refine.")
            break
            
        best1 = completed_trials[0]
        best2 = completed_trials[1]
        
        print(f"Refine Step {i+1}:")
        print(f"  Best1: Score={best1.value}, Params={best1.params}")
        print(f"  Best2: Score={best2.value}, Params={best2.params}")
        
        # パラメータの中間値を計算
        new_params = {}
        changed = False
        
        for key in best1.params.keys():
            val1 = best1.params[key]
            val2 = best2.params[key]
            
            # 中間値（整数、切り捨て）
            mid_val = int((val1 + val2) / 2)
            
            # 変化がない場合はBest1の値を採用（変化フラグは立てない）
            # ただし、これでは同じ試行を繰り返すことになるので、
            # 少なくとも1つのパラメータが既存の試行と異なる必要がある。
            new_params[key] = mid_val
            
            # 既に探索済みの組み合わせかチェックするために、この時点では何もしない
            # 後でまとめてチェックする
        
        # 提案されたパラメータセットが既存の試行に含まれているか確認
        is_duplicate = False
        for t in completed_trials:
            if t.params == new_params:
                is_duplicate = True
                break
        
        if is_duplicate:
            print("  Proposed parameters already exist. Refinement converged.")
            break
            
        print(f"  Enqueuing new trial with params: {new_params}")
        
        # 新しい試行をキューに追加
        study.enqueue_trial(new_params)
        
        # 1回だけ実行
        # 注意: enqueue_trial したパラメータは次の optimize で消費される
        study.optimize(objective_func, n_trials=1, catch=(Exception,))
        
    print("="*60)
    print("REFINEMENT COMPLETE")
    print(f"Final Best Trial: Score={study.best_trial.value}")
    print(f"Params: {study.best_trial.params}")
    print("="*60)

if __name__ == "__main__":
    main()
