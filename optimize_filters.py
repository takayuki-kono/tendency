import keras_tuner as kt
import subprocess
import sys
import re
import os
import shutil

# 最適化設定
MAX_TRIALS = 30
INITIAL_POINTS = 5
PROJECT_NAME = 'filter_optimization'
DIR_NAME = 'filter_opt_logs'

# Pythonパス設定
PYTHON_PREPROCESS = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe"
PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"

if not os.path.exists(PYTHON_PREPROCESS): PYTHON_PREPROCESS = "python"
if not os.path.exists(PYTHON_TRAIN): PYTHON_TRAIN = "python"

class FilterHyperModel(kt.HyperModel):
    def build(self, hp):
        # このメソッドはモデル構築用だが、今回はダミーモデルを返す
        # 実際の評価は fit() 内で行う（Keras Tunerのハック的な使い方）
        return None

    def fit(self, hp, model, *args, **kwargs):
        # 1. パラメータ決定
        pitch = hp.Int('pitch', min_value=0, max_value=50, step=5)
        symmetry = hp.Int('symmetry', min_value=0, max_value=50, step=5)
        
        print(f"\n{'='*60}")
        print(f"Trial: Pitch={pitch}, Symmetry={symmetry}")
        print(f"{'='*60}")

        # 2. 前処理実行
        cmd_pre = [
            PYTHON_PREPROCESS, "preprocess_multitask.py",
            "--pitch_percentile", str(pitch),
            "--symmetry_percentile", str(symmetry)
        ]
        print("Running preprocessing...")
        ret_pre = subprocess.run(cmd_pre, capture_output=True, text=True)
        if ret_pre.returncode != 0:
            print(f"Preprocessing failed: {ret_pre.stderr}")
            return 0.0 # 失敗したらスコア0

        # 3. 軽量学習実行
        cmd_train = [PYTHON_TRAIN, "train_for_filter_search.py"]
        print("Running evaluation training...")
        # 出力をリアルタイム表示しつつキャプチャ
        process = subprocess.Popen(cmd_train, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        
        output = ""
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                print(line.strip())
                output += line
        
        if process.returncode != 0:
            print("Training failed.")
            return 0.0

        # 4. スコア抽出
        match = re.search(r"FINAL_SCORE:\s*(\d+\.\d+)", output)
        if match:
            score = float(match.group(1))
            print(f"Score: {score}")
            return score
        else:
            print("Score not found.")
            return 0.0

class FilterOracle(kt.oracles.BayesianOptimizationOracle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FilterTuner(kt.Tuner):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        score = self.hypermodel.fit(hp, None, *args, **kwargs)
        # スコアを報告
        self.oracle.update_trial(trial.trial_id, {'val_accuracy': score})
        self.save_model(trial.trial_id, None) # ダミー保存

    def save_model(self, trial_id, model, step=0):
        pass

    def load_model(self, trial):
        return None

def main():
    tuner = FilterTuner(
        oracle=FilterOracle(
            objective=kt.Objective('val_accuracy', direction='max'),
            max_trials=MAX_TRIALS,
            num_initial_points=INITIAL_POINTS,
        ),
        hypermodel=FilterHyperModel(),
        directory=DIR_NAME,
        project_name=PROJECT_NAME,
        overwrite=True
    )

    tuner.search()

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print(f"Best Pitch: {best_hps.get('pitch')}")
    print(f"Best Symmetry: {best_hps.get('symmetry')}")
    print("="*60)

if __name__ == "__main__":
    main()
