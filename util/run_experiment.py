import subprocess
import argparse
import csv
import re
import datetime
import os
import sys

def run_command(command):
    print(f"\nRunning: {command}")
    # バイト列として読み込み、手動でデコードする
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ""
    while True:
        line_bytes = process.stdout.readline()
        if line_bytes == b'' and process.poll() is not None:
            break
        if line_bytes:
            # 複数のエンコーディングを試す、または replace で凌ぐ
            try:
                line = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    line = line_bytes.decode('cp932')
                except UnicodeDecodeError:
                    line = line_bytes.decode('utf-8', errors='replace')
            
            # 表示時はコンソールのエンコーディングに合わせて安全に出力
            try:
                print(line.strip())
            except UnicodeEncodeError:
                print(line.strip().encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
                
            output += line
            
    return output, process.returncode

def parse_results(output):
    results = {}
    # Parse "Final Val Acc Task A (拡張ゾーン): 0.xxxx"
    # 正規表現を少し緩める（日本語部分に依存しないように）
    patterns = {
        'Task A': r"Final Val Acc Task A.*:\s*(\d+\.\d+)",
        'Task B': r"Final Val Acc Task B.*:\s*(\d+\.\d+)",
        'Task C': r"Final Val Acc Task C.*:\s*(\d+\.\d+)",
        'Task D': r"Final Val Acc Task D.*:\s*(\d+\.\d+)",
    }
    
    print("\n--- Parsing Results ---")
    found_any = False
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            val = float(match.group(1))
            results[key] = val
            print(f"Found {key}: {val}")
            found_any = True
    
    if not found_any:
        print("Warning: No accuracy results found in output.")
        # デバッグ用に末尾を表示
        print("Output tail (last 500 chars):")
        print(output[-500:])
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run experiment: Preprocess -> Train -> Log")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch filter percentile")
    parser.add_argument("--symmetry", type=int, default=0, help="Symmetry filter percentile")
    parser.add_argument("--y_diff", type=int, default=0, help="Y-diff filter percentile")
    parser.add_argument("--mouth_open", type=int, default=0, help="Mouth open filter percentile")
    parser.add_argument("--note", type=str, default="", help="Note for this experiment")
    parser.add_argument("--skip_pre", action="store_true", help="Skip preprocessing step")
    args = parser.parse_args()
    
    # Define Python paths
    # Preprocess runs in 'windows_gpu' environment
    PYTHON_PREPROCESS = r"d:\tendency\.venv_windows_gpu\Scripts\python.exe"
    # Train runs in 'tendency' environment (tendency.venv_tf210_gpu)
    PYTHON_TRAIN = r"d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe"
    
    # Verify paths exist
    if not os.path.exists(PYTHON_PREPROCESS):
        print(f"Warning: Preprocess Python not found at {PYTHON_PREPROCESS}. Using default 'python'.")
        PYTHON_PREPROCESS = "python"
    if not os.path.exists(PYTHON_TRAIN):
        print(f"Warning: Train Python not found at {PYTHON_TRAIN}. Using default 'python'.")
        PYTHON_TRAIN = "python"

    # 1. Preprocess
    if not args.skip_pre:
        cmd_pre = f'"{PYTHON_PREPROCESS}" preprocess_multitask.py --pitch_percentile {args.pitch} --symmetry_percentile {args.symmetry} --y_diff_percentile {args.y_diff} --mouth_open_percentile {args.mouth_open}'
        print("="*60)
        print("STEP 1: PREPROCESSING")
        print("="*60)
        out_pre, ret_pre = run_command(cmd_pre)
        if ret_pre != 0:
            print(f"Preprocessing failed with return code {ret_pre}. Aborting.")
            sys.exit(ret_pre)
    else:
        print("Skipping preprocessing step.")
    
    # 2. Train
    print("\n" + "="*60)
    print("STEP 2: TRAINING")
    print("="*60)
    cmd_train = f'"{PYTHON_TRAIN}" train_with_efficient_tuner.py'
    output, ret_train = run_command(cmd_train)
    
    # 3. Log results
    print("\n" + "="*60)
    print("STEP 3: LOGGING RESULTS")
    print("="*60)
    
    if ret_train == 0:
        results = parse_results(output)
        status = "Success"
    else:
        print(f"Training failed with return code {ret_train}")
        results = {}
        status = "Failed"
    
    log_file = "experiment_log.csv"
    file_exists = os.path.exists(log_file)
    
    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Date', 'Pitch %', 'Symmetry %', 'Y-Diff %', 'Mouth-Open %', 'Note', 'Task A Acc', 'Task B Acc', 'Task C Acc', 'Task D Acc'])
            
            # エラー時は 'Error' を記録、または空文字
            task_a = results.get('Task A', 'Error' if status == 'Failed' else '')
            task_b = results.get('Task B', 'Error' if status == 'Failed' else '')
            task_c = results.get('Task C', 'Error' if status == 'Failed' else '')
            task_d = results.get('Task D', 'Error' if status == 'Failed' else '')
            
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                args.pitch,
                args.symmetry,
                args.y_diff,
                args.mouth_open,
                args.note,
                task_a,
                task_b,
                task_c,
                task_d
            ])
        print(f"Results logged to {log_file}")
        
        if status == 'Failed':
            print("Experiment failed.")
            sys.exit(1)
        else:
            print("Experiment completed successfully.")
        
    except Exception as e:
        print(f"Error logging results: {e}")

if __name__ == "__main__":
    main()
