
import re
import sys
import os

def analyze_log(log_path):
    output_path = "outputs/epoch_analysis.txt"
    print(f"Analyzing {log_path} -> {output_path}")
    
    with open(output_path, "w", encoding='utf-8') as out_f:
        def print_out(s=""):
            print(s)
            out_f.write(str(s) + "\n")
            
        print_out(f"Analyzing log file: {log_path}")
        
        if not os.path.exists(log_path):
            print_out(f"Error: File not found: {log_path}")
            return

        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            print_out(f"Error reading file: {e}")
            return

        trials = re.split(r"(Evaluating: Model=.*)", content)
        
        epoch_1_trials = []
        epoch_20_trials = []
        
        for i in range(1, len(trials), 2):
            header = trials[i].strip()
            body = trials[i+1] if i+1 < len(trials) else ""
            
            match = re.search(r"BEST_EPOCH:\s*(\d+)", body)
            if match:
                best_epoch = int(match.group(1))
                score_match = re.search(r"FINAL_VAL_ACCURACY:\s*([\d\.]+)", body)
                score = float(score_match.group(1)) if score_match else 0.0
                
                info = f"{header} -> Score={score:.4f}"
                
                if best_epoch == 1:
                    epoch_1_trials.append(info)
                elif best_epoch == 20:
                    epoch_20_trials.append(info)
        
        print_out("\n" + "="*50)
        print_out(f"Trials with Best Epoch = 1 ({len(epoch_1_trials)})")
        print_out("="*50)
        for t in epoch_1_trials:
            print_out(t)
            
        print_out("\n" + "="*50)
        print_out(f"Trials with Best Epoch = 20 ({len(epoch_20_trials)})")
        print_out("="*50)
        for t in epoch_20_trials:
            print_out(t)
            
        print_out(f"\nSummary: Epoch 1: {len(epoch_1_trials)}, Epoch 20: {len(epoch_20_trials)}")

if __name__ == "__main__":
    log_file = r"outputs\logs\sequential_opt_log_20260218_002655.txt"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    analyze_log(log_file)
