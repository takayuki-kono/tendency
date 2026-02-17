import re
import os
import glob
from collections import defaultdict

LOG_FILE = "outputs/logs/sequential_opt_log.txt"

def analyze_best_epochs(log_file):
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        return

    output_file = "outputs/analysis_best_epochs_result.txt"
    with open(output_file, 'w', encoding='utf-8') as f_out:
        def print_out(s=""):
            print(s)
            f_out.write(s + "\n")

        print_out(f"Analyzing log file: {log_file}")
        
        # ... (rest of logic) ...
        # Replace all print() calls with print_out()
        # To avoid massive indentation changes, I will assign print = print_out
        
        # Re-implementing the print logic cleanly:
        f_out.write(f"Analyzing log file: {log_file}\n")

        # Split log by "Evaluat"
        trials = re.split(r"(Evaluating: Model=.*)", log_content)
        
        epoch_counts = defaultdict(int)
        epoch_1_trials = []
        epoch_20_trials = []
        
        if len(trials) > 1:
            for i in range(1, len(trials), 2):
                header = trials[i].strip()
                body = trials[i+1] if i+1 < len(trials) else ""
                params_line = header.replace("Evaluating: ", "")
                best_epoch = -1
                best_score = -1.0
                pattern = re.compile(r"Epoch (\d+)/(\d+).*?(?:Avg=|MinClassAcc=)([\d\.]+)")
                epoch_data = []
                for line in body.split('\n'):
                    match = pattern.search(line)
                    if match:
                        ep_num = int(match.group(1))
                        score = float(match.group(3))
                        epoch_data.append((ep_num, score))
                
                if epoch_data:
                    epoch_data.sort(key=lambda x: (-x[1], x[0]))
                    best_epoch, best_score = epoch_data[0]
                    epoch_counts[best_epoch] += 1
                    if best_epoch == 1:
                        epoch_1_trials.append(f"{params_line} (Score={best_score:.4f})")
                    elif best_epoch == 20:
                        epoch_20_trials.append(f"{params_line} (Score={best_score:.4f})")

        print_out("\n" + "="*50)
        print_out("Analysis Results")
        print_out("="*50)
        
        total_trials = sum(epoch_counts.values())
        print_out(f"Total Trials Analyzed: {total_trials}")
        
        if total_trials == 0:
            print_out("No valid trial data found.")
            return

        print_out("\n[Best Epoch Distribution]")
        for ep in sorted(epoch_counts.keys()):
            count = epoch_counts[ep]
            print_out(f"  Epoch {ep:2d}: {count} trials ({count/total_trials*100:5.1f}%)")
            
        print_out("\n" + "-"*50)
        print_out(f"Trials with Best Epoch = 1 ({len(epoch_1_trials)} trials)")
        print_out("Likely implies: Overfitting started immediately, or LR too high, or data too noisy/small.")
        print_out("-" * 50)
        for p in epoch_1_trials[:20]:
            print_out(f"  - {p}")
        if len(epoch_1_trials) > 20:
            print_out(f"  ... and {len(epoch_1_trials)-20} more.")

        print_out("\n" + "-"*50)
        print_out(f"Trials with Best Epoch = 20 ({len(epoch_20_trials)} trials)")
        print_out("Likely implies: Underfitting (LR too low?), or simply kept improving (Good sign if score is high).")
        print_out("-" * 50)
        for p in epoch_20_trials[:20]:
            print_out(f"  - {p}")
        if len(epoch_20_trials) > 20:
            print_out(f"  ... and {len(epoch_20_trials)-20} more.")
            
        print_out("\nAnalysis Complete.")
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    log_target = None
    
    # Check command line args
    if len(sys.argv) > 1:
        log_target = sys.argv[1]
    
    # If not provided, find latest log in default dir
    if not log_target or not os.path.exists(log_target):
        # Default directory
        base_dir = "outputs/logs"
        if not os.path.exists(base_dir) and os.path.exists(r"d:\tendency\outputs\logs"):
             base_dir = r"d:\tendency\outputs\logs"
             
        # Look for timestamped logs first
        logs = glob.glob(os.path.join(base_dir, "sequential_opt_log_*.txt"))
        # Also look for the old static log
        static_log = os.path.join(base_dir, "sequential_opt_log.txt")
        if os.path.exists(static_log):
            logs.append(static_log)
            
        if logs:
            # Sort by modification time, newest first
            logs.sort(key=os.path.getmtime, reverse=True)
            log_target = logs[0]
            print(f"Latest log file found: {log_target}")
        else:
            print("No log files found.")
            sys.exit(1)
            
    analyze_best_epochs(log_target)
