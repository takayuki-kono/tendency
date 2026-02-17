import re
import os
import sys

LOG_FILE = "outputs/logs/lr_scaling_calibration.txt"

def analyze_calibration_log(log_path):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    output_file = "outputs/analysis_calibration_result.txt"
    with open(output_file, 'w', encoding='utf-8') as f_out:
        def print_out(s=""):
            print(s)
            f_out.write(s + "\n")

        print_out(f"Analyzing: {log_path}")
        
        # ... (rest of logic) ...
        # Re-implementing the print logic cleanly:
        f_out.write(f"Analyzing: {log_path}\n")

        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_context = "Init"
        events = []
        
        p_summary = re.compile(r"BestEpoch=(\d+)/(\d+)")
        p_raw = re.compile(r"BEST_EPOCH:\s*(\d+)")
        
        for line in lines:
            line = line.strip()
            
            if "Optimizing" in line and "for levels" in line:
                current_context = line
            elif "Level " in line and "Ratio=" in line:
                current_context = line
            elif "Baseline: LR=" in line:
                current_context = f"Baseline Calculation ({line})"

            match = p_summary.search(line)
            if match:
                best_epoch = int(match.group(1))
                total_epochs = int(match.group(2))
                if best_epoch == 1 or best_epoch == total_epochs:
                    events.append({
                        "best_epoch": best_epoch,
                        "total_epochs": total_epochs,
                        "context": current_context,
                        "line": line
                    })
                continue

            match_raw = p_raw.search(line)
            if match_raw:
                best_epoch = int(match_raw.group(1))
                if best_epoch == 1 or best_epoch == 20: 
                     events.append({
                        "best_epoch": best_epoch,
                        "total_epochs": 20, 
                        "context": current_context,
                        "line": line
                    })

        print_out("\n" + "="*60)
        print_out(f"Analysis Results: {len(events)} suspicious trials found.")
        print_out("="*60)
        
        for i, e in enumerate(events):
            print_out(f"#{i+1}: BestEpoch={e['best_epoch']}/{e['total_epochs']}")
            print_out(f"  Context: {e['context']}")
            print_out(f"  LogLine: {e['line']}")
            print_out("-" * 40)
            
        if not events:
            print_out("No trials with BestEpoch = 1 or Max found. Calibration looks stable.")
            
    print(f"Done. Result saved to {output_file}")

if __name__ == "__main__":
    target = LOG_FILE
    if len(sys.argv) > 1:
        target = sys.argv[1]
    
    analyze_calibration_log(target)
