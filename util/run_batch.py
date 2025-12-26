import subprocess
import sys
import time
import itertools
import ctypes

values = [0, 25, 50, 75]
combinations = list(itertools.product(values, values))

print(f"Starting batch execution of {len(combinations)} experiments...")

for i, (pitch, symmetry) in enumerate(combinations):
    print(f"\n{'='*60}")
    print(f"Experiment {i+1}/{len(combinations)}: Pitch={pitch}, Symmetry={symmetry}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    cmd = [
        sys.executable, 
        "run_experiment.py", 
        "--pitch", str(pitch), 
        "--symmetry", str(symmetry), 
        "--note", f"Batch run P{pitch} S{symmetry}"
    ]
    
    try:
        # Run and stream output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            encoding='utf-8', 
            errors='replace'
        )
        
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                try:
                    print(line.strip())
                except UnicodeEncodeError:
                    print(line.strip().encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
        
        if process.returncode != 0:
            print(f"Experiment failed with return code {process.returncode}")
            # エラー時にメッセージボックスを表示
            # 0x10 = MB_ICONERROR, 0x0 = MB_OK
            ctypes.windll.user32.MessageBoxW(0, f"Experiment failed!\nPitch={pitch}, Symmetry={symmetry}\nReturn Code: {process.returncode}", "Batch Execution Error", 0x10)
        else:
            elapsed = time.time() - start_time
            print(f"Experiment finished in {elapsed:.1f} seconds.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        ctypes.windll.user32.MessageBoxW(0, f"An error occurred during batch execution:\n{e}", "Batch Execution Exception", 0x10)

print("\n" + "="*60)
print("All batch experiments completed.")
print("="*60)
