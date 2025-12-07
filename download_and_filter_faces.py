import subprocess
import sys
import os
import argparse
import datetime

# --- Configuration ---
# Default list of keywords to process if no CLI arguments are provided
DEFAULT_KEYWORDS = ["ゴミ箱"]

# --- Main Orchestrator ---
def main():
    parser = argparse.ArgumentParser(description="Image Processing Pipeline Orchestrator")
    parser.add_argument("keywords", nargs="*", help="Keywords to process. If empty, uses the internal DEFAULT_KEYWORDS list.")
    args = parser.parse_args()

    # Hybrid approach: Use CLI args if provided, otherwise fallback to internal list
    keywords_to_process = args.keywords if args.keywords else DEFAULT_KEYWORDS

    print("--- Starting Image Processing Pipeline ---")
    print(f"Target Keywords: {keywords_to_process}")
    
    python_executable = sys.executable

    for keyword in keywords_to_process:
        print(f"\n{'='*20}\nProcessing keyword: {keyword}\n{'='*20}")
        
        # Generate a deterministic directory name based on keyword and timestamp
        # Format: output_<keyword>_<YYYYMMDD>
        # Sanitizing keyword to be safe for directory names
        safe_keyword = "".join(c for c in keyword if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        output_dir = f"output_{safe_keyword}_{timestamp}"
        
        # Ensure unique output dir if multiple runs happen on same day (optional, but good for safety)
        # If you prefer strictly one folder per day per keyword, you can remove this check or handle it differently.
        # For now, let's keep it simple: if it exists, we use it (scripts inside should handle overwrite/append if needed)
        # or we can append a counter if we want fresh runs. 
        # Let's stick to the simple deterministic name for now as requested.
        
        print(f"Output will be in directory: {output_dir}")

        # Define the sequence of scripts to run for this keyword
        scripts_to_run = [
            ("part1_setup.py", [keyword, output_dir]),
            ("part2a_similarity.py", [output_dir]),
            ("part2b_filter.py", [output_dir])
        ]

        for i, (script_name, args) in enumerate(scripts_to_run, 1):
            print(f"\n[Step {i}/{len(scripts_to_run)}] Running {script_name} for '{keyword}'...")
            
            try:
                command = [python_executable, script_name] + args
                process = subprocess.run(
                    command,
                    capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore'
                )
                
                # Show output only if there's an error, to keep the main log clean
                if process.returncode != 0:
                    print(f"--- {script_name} STDOUT ---\n" + process.stdout)
                    print(f"--- {script_name} STDERR ---\n" + process.stderr, file=sys.stderr)
                    print(f"\nError: {script_name} failed for keyword '{keyword}'. Halting pipeline.", file=sys.stderr)
                    sys.exit(1)
                else:
                    # Optionally print a snippet of the log for confirmation
                    print(f"{script_name} completed successfully.")

            except FileNotFoundError:
                print(f"Error: Could not find the script '{script_name}'.", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred while running {script_name}: {e}", file=sys.stderr)
                sys.exit(1)

    print("\n--- All keywords processed. Image Processing Pipeline Finished ---")

if __name__ == "__main__":
    main()
