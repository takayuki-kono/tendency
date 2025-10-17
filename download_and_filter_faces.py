import subprocess
import sys
import os
import random

# --- Configuration ---
# Edit this list to process multiple keywords
KEYWORDS = ["石井杏奈", "和久井映見"]

# --- Main Orchestrator ---
def main():
    print("--- Starting Image Processing Pipeline ---")
    python_executable = sys.executable

    for keyword in KEYWORDS:
        print(f"\n{'='*20}\nProcessing keyword: {keyword}\n{'='*20}")
        
        # Generate a unique directory for this keyword's results
        output_dir = str(random.randint(0, 9999)).zfill(4)
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
