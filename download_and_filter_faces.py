import os
import sys
import shutil
import logging

# Setup output directories
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "log_pipeline_v2.txt"), mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
KEYWORDS = [
    "有村架純",
] 
BASE_OUTPUT_DIR = "master_data"
PHYSICAL_DELETE = True # True: Permanently delete, False: Move to 'deleted' folder

import subprocess

def run_script(script_path, args):
    """Runs a python script with arguments using the current python interpreter."""
    # Use sys.executable to ensure we use the same python environment (e.g. venv)
    command = [sys.executable, script_path] + args
    
    logger.info(f"Running command: {' '.join(command)}")
    try:
        # subprocess.run is safer and allows capturing output if needed
        result = subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}: {' '.join(command)}")
        return False
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return False

def main():
    sys.path.append(os.getcwd())
    
    # Scripts
    part1_script = os.path.join("components", "part1_setup.py")
    part2a_script = os.path.join("components", "part2a_similarity.py")
    part2b_script = os.path.join("components", "part2b_filter.py")

    for keyword in KEYWORDS:
        logger.info(f"Processing keyword: {keyword}")
        output_dir = os.path.join(BASE_OUTPUT_DIR, keyword)

        # "Refresh" - If directory exists, delete it for a clean start
        if os.path.exists(output_dir):
            logger.info(f"Removing existing directory for fresh start: {output_dir}")
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                logger.warning(f"Failed to remove {output_dir}: {e}")
        
        # Step 1: Run Part 1 (Download, Detect, Crop)
        # Part 1 creates: output_dir/rotated/
        if not run_script(part1_script, [keyword, output_dir]):
            logger.error("Part 1 failed.")
            continue
            
        # Step 1.5: Process any remaining raw files in output_dir
        # This catches files that might have been downloaded but not processed in previous runs
        # or if they were manually placed there.
        process_local_script = os.path.join("components", "process_local_raw.py")
        logger.info(f"Processing any local raw files in {output_dir}")
        run_script(process_local_script, [output_dir])

        # Part 1 output is now: output_dir/rotated/
        rotated_dir = os.path.join(output_dir, "rotated")

        # Step 2: Run Part 2a & 2b for 'rotated'
        logger.info(f"--- Processing rotated pipeline ---")
        
        # Prepare arguments for deletion mode
        delete_args = [rotated_dir]
        if PHYSICAL_DELETE:
            delete_args.append("--physical_delete")

        if not run_script(part2a_script, delete_args):
            logger.error("Part 2a failed.")
        if not run_script(part2b_script, delete_args):
            logger.error("Part 2b failed.")
        
        logger.info(f"Pipeline finished for {keyword}")

if __name__ == "__main__":
    main()
