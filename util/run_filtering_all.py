import os
import sys
import argparse
import subprocess
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path, args):
    """Runs a python script with arguments using the current python interpreter."""
    command = [sys.executable, script_path] + args
    logger.info(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Part2A (Similarity) and Part2B (Filter) on all master_data folders.")
    parser.add_argument("--master_dir", default="master_data", help="Root directory of master data")
    parser.add_argument("--physical_delete", action="store_true", help="Enable physical deletion (default: Move to deleted folder)")
    args = parser.parse_args()

    part2a_script = os.path.join("components", "part2a_similarity.py")

    if not os.path.exists(part2a_script):
        logger.error("Component script (part2a_similarity.py) not found. Please run from project root.")
        return

    master_dir = os.path.abspath(args.master_dir)
    if not os.path.exists(master_dir):
        logger.error(f"Master directory not found: {master_dir}")
        return

    # Folder Structure: master_data / label / person_name / (rotated/) *.jpg
    # We need to iterate: label -> person -> find images
    
    processed_count = 0
    
    for label_name in os.listdir(master_dir):
        label_path = os.path.join(master_dir, label_name)
        if not os.path.isdir(label_path):
            continue
            
        for person_name in os.listdir(label_path):
            person_path = os.path.join(label_path, person_name)
            if not os.path.isdir(person_path):
                continue
            
            # Target directory logic: Prefer 'rotated', else use person dir itself
            target_dir = os.path.join(person_path, "rotated")
            if not os.path.exists(target_dir):
                target_dir = person_path
            
            # Check if images exist
            imgs = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not imgs:
                logger.debug(f"No images in {label_name}/{person_name}. Skipping.")
                continue

            logger.info(f"Processing: {label_name}/{person_name} ({len(imgs)} images)")

            script_args = [target_dir]
            if args.physical_delete:
                script_args.append("--physical_delete")

            # Run Part 2a (Similarity) のみ
            # part2b はクリーンアップ処理が master_data 構造と互換性がないためスキップ
            if not run_script(part2a_script, script_args):
                logger.error(f"Part 2a failed for {label_name}/{person_name}")
            
            processed_count += 1

    logger.info(f"All processing finished. Processed {processed_count} folders.")

if __name__ == "__main__":
    main()
