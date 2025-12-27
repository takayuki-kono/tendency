import os
import sys
import shutil
import logging

# Setup output directories
LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'log_pipeline.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Configuration
KEYWORDS = [
    "奈緒"
    # , "黒谷友香", "比嘉愛美", "石野真子", "戸田恵梨香", "菅野美穂", "宮沢りえ", "中山美穂", 
    # "髙橋ひかる", "桜井日奈子", "松本まりか", "松下由樹", "市川由衣", "新川優愛", "堀田真由", 
    # "国仲涼子", "黒島結菜", "大塚寧々", "鈴木京香", "森口瑤子", "中村アン", "大島優子", 
    # "上白石萌歌", "賀来千香子", "葵わかな", "薬師丸ひろ子", "門脇麦", "秋田汐梨", "武井咲", 
    # "吉高由里子", "川栄李奈", "中条あやみ", "観月ありさ", "杏", "木村文乃", "清原果耶", 
    # "稲森いずみ", "西野七瀬", "土屋太鳳", "石原さとみ", "多部未華子", "伊藤沙莉", "今井美樹", 
    # "和久井映見", "本田翼", "片桐はいり", "仲里依紗", "高橋メアリージュン", "内田理央", 
    # "大原櫻子", "小雪", "安藤玉恵", "石井杏奈", "安田成美", "橋本愛"
] 
BASE_OUTPUT_DIR = "master_data"

def run_script(script_path, args):
    """Runs a python script with arguments."""
    command = f"python {script_path} {' '.join(args)}"
    logger.info(f"Running command: {command}")
    ret = os.system(command)
    if ret != 0:
        logger.error(f"Command failed: {command}")
        return False
    return True

def main():
    sys.path.append(os.getcwd())
    
    # Scripts
    part1_script = os.path.join("components", "part1_setup.py")
    part2a_script = os.path.join("components", "part2a_similarity.py")
    part2b_script = os.path.join("components", "part2b_filter.py")

    for keyword in KEYWORDS:
        logger.info(f"Processing keyword: {keyword}")
        output_dir = os.path.join(BASE_OUTPUT_DIR, keyword)
        
        # Step 1: Run Part 1 (Download, Detect, Crop)
        # Part 1 creates: output_dir/rotated/
        if not run_script(part1_script, [keyword, output_dir]):
            logger.error("Part 1 failed.")
            continue
            
        # Part 1 output is now: output_dir/rotated/
        rotated_dir = os.path.join(output_dir, "rotated")

        # Step 2: Run Part 2a & 2b for 'rotated'
        logger.info(f"--- Processing rotated pipeline ---")
        if not run_script(part2a_script, [rotated_dir]):
            logger.error("Part 2a failed.")
        if not run_script(part2b_script, [rotated_dir]):
            logger.error("Part 2b failed.")
        
        logger.info(f"Pipeline finished for {keyword}")

if __name__ == "__main__":
    main()
