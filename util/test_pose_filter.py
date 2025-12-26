import os
import sys
import logging
from insightface.app import FaceAnalysis
from apply_filter_to_dataset import filter_folder, DET_SIZE, PROVIDERS

# Setup logging to console
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

def main():
    target_dir = r"D:\tendency\train\aefh\harukaFukuhara"
    
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    print(f"Testing filter on: {target_dir}")
    
    # Initialize App
    app = FaceAnalysis(providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    
    # Run filter
    filter_folder(target_dir, app)
    
    print("Test complete. Check 'deleted' folder.")

if __name__ == "__main__":
    main()
