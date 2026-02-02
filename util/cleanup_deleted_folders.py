import os
import shutil

# Target root directory
TARGET_ROOT = r"d:\tendency\master_data"

def delete_deleted_folders(root_dir):
    print(f"Scanning {root_dir}...")
    count = 0
    for root, dirs, files in os.walk(root_dir):
        # Must act on a copy to safely modify list during iteration
        for d in list(dirs):
            if d.startswith("deleted_"):
                full_path = os.path.join(root, d)
                try:
                    print(f"Removing: {full_path}")
                    shutil.rmtree(full_path)
                    count += 1
                    # Remove from dirs list so we don't traverse into deleted folder
                    dirs.remove(d) 
                except Exception as e:
                    print(f"Failed to remove {full_path}: {e}")
    print(f"Finished. Removed {count} folders.")

if __name__ == "__main__":
    delete_deleted_folders(TARGET_ROOT)
