import os
import shutil

master = r'd:\tendency\master_data'
for keyword in os.listdir(master):
    kwd_path = os.path.join(master, keyword)
    if not os.path.isdir(kwd_path):
        continue
    
    # Old location: crop_eyebrow/rotated/
    old_rotated = os.path.join(kwd_path, 'crop_eyebrow', 'rotated')
    # New location: rotated/
    new_rotated = os.path.join(kwd_path, 'rotated')
    
    if os.path.exists(old_rotated):
        # Move contents
        if not os.path.exists(new_rotated):
            shutil.move(old_rotated, new_rotated)
            print(f'Moved {old_rotated} -> {new_rotated}')
        else:
            # Merge
            for f in os.listdir(old_rotated):
                src = os.path.join(old_rotated, f)
                dst = os.path.join(new_rotated, f)
                if os.path.isfile(src):
                    shutil.move(src, dst)
            shutil.rmtree(old_rotated)
            print(f'Merged {old_rotated} into {new_rotated}')
    
    # Delete crop_eyebrow folder
    crop_eyebrow = os.path.join(kwd_path, 'crop_eyebrow')
    if os.path.exists(crop_eyebrow):
        shutil.rmtree(crop_eyebrow)
        print(f'Deleted {crop_eyebrow}')
    
    # Delete loose img files at keyword level
    for f in os.listdir(kwd_path):
        fp = os.path.join(kwd_path, f)
        if os.path.isfile(fp):
            os.remove(fp)
            
print('Done.')
