@echo off
REM 破損画像ファイルの検出バッチ
REM 使用法: check_images.bat [ディレクトリ]
REM デフォルトは preprocessed_multitask

set TARGET_DIR=%1
if "%TARGET_DIR%"=="" set TARGET_DIR=preprocessed_multitask

echo ==========================================
echo 破損画像チェック: %TARGET_DIR%
echo ==========================================

d:\tendency\.venv_windows_gpu\Scripts\python.exe -c "
import os, sys
from PIL import Image

target = sys.argv[1]
bad_files = []
total = 0

for root, dirs, files in os.walk(target):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            total += 1
            fpath = os.path.join(root, f)
            try:
                img = Image.open(fpath)
                img.verify()
            except Exception as e:
                bad_files.append((fpath, str(e)))

print(f'Total images scanned: {total}')
if bad_files:
    print(f'BAD FILES FOUND: {len(bad_files)}')
    for path, err in bad_files:
        print(f'  {path} -> {err}')
else:
    print('All images OK!')
" %TARGET_DIR%

pause
