@echo off
set /p TARGET_DIR="Enter directory to classify: "
d:\tendency\.venv_windows_gpu\Scripts\python.exe d:\tendency\predict_svm.py --input_dir "%TARGET_DIR%" --output_dir "outputs/svm_predictions"
pause
