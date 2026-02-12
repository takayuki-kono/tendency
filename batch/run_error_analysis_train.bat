@echo off
cd /d %~dp0

echo Running Error Analysis on Training Data...
d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe util/analyze_errors.py --data_dir preprocessed_multitask/train --out_dir error_analysis_train

echo Done.
pause
