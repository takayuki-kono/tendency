@echo off
cd /d D:\tendency
echo Starting Preprocessing with Optimized Filters...
echo Pitch: 25%%, Symmetry: 25%%, Y-Diff: 50%%, Mouth-Open: 0%%
d:\tendency\.venv_windows_gpu\Scripts\python.exe preprocess_multitask.py --pitch_percentile 25 --symmetry_percentile 25 --y_diff_percentile 50 --mouth_open_percentile 0
pause
