@echo off
echo Starting analysis... > outputs/analysis_debug.txt
d:\tendency\.venv_windows_gpu\Scripts\python.exe util/analyze_log_best_epochs.py >> outputs/analysis_debug.txt 2>&1
echo Analysis finished. >> outputs/analysis_debug.txt
