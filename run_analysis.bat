@echo off
echo Starting analysis...
d:\tendency\.venv_windows_gpu\Scripts\python.exe util/analyze_log_best_epochs.py
echo.
echo Analysis finished. Results saved to outputs/analysis_best_epochs_result.txt
echo.
type outputs\analysis_best_epochs_result.txt
pause
