@echo off
d:\tendency\.venv_windows_gpu\Scripts\python.exe util/analyze_calibration_log.py
echo.
echo Analysis Complete.
type outputs\analysis_calibration_result.txt
pause
