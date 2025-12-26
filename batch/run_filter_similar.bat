@echo off
cd /d D:\tendency
echo Starting Similar Image Filter on Preprocessed Data...
echo Tolerance: 0.26 (cosine distance)
echo.
d:\tendency\.venv_windows_gpu\Scripts\python.exe filter_similar_preprocessed.py --tolerance 0.26
pause
