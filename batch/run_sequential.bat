@echo off
cd /d D:\tendency
echo Starting Sequential Optimization (low priority)...
start /belownormal /wait d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe optimize_sequential.py
pause
