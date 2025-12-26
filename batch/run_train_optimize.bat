@echo off
cd /d D:\tendency
echo Starting Training Optimization (Sequential)...
start /belownormal /wait d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe train_optimize_sequential.py
pause
