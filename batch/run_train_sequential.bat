@echo off
cd /d D:\tendency
echo Starting Sequential Training Optimization (low priority)...
start /belownormal /wait d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe train_sequential.py
pause
