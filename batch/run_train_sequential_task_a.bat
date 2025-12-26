@echo off
cd /d %~dp0
call tendency.venv_tf210_gpu\Scripts\activate
python train_sequential_task_a.py
pause
