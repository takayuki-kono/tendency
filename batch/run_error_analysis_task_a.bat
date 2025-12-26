@echo off
cd /d %~dp0
call tendency.venv_tf210_gpu\Scripts\activate
python analyze_errors_task_a.py
pause
