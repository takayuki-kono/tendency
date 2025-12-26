@echo off
cd /d %~dp0
call tendency.venv_tf210_gpu\Scripts\activate
python analyze_errors.py --model best_sequential_model.keras
pause
