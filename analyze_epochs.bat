@echo off
echo Generating Epoch Analysis...
set LOGfile=outputs\logs\sequential_opt_log_20260218_002655.txt
if not exist "%LOGFILE%" (
    echo Log file not found: %LOGFILE%
    exit /b 1
)

echo --- BEST_EPOCH Distribution --- > outputs\epoch_summary.txt
findstr "BEST_EPOCH" "%LOGFILE%" >> outputs\epoch_summary.txt

echo.
echo Analysis of Epoch 1 and 20:
echo See outputs\epoch_summary.txt for details.
type outputs\epoch_summary.txt | findstr /C:"BEST_EPOCH: 1" 
type outputs\epoch_summary.txt | findstr /C:"BEST_EPOCH: 20"
pause
