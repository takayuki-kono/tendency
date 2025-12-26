@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo Starting Batch Experiments (Direct Execution)
echo ========================================================

:: Run batch script
echo.
echo Running batch script...
python run_batch.py
if errorlevel 1 (
    echo Error occurred in batch script
    goto :error
)

echo.
echo ========================================================
echo All experiments completed successfully!
echo ========================================================
goto :end

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo Batch execution failed!
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
exit /b 1

:end
exit /b 0
