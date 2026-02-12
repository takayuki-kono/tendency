@echo off
setlocal
set PYTHON_EXEC=d:\tendency\tendency.venv_tf210_gpu\Scripts\python.exe

echo Checking train_sequential.py...
"%PYTHON_EXEC%" -m py_compile train_sequential.py
if errorlevel 1 goto :error

echo Checking train_multitask_trial.py...
"%PYTHON_EXEC%" -m py_compile components/train_multitask_trial.py
if errorlevel 1 goto :error

echo Checking train_single_trial.py...
"%PYTHON_EXEC%" -m py_compile components/train_single_trial.py
if errorlevel 1 goto :error

echo Checking common.py...
"%PYTHON_EXEC%" -m py_compile components/common.py
if errorlevel 1 goto :error

echo Checking model_factory.py...
"%PYTHON_EXEC%" -m py_compile components/model_factory.py
if errorlevel 1 goto :error

echo All syntax checks passed!
exit /b 0

:error
echo Syntax check failed!
exit /b 1
