@echo off
chcp 65001 >nul
echo =======================================
echo  componentsフォルダのファイル整理
echo =======================================
echo.

cd /d d:\tendency\components

REM 既存のtrain_single_trial.pyをバックアップ（マルチタスク用）
if exist "train_single_trial.py" (
    ren "train_single_trial.py" "train_multitask_trial.py"
    echo   train_single_trial.py → train_multitask_trial.py (backup)
)

REM train_person_trial.pyをtrain_single_trial.pyにリネーム
if exist "train_person_trial.py" (
    ren "train_person_trial.py" "train_single_trial.py"
    echo   train_person_trial.py → train_single_trial.py
)

REM train_filter_person_trial.pyも確認
if exist "train_filter_person_trial.py" (
    ren "train_filter_person_trial.py" "train_filter_single_trial.py"
    echo   train_filter_person_trial.py → train_filter_single_trial.py
)

echo.
echo 完了！
pause
