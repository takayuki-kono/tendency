@echo off
chcp 65001 >nul
echo =======================================
echo  person → single リネームスクリプト
echo =======================================
echo.

cd /d d:\tendency

echo [Step 1] ファイルのリネーム...

REM メインスクリプト
if exist "optimize_sequential_person.py" (
    ren "optimize_sequential_person.py" "optimize_sequential_single.py"
    echo   optimize_sequential_person.py → optimize_sequential_single.py
)

if exist "preprocess_person.py" (
    ren "preprocess_person.py" "preprocess_single.py"
    echo   preprocess_person.py → preprocess_single.py
)

if exist "train_sequential_person.py" (
    ren "train_sequential_person.py" "train_sequential_single.py"
    echo   train_sequential_person.py → train_sequential_single.py
)

REM utilフォルダ
if exist "util\analyze_errors_person.py" (
    ren "util\analyze_errors_person.py" "analyze_errors_single.py"
    echo   util\analyze_errors_person.py → analyze_errors_single.py
)

if exist "util\create_person_split.py" (
    ren "util\create_person_split.py" "create_single_split.py"
    echo   util\create_person_split.py → create_single_split.py
)

REM componentsフォルダ
if exist "components\train_person_trial.py" (
    ren "components\train_person_trial.py" "train_single_trial.py"
    echo   components\train_person_trial.py → train_single_trial.py
)

if exist "components\train_filter_person_trial.py" (
    ren "components\train_filter_person_trial.py" "train_filter_single_trial.py"
    echo   components\train_filter_person_trial.py → train_filter_single_trial.py
)

echo.
echo [Step 2] ディレクトリのリネーム...

if exist "preprocessed_person" (
    ren "preprocessed_person" "preprocessed_single"
    echo   preprocessed_person\ → preprocessed_single\
)

if exist "error_analysis_person" (
    ren "error_analysis_person" "error_analysis_single"
    echo   error_analysis_person\ → error_analysis_single\
)

echo.
echo =======================================
echo  リネーム完了！
echo  ※ファイル内の参照は別途更新が必要です
echo =======================================
pause
