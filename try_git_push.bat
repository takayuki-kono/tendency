@echo off
cd /d d:\tendency
echo ---------------------------------------------------
echo Executing 'git push' to remote...
echo If this screen freezes for more than 2 minutes, 
echo please close this window (Ctrl+C or X button).
echo ---------------------------------------------------

git push origin master

if %errorlevel% neq 0 (
    echo.
    echo 'git push origin master' failed.
    echo Trying simple 'git push'...
    git push
)

echo.
echo ---------------------------------------------------
echo Command finished.
echo ---------------------------------------------------
pause
