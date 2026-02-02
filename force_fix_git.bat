@echo off
cd /d d:\tendency

echo === 1. Cleanup Processes ===
taskkill /F /IM git.exe 2>nul
if exist .git\index.lock del .git\index.lock

echo.
echo === 2. Reset Staging Area (Essential!) ===
echo This un-stages any accidental massive file additions from before.
git reset

echo.
echo === 3. Selective Add ===
echo Only adding the necessary source files...
git add .gitignore
git add components/part1_setup.py
git add pipeline_specs.md

echo.
echo === 4. Commit ===
git commit -m "Fix timeouts and docs (Explicit Commit)" --author="Gemini <gemini@example.com>"

echo.
echo === 5. Push ===
git push origin master
if %errorlevel% neq 0 (
    echo Push failed. Trying simple push...
    git push
)

echo.
echo === Finished ===
pause
