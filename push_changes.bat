@echo off
cd /d d:\tendency

if exist .git\index.lock (
    echo Deleting leftover index.lock...
    del .git\index.lock
)

echo Adding files...
git add .gitignore components/part1_setup.py pipeline_specs.md

echo Committing...
git commit -m "Update part1_setup.py (timeout) and docs, fix gitignore" --author="Gemini <gemini@example.com>"

echo Pushing...
git push origin master
if %errorlevel% neq 0 (
  echo Push failed, trying default push...
  git push
)
echo Done.
