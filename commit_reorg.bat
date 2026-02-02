@echo off
cd /d d:\tendency

echo Adding files...
git add util/reorganize_by_label.py run_reorganize.bat

echo Committing...
git commit -m "feat: ラベル振り分けスクリプトの女優マッピングを更新"

echo Done. Check output for errors.
pause
