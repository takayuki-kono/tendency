@echo off
chcp 65001 > nul
echo Adding files...
git add util/create_person_split.py docs/03_training_workflow.md preprocess_person.py components/dataset_loader.py download_and_filter_faces.py

echo Committing...
git commit -m "fix: ダウンロードキーワード追加、データ分割ロジック修正、前処理・学習の堅牢化"

echo Pushing...
git push

echo Done.
pause
