@echo off
echo Splitting people into Train/Validation...
python util/create_person_split.py
echo Done.
pause
