@echo off
echo Splitting people into Train/Validation...
python util/create_single_split.py
echo Done.
pause
