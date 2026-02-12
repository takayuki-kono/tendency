@echo off
if not exist archive_single mkdir archive_single

move train_sequential_single.py archive_single\
move run_optimization_single.bat archive_single\
move preprocess_single.py archive_single\
move optimize_sequential_single.py archive_single\

move components\train_single_trial.py archive_single\
move components\train_single_trial_task_a.py archive_single\
move components\train_filter_single_trial.py archive_single\

move util\analyze_errors_single.py archive_single\

if exist preprocessed_single move preprocessed_single archive_single\
if exist error_analysis_single move error_analysis_single archive_single\

echo DONE > move_log.txt
