@echo off
set "CACHE_FILE=outputs\cache\calibrated_lr.json"

if exist "%CACHE_FILE%" (
    del "%CACHE_FILE%"
    echo [INFO] Deleted: %CACHE_FILE%
) else (
    echo [INFO] File not found: %CACHE_FILE%
)

pause
