@echo off
echo ========================================
echo Deleting all optimization caches...
echo ========================================

if exist outputs\cache (
    rmdir /s /q outputs\cache
    echo Cache deleted.
) else (
    echo No cache directory found.
)

mkdir outputs\cache
echo Cache directory recreated.

echo ========================================
echo Done.
echo ========================================
pause
