@echo off
echo Starting Gemini session in WSL...
wsl.exe -e bash -c "cd /mnt/d/tendency && ./start_gemini_logged.sh"
echo.
echo Session has ended. The window will close after you press any key.
pause
