@echo off
rem Reload the data into the running containers (resets the demo).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\demo.ps1" seed
echo.
pause
