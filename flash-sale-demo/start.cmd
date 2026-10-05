@echo off
rem Start the three databases and load the shared dataset.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\demo.ps1" start
echo.
pause
