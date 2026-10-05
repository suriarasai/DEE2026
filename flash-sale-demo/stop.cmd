@echo off
rem Remove the three demo containers.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\demo.ps1" stop
echo.
pause
