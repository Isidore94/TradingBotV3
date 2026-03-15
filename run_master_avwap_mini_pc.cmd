@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_python_script.ps1" scripts\master_avwap_mini_pc.py %*
exit /b %ERRORLEVEL%
