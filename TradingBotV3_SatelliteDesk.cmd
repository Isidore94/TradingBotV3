@echo off
REM Compatibility shortcut only. launch_gui.py is the one supported entrypoint;
REM choose Main/Satellite inside Settings -> Desk Link.
setlocal

cd /d "%~dp0"

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
)

if defined PYTHON_EXE (
    "%PYTHON_EXE%" "%~dp0launch_gui.py" --desk-role satellite %*
    goto done
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0launch_gui.py" --desk-role satellite %*
    goto done
)

python "%~dp0launch_gui.py" --desk-role satellite %*

:done
if errorlevel 1 (
    echo.
    echo TradingBotV3 Satellite Desk exited with an error.
    pause
)
