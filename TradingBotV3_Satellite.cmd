@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
)

if defined PYTHON_EXE (
    "%PYTHON_EXE%" "%~dp0scripts\gui.py" --ui qt --satellite %*
    goto done
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0scripts\gui.py" --ui qt --satellite %*
    goto done
)

python "%~dp0scripts\gui.py" --ui qt --satellite %*

:done
if errorlevel 1 (
    echo.
    echo TradingBotV3 Satellite exited with an error.
    pause
)
