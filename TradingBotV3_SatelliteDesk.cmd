@echo off
REM Windows launcher for the SATELLITE DESK: the full Trading Desk UI fed by the
REM main PC's Desk Link relay instead of TWS. Alerts land in the real Alert
REM Center as if this machine were connected to the API. Pair it (or re-point
REM it) in Settings -> Desk Link -> "Connect to a main desk".
setlocal

cd /d "%~dp0"

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
)

if defined PYTHON_EXE (
    "%PYTHON_EXE%" "%~dp0scripts\gui.py" --ui qt --satellite-desk %*
    goto done
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0scripts\gui.py" --ui qt --satellite-desk %*
    goto done
)

python "%~dp0scripts\gui.py" --ui qt --satellite-desk %*

:done
if errorlevel 1 (
    echo.
    echo TradingBotV3 Satellite Desk exited with an error.
    pause
)
