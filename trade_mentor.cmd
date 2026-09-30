@echo off
rem ---------------------------------------------------------------------------
rem  TradingBotV3 Trade Mentor - its own window, its own process (source launch).
rem  A second launch brings the running window to the front. pythonw keeps the
rem  app windowless; its log goes to the TradingBotV3 logs folder.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo(
    echo   ERROR: .venv\Scripts\pythonw.exe not found in %CD%
    echo   The repo virtual environment is missing - the Trade Mentor cannot start.
    echo(
    pause
    exit /b 1
)

start "TradingBotV3 Trade Mentor" ".venv\Scripts\pythonw.exe" "launch_mentor.py"
exit /b 0
