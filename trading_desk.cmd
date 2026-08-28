@echo off
rem ---------------------------------------------------------------------------
rem  TradingBotV3 Trading Desk - production launcher (source launch).
rem
rem  The desk runs from SOURCE, not from dist\TradingBotV3\TradingBotV3.exe,
rem  by trader decision (2026-08-26). It started because Windows Smart App
rem  Control refused the unsigned local build; SAC read OFF on 2026-08-26 and
rem  the source launch stays production anyway. See "Frozen exe rebuild
rem  policy" in CLAUDE.md.
rem
rem  The console window is started minimized on purpose - it holds the desk's
rem  stdout/stderr, which is what you read when something misbehaves. Closing
rem  it stops the desk. For a truly windowless launch swap python.exe for
rem  pythonw.exe below, but then those logs go nowhere.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo(
    echo   ERROR: .venv\Scripts\python.exe not found in %CD%
    echo   The repo virtual environment is missing - the desk cannot start.
    echo(
    pause
    exit /b 1
)

start "TradingBotV3 Trading Desk" /min ".venv\Scripts\python.exe" "launch_gui.py"
exit /b 0
