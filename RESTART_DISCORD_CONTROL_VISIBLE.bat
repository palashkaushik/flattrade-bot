@echo off
setlocal
cd /d "%~dp0"
title Flattrade Discord Control - Visible Verification

echo ================================================
echo   Flattrade Discord Controller
echo   Visible verification mode
echo ================================================
echo.

echo Stopping the scheduled Discord controller...
schtasks /End /TN "\Flattrade Discord Control" >nul 2>&1

choice /C YN /N /M "Stop the current bot so the next start gets a visible bot window? [Y/N] "
if errorlevel 2 goto controller_only

echo Requesting a graceful bot stop...
python -c "from pathlib import Path; from flattrade_bot.control import TradingProcessManager; m=TradingProcessManager(Path.cwd()); print('Stop requested:', m.request_stop()); print('Exited:', m.wait_for_exit(45.0))"
if errorlevel 1 echo Bot stop request could not be completed; inspect logs before starting another bot.

:controller_only
echo.
echo Starting the Discord controller in this visible terminal.
echo Keep this window open while using Discord commands.
echo Press Ctrl+C to end this verification session.
echo.
python -m flattrade_bot.discord_control
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo Restoring the scheduled Discord controller...
schtasks /Run /TN "\Flattrade Discord Control" >nul 2>&1
if errorlevel 1 echo Could not restart the scheduled task; run START_DISCORD_CONTROL.bat manually.

endlocal & exit /b %EXIT_CODE%
